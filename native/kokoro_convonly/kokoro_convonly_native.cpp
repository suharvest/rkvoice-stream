#include "kokoro_convonly_native.h"
#include "kokoro_rk3588_bridge.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cstdio>
#include <sys/utsname.h>

using Clock=std::chrono::steady_clock;
constexpr uint32_t C=128, TILE=8192, OUT_C=22;
constexpr int KERNELS[3]={3,7,11}, DILATIONS[3]={1,3,5}, MASKS[3]={1,2,1};

struct Fixture { uint32_t n{}; std::vector<float>x,gamma,beta,slopes; };
struct Slice { uint32_t i0,i1,o0,o1; };
struct Model {
  kokoro_rk3588_handle*h{}; uint32_t in_n{},out_n{};
  std::string path; kokoro_rk3588_contract contract{};
  Model()=default;
  Model(const Model&)=delete;
  Model& operator=(const Model&)=delete;
  Model(Model&& other) noexcept : h(other.h),in_n(other.in_n),out_n(other.out_n),
      path(std::move(other.path)),contract(other.contract) { other.h=nullptr; }
  Model& operator=(Model&& other) noexcept {
    if(this!=&other){reset();h=other.h;other.h=nullptr;in_n=other.in_n;
      out_n=other.out_n;path=std::move(other.path);contract=other.contract;}
    return *this;
  }
  void reset() noexcept {if(h){kokoro_rk3588_destroy(h);h=nullptr;}}
  ~Model(){reset();}
};
struct Metric { std::string name; double maxabs{},rel{},cos{}; };
struct BranchScratch { std::vector<float> y,residual,z,tile,out,next; };

static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static std::vector<Slice> schedule(uint32_t n,uint32_t halo){
  if(n<=TILE)return{{0,n,0,n}}; const uint32_t core=TILE-2*halo; uint32_t out=0; std::vector<Slice>v;
  while(n-out>core){uint32_t end=out+core,start=out?out-halo:0,input_end=std::min(n,end+halo);v.push_back({start,input_end,out-start,out-start+std::min(core,n-out)});out=end;}
  uint32_t start=n-TILE;v.push_back({start,n,out-start,n-start});return v;
}
static Model load_model(const std::string&p,int mask,uint32_t want_in,uint32_t want_out){
  Model m;m.path=p;m.h=kokoro_rk3588_create(p.c_str(),mask);if(!m.h)throw std::runtime_error("create failed: "+p);kokoro_rk3588_contract c{};
  if(kokoro_rk3588_query_contract(m.h,&c)||c.n_input!=1||c.n_output!=1||c.input_floats[0]!=want_in||c.output_floats!=want_out)throw std::runtime_error("contract mismatch: "+p);
  m.in_n=c.input_floats[0];m.out_n=c.output_floats;m.contract=c;return m;
}
static Model load_merge(const std::string&p){
  Model m;m.path=p;m.h=kokoro_rk3588_create(p.c_str(),0);if(!m.h)throw std::runtime_error("create failed: "+p);kokoro_rk3588_contract c{};
  if(kokoro_rk3588_query_contract(m.h,&c)||c.n_input!=3||c.n_output!=1||c.output_floats!=OUT_C*TILE)throw std::runtime_error("merge contract mismatch: "+p);
  for(int i=0;i<3;++i){
    if(c.input_floats[i]!=C*TILE||c.input_ndims[i]!=4||c.input_dims[i][0]!=1)
      throw std::runtime_error("merge input contract mismatch: "+p);
    bool nhwc=c.input_fmt[i]==1&&c.input_dims[i][1]==1&&c.input_dims[i][2]==TILE&&c.input_dims[i][3]==C;
    bool nchw=c.input_fmt[i]==0&&c.input_dims[i][1]==C&&c.input_dims[i][2]==1&&c.input_dims[i][3]==TILE;
    if(!nhwc&&!nchw)throw std::runtime_error("unsupported merge input layout: "+p);
  }
  if(c.output_ndims!=4||c.output_dims[0]!=1)throw std::runtime_error("merge output rank/batch mismatch: "+p);
  bool nhwc=c.output_fmt==1&&c.output_dims[1]==1&&c.output_dims[2]==TILE&&c.output_dims[3]==OUT_C;
  bool nchw=c.output_fmt==0&&c.output_dims[1]==OUT_C&&c.output_dims[2]==1&&c.output_dims[3]==TILE;
  if(!nhwc&&!nchw)throw std::runtime_error("unsupported merge output layout: "+p);
  m.in_n=c.input_floats[0];m.out_n=c.output_floats;m.contract=c;return m;
}
static void run(Model&m,const float*input,float*output){const float*ins[6]={input};uint32_t sizes[6]={m.in_n};int rc=kokoro_rk3588_run_float32(m.h,ins,sizes,output,m.out_n);if(rc)throw std::runtime_error(m.path+": "+kokoro_rk3588_last_error(m.h));}

// NumPy's contiguous float32 reduction uses recursive pairwise summation with
// an eight-lane base case. Reproducing it avoids crossing INT8 quantization
// boundaries merely because the host language changed.
template<bool SquaredDiff=false>
static float np_pairwise(const float*p,uint32_t n,float mean=0){
  if(n<8){float r=-0.0f;for(uint32_t i=0;i<n;++i){float v=p[i]-mean;r+=SquaredDiff?v*v:p[i];}return r;}
  if(n<=128){float r[8];for(int j=0;j<8;++j){float v=p[j]-mean;r[j]=SquaredDiff?v*v:p[j];}uint32_t i=8,stop=n-(n%8);
    for(;i<stop;i+=8)for(int j=0;j<8;++j){float v=p[i+j]-mean;r[j]+=SquaredDiff?v*v:p[i+j];}
    float out=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
    for(;i<n;++i){float v=p[i]-mean;out+=SquaredDiff?v*v:p[i];}return out;}
  uint32_t n2=n/2;n2-=n2%8;return np_pairwise<SquaredDiff>(p,n2,mean)+np_pairwise<SquaredDiff>(p+n2,n-n2,mean);
}
static void film_prelu(const float*y,float*z,uint32_t n,const float*g,const float*b,const float*s){
  for(uint32_t ch=0;ch<C;++ch){const float*p=y+(size_t)ch*n;float mean=np_pairwise(p,n)/(float)n;float vs=np_pairwise<true>(p,n,mean);float inv=1.0f/std::sqrt(vs/(float)n+1e-5f);float scale=1.0f+g[ch],bias=b[ch],slope=s[ch];float*o=z+(size_t)ch*n;
    #pragma GCC ivdep
    for(uint32_t t=0;t<n;++t){float v=(p[t]-mean)*inv*scale+bias;o[t]=v>=0?v:v*slope;}
  }
}
static Metric compare_file(const std::string&name,const std::string&p,const float*x,size_t count){
  std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)return{name,-1,-1,-1};if((size_t)f.tellg()!=count*sizeof(float))throw std::runtime_error("parity size mismatch: "+p);f.seekg(0);std::vector<float>r(count);f.read((char*)r.data(),count*4);
  double se=0,sr=0,dot=0,sx=0;double ma=0;for(size_t i=0;i<count;++i){double d=(double)x[i]-r[i];ma=std::max(ma,std::abs(d));se+=d*d;sr+=(double)r[i]*r[i];dot+=(double)x[i]*r[i];sx+=(double)x[i]*x[i];}
  return{name,ma,std::sqrt(se/std::max(sr,1e-30)),dot/std::sqrt(std::max(sx*sr,1e-30))};
}
static std::string site_name(int bid,int u,int c,const char*kind){std::ostringstream o;o<<"b"<<bid<<"_u"<<u<<"_c"<<c<<"_"<<kind;return o.str();}

struct BranchWorker{
  int bid;const Fixture*fixture;std::string root,i8,parity;std::array<Model,6>models{};BranchScratch s;std::vector<float>output;std::vector<Metric>metrics;
  std::mutex mu;std::condition_variable cv;bool ready=false,job=false,done=true,stop=false;std::exception_ptr error;std::thread thread;
  BranchWorker(int id,const Fixture*f,const std::string&r,const std::string&q,const std::string&p):bid(id),fixture(f),root(r),i8(q),parity(p),thread(&BranchWorker::loop,this){
    std::unique_lock<std::mutex>lk(mu);cv.wait(lk,[&]{return ready;});if(error){lk.unlock();thread.join();std::rethrow_exception(error);}}
  ~BranchWorker(){{std::lock_guard<std::mutex>lk(mu);stop=true;cv.notify_all();}if(thread.joinable())thread.join();}
  void loop(){
    try{for(int u=0;u<3;++u)for(int c=1;c<=2;++c){int local=u*2+c-1;std::ostringstream name;name<<"branch"<<bid<<"_unit"<<u<<"_conv"<<c<<".convonly.rk3576.";bool use_i8=i8!="-"&&bid==2&&!(u==2&&c==2);std::string p=(use_i8?i8:root)+"/"+name.str()+(use_i8?"int8.rknn":"fp16.rknn");models[local]=load_model(p,MASKS[bid],C*TILE,C*TILE);}
      const size_t n=(size_t)C*fixture->n;s.y.resize(n);s.residual.resize(n);s.z.resize(n);s.tile.resize((size_t)C*TILE);s.out.resize((size_t)C*TILE);s.next.resize(n);
      {std::lock_guard<std::mutex>lk(mu);ready=true;cv.notify_all();}
      while(true){std::unique_lock<std::mutex>lk(mu);cv.wait(lk,[&]{return job||stop;});if(stop)break;job=false;lk.unlock();
        std::exception_ptr compute_error;
        try{compute();}catch(...){compute_error=std::current_exception();}
        lk.lock();error=compute_error;done=true;cv.notify_all();}
    }catch(...){std::lock_guard<std::mutex>lk(mu);error=std::current_exception();ready=true;done=true;cv.notify_all();}
    for(auto&m:models)m.reset();
  }
  void submit(){std::lock_guard<std::mutex>lk(mu);if(!done||job)throw std::runtime_error("worker already busy");done=false;error=nullptr;job=true;cv.notify_all();}
  void wait(){std::unique_lock<std::mutex>lk(mu);cv.wait(lk,[&]{return done;});if(error)std::rethrow_exception(error);}
  void compute(){
    const uint32_t n=fixture->n;
    // No caller mutates fixture until all workers have drained.
    const size_t count=(size_t)C*n;
    s.residual.resize(count);s.z.resize(count);s.next.resize(count);
    s.y=fixture->x;metrics.clear();
    for(int u=0;u<3;++u){s.residual=s.y;for(int c=1;c<=2;++c){int sid=bid*6+u*2+c-1,local=u*2+c-1;film_prelu(s.y.data(),s.z.data(),n,&fixture->gamma[sid*C],&fixture->beta[sid*C],&fixture->slopes[sid*C]);
        std::string sn=site_name(bid,u,c,"pre");if(!parity.empty())metrics.push_back(compare_file(sn,parity+"/"+sn+".f32",s.z.data(),(size_t)C*n));
        uint32_t halo=(KERNELS[bid]-1)*(c==1?DILATIONS[u]:1)/2,outpos=0;for(auto sl:schedule(n,halo)){std::fill(s.tile.begin(),s.tile.end(),0);uint32_t len=sl.i1-sl.i0,keep=sl.o1-sl.o0;
          for(uint32_t ch=0;ch<C;++ch)std::memcpy(&s.tile[(size_t)ch*TILE],&s.z[(size_t)ch*n+sl.i0],len*4);run(models[local],s.tile.data(),s.out.data());
          for(uint32_t ch=0;ch<C;++ch)std::memcpy(&s.next[(size_t)ch*n+outpos],&s.out[(size_t)ch*TILE+sl.o0],keep*4);outpos+=keep;}
        if(outpos!=n)throw std::runtime_error("stitch length mismatch");s.y.swap(s.next);sn=site_name(bid,u,c,"conv");if(!parity.empty())metrics.push_back(compare_file(sn,parity+"/"+sn+".f32",s.y.data(),(size_t)C*n));}
      #pragma GCC ivdep
      for(size_t i=0;i<s.y.size();++i)s.y[i]+=s.residual[i];}output=s.y;}
};

struct Engine{
  Fixture fixture;Model merge;std::array<std::unique_ptr<BranchWorker>,3>workers;std::string parity;std::vector<Metric>metrics;
  Engine(Fixture f,const std::string&root,const std::string&i8,const std::string&merge_path,const std::string&par):fixture(std::move(f)),merge(load_merge(merge_path)),parity(par){for(int i=0;i<3;++i)workers[i]=std::make_unique<BranchWorker>(i,&fixture,root,i8,parity);}
  ~Engine(){for(auto&w:workers)w.reset();}
  void submit_wait(int i){workers[i]->submit();workers[i]->wait();}
  std::vector<float> once(const std::string&mode){metrics.clear();
    if(mode=="all"){
      std::array<bool,3> submitted{};std::exception_ptr first;
      try{for(size_t i=0;i<workers.size();++i){workers[i]->submit();submitted[i]=true;}}
      catch(...){first=std::current_exception();}
      for(size_t i=0;i<workers.size();++i)if(submitted[i]){
        try{workers[i]->wait();}catch(...){if(!first)first=std::current_exception();}
      }
      if(first)std::rethrow_exception(first);
    }
    else if(mode=="01"||mode=="02"||mode=="12"){int a=mode[0]-'0',b=mode[1]-'0',last=3-a-b;workers[a]->submit();workers[b]->submit();workers[a]->wait();workers[b]->wait();submit_wait(last);}
    else if(mode=="serial")for(int i=0;i<3;++i)submit_wait(i);else throw std::runtime_error("schedule must be serial, 01, 02, 12, or all");
    for(auto&w:workers)metrics.insert(metrics.end(),w->metrics.begin(),w->metrics.end());const uint32_t n=fixture.n;std::vector<float>result((size_t)OUT_C*n),ins(3ULL*C*TILE),out((size_t)OUT_C*TILE);uint32_t pos=0;
    const auto&mc=merge.contract;std::array<bool,3>in_nhwc{},in_nchw{};
    for(int k=0;k<3;++k){in_nhwc[k]=mc.input_fmt[k]==1&&mc.input_dims[k][1]==1&&mc.input_dims[k][2]==TILE&&mc.input_dims[k][3]==C;
      in_nchw[k]=mc.input_fmt[k]==0&&mc.input_dims[k][1]==C&&mc.input_dims[k][2]==1&&mc.input_dims[k][3]==TILE;}
    bool out_nhwc=mc.output_fmt==1&&mc.output_dims[1]==1&&mc.output_dims[2]==TILE&&mc.output_dims[3]==OUT_C;
    bool out_nchw=mc.output_fmt==0&&mc.output_dims[1]==OUT_C&&mc.output_dims[2]==1&&mc.output_dims[3]==TILE;
    for(int k=0;k<3;++k)if(!in_nhwc[k]&&!in_nchw[k])throw std::runtime_error("unsupported merge input tensor layout");
    if(!out_nhwc&&!out_nchw)throw std::runtime_error("unsupported merge output tensor layout");
    // RK3576 normally exposes NHWC [1,1,T,128] for the logical NCHW merge
    // input. Query-driven packing also keeps the runner safe if firmware/compiler
    // later exposes NCHW directly.
    for(auto sl:schedule(n,3)){std::fill(ins.begin(),ins.end(),0);uint32_t len=sl.i1-sl.i0,keep=sl.o1-sl.o0;for(int k=0;k<3;++k){if(in_nhwc[k]){for(uint32_t t=0;t<len;++t)for(uint32_t ch=0;ch<C;++ch)ins[(size_t)k*C*TILE+(size_t)t*C+ch]=workers[k]->output[(size_t)ch*n+sl.i0+t];}else{for(uint32_t ch=0;ch<C;++ch)std::memcpy(&ins[(size_t)k*C*TILE+(size_t)ch*TILE],&workers[k]->output[(size_t)ch*n+sl.i0],len*4);}}
      const float*ip[6]={ins.data(),ins.data()+C*TILE,ins.data()+2*C*TILE};uint32_t sizes[6]={C*TILE,C*TILE,C*TILE};if(kokoro_rk3588_run_float32(merge.h,ip,sizes,out.data(),OUT_C*TILE))throw std::runtime_error(kokoro_rk3588_last_error(merge.h));
      if(out_nhwc){for(uint32_t t=0;t<keep;++t)for(uint32_t ch=0;ch<OUT_C;++ch)result[(size_t)ch*n+pos+t]=out[(size_t)(sl.o0+t)*OUT_C+ch];}else{for(uint32_t ch=0;ch<OUT_C;++ch)std::memcpy(&result[(size_t)ch*n+pos],&out[(size_t)ch*TILE+sl.o0],keep*4);}pos+=keep;}
    if(pos!=n)throw std::runtime_error("merge stitch length mismatch");
    if(!parity.empty())metrics.push_back(compare_file("final",parity+"/final.f32",result.data(),result.size()));return result;}
};

namespace {
constexpr uint32_t MAX_N=38401;
void error_text(char* dst,uint32_t size,const char* message) noexcept {
  if(dst&&size){std::strncpy(dst,message?message:"unknown error",size-1);dst[size-1]='\0';}
}
bool all_finite(const float* value,size_t count) noexcept {
  if(!value)return false;
  for(size_t i=0;i<count;++i)if(!std::isfinite(value[i]))return false;
  return true;
}
}
struct kokoro_convonly_handle {
  std::mutex mutex;
  uint32_t max_n;
  std::unique_ptr<Engine> engine;
  explicit kokoro_convonly_handle(uint32_t limit):max_n(limit){}
};

extern "C" kokoro_convonly_handle* kokoro_convonly_create(
    const char* root,const char* merge_path,const float* slopes,
    uint32_t slope_count,uint32_t max_n,char* error,uint32_t error_bytes) {
  try {
    if(!root||!*root||!merge_path||!*merge_path||!slopes||
       slope_count!=18*C||max_n<1||max_n>MAX_N)
      throw std::invalid_argument("invalid create arguments");
    if(!all_finite(slopes,slope_count))throw std::invalid_argument("nonfinite slopes");
    auto handle=std::make_unique<kokoro_convonly_handle>(max_n);
    Fixture f;f.n=1;f.x.resize(C);f.gamma.resize(18*C);f.beta.resize(18*C);
    f.slopes.assign(slopes,slopes+slope_count);
    // The production interface never exposes the historical mixed-INT8 path.
    handle->engine=std::make_unique<Engine>(std::move(f),root,"-",merge_path,"");
    error_text(error,error_bytes,"");
    return handle.release();
  } catch(const std::exception& exc){error_text(error,error_bytes,exc.what());}
    catch(...){error_text(error,error_bytes,"unknown create failure");}
  return nullptr;
}
extern "C" int kokoro_convonly_run(
    kokoro_convonly_handle* handle,uint32_t n,const float* j1,uint32_t j1_count,
    const float* gamma,uint32_t gamma_count,const float* beta,uint32_t beta_count,
    float* output,uint32_t output_count,char* error,uint32_t error_bytes) {
  try {
    if(!handle)throw std::invalid_argument("null handle");
    std::lock_guard<std::mutex> lock(handle->mutex);
    if(n<1||n>handle->max_n||!j1||!gamma||!beta||!output||
       uint64_t(j1_count)!=uint64_t(C)*n||gamma_count!=18*C||
       beta_count!=18*C||uint64_t(output_count)!=uint64_t(OUT_C)*n)
      throw std::invalid_argument("buffer element count mismatch");
    if(!all_finite(j1,j1_count)||!all_finite(gamma,gamma_count)||!all_finite(beta,beta_count))
      throw std::invalid_argument("nonfinite input");
    auto& f=handle->engine->fixture;f.n=n;
    f.x.assign(j1,j1+j1_count);f.gamma.assign(gamma,gamma+gamma_count);
    f.beta.assign(beta,beta+beta_count);
    auto result=handle->engine->once("all");
    if(result.size()!=output_count||!all_finite(result.data(),result.size()))
      throw std::runtime_error("nonfinite or invalid output");
    std::memcpy(output,result.data(),result.size()*sizeof(float));
    error_text(error,error_bytes,"");return 0;
  } catch(const std::exception& exc){error_text(error,error_bytes,exc.what());}
    catch(...){error_text(error,error_bytes,"unknown run failure");}
  return -1;
}
extern "C" void kokoro_convonly_destroy(kokoro_convonly_handle* handle) {
  // Caller MUST quiesce all API calls first; a mutex cannot validate a freed pointer.
  delete handle;
}

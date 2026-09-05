#include "../kokoro_rk3588_bridge.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <atomic>

struct kokoro_rk3588_handle { bool merge; kokoro_rk3588_contract c{}; std::string error; };
static std::atomic<int> creates{0},destroys{0},runs{0},fail_create{0},fail_run{0};
extern "C" void stub_reset(){creates=0;destroys=0;runs=0;fail_create=0;fail_run=0;}
extern "C" void stub_set_fail_at_create(int n){fail_create=n;}
extern "C" void stub_set_fail_at_run(int n){fail_run=n;}
extern "C" int stub_create_count(){return creates;} extern "C" int stub_destroy_count(){return destroys;}
extern "C" int stub_run_count(){return runs;}
extern "C" kokoro_rk3588_handle*kokoro_rk3588_create(const char*p,int){
  const int call=++creates;if(fail_create&&call==fail_create)return nullptr;auto*h=new kokoro_rk3588_handle;h->merge=std::string(p?p:"").find("merge")!=std::string::npos;
  h->c.n_input=h->merge?3:1;h->c.n_output=1;for(unsigned i=0;i<h->c.n_input;++i){h->c.input_ndims[i]=4;h->c.input_dims[i][0]=1;h->c.input_dims[i][1]=h->merge?1:128;h->c.input_dims[i][2]=h->merge?8192:1;h->c.input_dims[i][3]=h->merge?128:8192;h->c.input_floats[i]=128*8192;h->c.input_fmt[i]=h->merge?1:0;}
  h->c.output_ndims=4;h->c.output_dims[0]=1;h->c.output_dims[1]=h->merge?22:128;h->c.output_dims[2]=1;h->c.output_dims[3]=8192;h->c.output_floats=(h->merge?22:128)*8192;h->c.output_fmt=0;return h;
}
extern "C" int kokoro_rk3588_query_contract(const kokoro_rk3588_handle*h,kokoro_rk3588_contract*o){if(!h||!o)return-1;*o=h->c;return 0;}
extern "C" int kokoro_rk3588_run_float32(kokoro_rk3588_handle*h,const float*const in[6],const uint32_t sz[6],float*out,uint32_t n){
  if(!h||!in||!out||n<h->c.output_floats)return-1;
  const int call=++runs;if(fail_run&&call==fail_run){h->error="stub injected run failure";return-1;}
  if(!h->merge){if(sz[0]!=128*8192)return-1;std::copy(in[0],in[0]+128*8192,out);return 0;}
  if(sz[0]!=128*8192||sz[1]!=128*8192||sz[2]!=128*8192)return-1;
  for(unsigned c=0;c<22;++c)for(unsigned t=0;t<8192;++t){float v=0;for(unsigned k=0;k<3;++k)v+=in[k][t*128+c];out[c*8192+t]=v/3.f;}return 0;
}
extern "C" const char*kokoro_rk3588_last_error(const kokoro_rk3588_handle*h){return h?h->error.c_str():"stub null";}
extern "C" void kokoro_rk3588_destroy(kokoro_rk3588_handle*h){if(h){++destroys;delete h;}}

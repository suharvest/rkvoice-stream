#!/usr/bin/env python3
"""Complete hybrid Conv-only tail runner.

The Python side owns full-sequence statistics, FiLM, PReLU, masking, halo
stitching and residuals.  Only the 18 static Conv graphs and the approved
three-input merge graph cross the native RKNN C-API boundary.
"""
from __future__ import annotations
import time
import threading
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

CORE_MASKS = {'rk3588': (1, 2, 4), 'rk3576': (1, 2, 1)}
KERNELS = (3, 7, 11)
DILATIONS = (1, 3, 5)
TILE = 8192

class PythonConvOnlyTail:
    """Persistent three-branch Conv-only tail using the approved arithmetic."""
    def __init__(self, model_root, merge_path, slopes, library, max_n=38401, runtime_cls=None):
        if isinstance(max_n, bool) or not isinstance(max_n, int) or not 1 <= max_n <= 38401:
            raise ValueError("max_n must be an integer in 1..38401")
        self._lock=threading.RLock(); self._closed=False; self.max_n=max_n
        self._cleanup_error=None
        self.models={}; self.merge=None; self.executor=None
        try:
            if runtime_cls is None:
                from .kokoro_convonly_capi import RK3588CapiRuntime
                runtime_cls=RK3588CapiRuntime
            self.slopes=self._load_slopes(slopes); self.models=make_models(Path(model_root),'rk3588',runtime_cls,library)
            self.merge=runtime_cls(str(merge_path),0,library); self.executor=ThreadPoolExecutor(max_workers=3)
        except BaseException as exc:
            try: self.close()
            except BaseException as cleanup:
                raise RuntimeError(f"tail initialization failed: {exc}; cleanup failed: {cleanup}") from exc
            raise
    @staticmethod
    def _load_slopes(slopes):
        if isinstance(slopes,(str,Path)):
            root=Path(slopes); values=[np.asarray(np.load(root/f'branch{b}_unit{u}_conv{c}.npy',allow_pickle=False)) for b in range(3) for u in range(3) for c in (1,2)]
        else: values=[np.asarray(x) for x in slopes]
        if len(values)!=18 or any(v.dtype!=np.float32 or v.size!=128 or not np.isfinite(v).all() for v in values): raise ValueError('slopes must be exactly 18 finite vectors of 128')
        return [np.array(v.reshape(1,128,1),dtype=np.float32,copy=True) for v in values]
    def run(self,j1,gamma,beta):
        with self._lock:
            if self._closed: raise RuntimeError('tail is closed')
            raw=np.asarray(j1); x=np.ascontiguousarray(raw,np.float32); n=int(x.shape[-1]) if x.ndim==3 else -1
            if raw.dtype != np.float32 or raw.shape!=(1,128,n) or n<1 or n>self.max_n or not raw.flags.c_contiguous or not np.isfinite(x).all(): raise ValueError('invalid j1')
            ga=np.asarray(gamma); be=np.asarray(beta)
            if ga.dtype!=np.float32 or be.dtype!=np.float32 or ga.shape!=(1,18,128) or be.shape!=ga.shape or not ga.flags.c_contiguous or not be.flags.c_contiguous: raise ValueError('FiLM must be contiguous [1,18,128]')
            g=sites(ga); b=sites(be)
            if not all(np.isfinite(v).all() for v in g+b): raise ValueError('non-finite FiLM')
            started=time.perf_counter(); fs=[]
            try:
                for i in range(3): fs.append(self.executor.submit(run_branch,i,x,g,b,self.models,n,self.slopes))
                got=[f.result() for f in fs]
            finally:
                for f in fs:f.cancel()
                for f in fs:
                    try:f.result()
                    except BaseException:pass
            pieces=[]; merge_ms=0.
            for i0,i1,o0,o1 in schedule(n,3):
                ins=[]
                for q,_ in got:
                    z=np.zeros((1,128,TILE),np.float32); z[:,:,:i1-i0]=q[:,:,i0:i1]; ins.append(z)
                mt=time.perf_counter(); out=np.asarray(self.merge.inference(ins)[0],np.float32); merge_ms+=(time.perf_counter()-mt)*1000
                if out.ndim==4 and out.shape[2]==1:out=out[:,:,0,:]
                if out.shape!=(1,22,TILE):raise RuntimeError(f'bad merge output {out.shape}')
                pieces.append(out[:,:,o0:o1])
            out=np.ascontiguousarray(np.concatenate(pieces,axis=-1),np.float32)
            if out.shape!=(1,22,n) or not np.isfinite(out).all():raise RuntimeError('tail output ABI failed')
            return out,{'tail_ms_including_merge':(time.perf_counter()-started)*1000,'merge_ms':merge_ms,'branch_ms':[q[1]['wall_ms'] for q in got]}
    def close(self):
        with self._lock:
            if self._closed:
                if self._cleanup_error: raise RuntimeError(self._cleanup_error)
                return
            errors=[]
            if self.executor is not None:
                try:self.executor.shutdown(wait=True,cancel_futures=True)
                except BaseException as exc:errors.append(f'executor: {exc}')
            if self.merge is not None:
                try:self.merge.release()
                except BaseException as exc:errors.append(f'merge: {exc}')
            for m in self.models.values():
                try:m.close()
                except BaseException as exc:errors.append(f'model: {exc}')
            self._closed=True
            if errors:
                self._cleanup_error='tail cleanup failed: '+'; '.join(errors)
                raise RuntimeError(self._cleanup_error)
    cleanup=close

def schedule(n, halo):
    if n <= TILE: return [(0,n,0,n)]
    core=TILE-2*halo; out=0; rows=[]
    while n-out > core:
        end=out+core; start=0 if out==0 else out-halo
        input_end=min(n,end+halo)
        rows.append((start,input_end,out-start,out-start+min(core,n-out)))
        assert input_end-start <= TILE and rows[-1][3] <= input_end-start; out=end
    start=max(0,n-TILE); rows.append((start,n,out-start,n-start))
    return rows

def sites(v):
    v=np.asarray(v)
    if v.dtype!=np.float32: raise ValueError("FiLM must be float32")
    if v.ndim==3 and v.shape==(1,18,128) and v.flags.c_contiguous:
        return [v[:,i,:].reshape(1,128,1) for i in range(18)]
    raise ValueError(f'style tensor must [1,18,128], got {v.shape}')

def run_branch(bid, x, g, b, models, n, slopes):
    y=x.copy(); binfo={'branch':bid,'units':[]}; started=time.perf_counter()
    for u in range(3):
      residual=y.copy(); ui={'unit':u,'sites':[]}
      for c in (1,2):
        st=time.perf_counter(); gamma=g[bid*6+u*2+c-1]; beta=b[bid*6+u*2+c-1]
        mean=y.mean(axis=-1,keepdims=True,dtype=np.float32); var=((y-mean)*(y-mean)).mean(axis=-1,keepdims=True,dtype=np.float32)
        inv=np.reciprocal(np.sqrt(var+np.float32(1e-5)),dtype=np.float32); z=(y-mean)*inv; z=z*(1+gamma)+beta
        z=np.where(z>=0,z,z*slopes[bid*6+u*2+c-1]).astype(np.float32,copy=False)
        halo=(KERNELS[bid]-1)*(DILATIONS[u] if c==1 else 1)//2; pieces=[]; npu=0.
        for i0,i1,o0,o1 in schedule(n,halo):
          tile=z[:,:,i0:i1]; tl=tile.shape[-1]; pad=np.zeros((1,128,TILE),np.float32); pad[:,:,:tl]=tile
          nt=time.perf_counter(); out=models[(bid,u,c)].run(pad); npu+=(time.perf_counter()-nt)*1000
          if out.ndim==4 and out.shape[2]==1: out=out[:,:,0,:]
          if out.shape!=(1,128,TILE): raise RuntimeError(f'bad site output {out.shape}')
          pieces.append(out[:,:,o0:o1])
        y=np.concatenate(pieces,axis=-1); ui['sites'].append({'conv':c,'halo':halo,'tiles':len(pieces),'cpu_pre_ms':(time.perf_counter()-st)*1000-npu,'npu_ms':npu})
      y=residual+y; ui['residual']=True; binfo['units'].append(ui)
    binfo['wall_ms']=(time.perf_counter()-started)*1000; return y,binfo

class Conv:
    def __init__(self,p,mask,Runtime,bridge):
        self.p=p; self.rt=Runtime(str(p),int(mask),bridge)
    def run(self,x): return np.asarray(self.rt.inference([x])[0],np.float32)
    def close(self): self.rt.release()

def make_models(root, platform, Runtime, bridge, mock=False):
    # mock is intentionally only a host contract aid; device runs always use C API.
    models={}
    masks = CORE_MASKS.get(platform)
    if masks is None: raise ValueError(f'unsupported platform: {platform}')
    try:
        for bid, mask in enumerate(masks):
            for u in range(3):
                for c in (1,2):
                    p=root/f'branch{bid}_unit{u}_conv{c}.convonly.{platform}.fp16.rknn'
                    if not p.is_file(): raise FileNotFoundError(p)
                    models[(bid,u,c)]=Conv(p,mask,Runtime,bridge)
    except BaseException as exc:
        errors=[]
        for model in reversed(list(models.values())):
            try: model.close()
            except BaseException as cleanup: errors.append(str(cleanup))
        if errors: raise RuntimeError(f"model initialization failed: {exc}; cleanup: {errors}") from exc
        raise
    return models

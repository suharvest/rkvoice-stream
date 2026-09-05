import numpy as np
import pytest
import threading
import time
from types import SimpleNamespace
from rkvoice_stream.backends.tts import kokoro_convonly_tail as M

def test_schedule_crosses_tile_boundary():
    rows=M.schedule(9000,3); assert len(rows)==2; assert rows[-1][1]==9000
    assert sum(row[3]-row[2] for row in rows)==9000
    assert all((b-a)<=M.TILE for a,b,_,_ in rows)

def test_sites_and_run_branch_shape():
    x=np.zeros((1,128,16),np.float32); g=[np.zeros((1,128,1),np.float32) for _ in range(18)]; b=[v.copy() for v in g]
    class Model:
        def run(self,z): return np.zeros((1,128,M.TILE),np.float32)
    models={(bid,u,c):Model() for bid in range(3) for u in range(3) for c in (1,2)}
    slopes=[np.ones((1,128,1),np.float32) for _ in range(18)]
    out,info=M.run_branch(0,x,g,b,models,16,slopes); assert out.shape==(1,128,16) and info["wall_ms"]>=0

def test_tail_partial_initialization_releases_registered_resources(monkeypatch,tmp_path):
    class R:
        def __init__(self,*a): pass
        def release(self): pass
    monkeypatch.setattr(M,"make_models",lambda *a,**k: (_ for _ in ()).throw(RuntimeError("model failed")))
    with pytest.raises(RuntimeError): M.PythonConvOnlyTail(tmp_path,tmp_path/"merge",[np.zeros(128,np.float32)]*18,tmp_path/"lib",runtime_cls=R)

def _models(tmp):
    for bid in range(3):
        for unit in range(3):
            for conv in (1,2): (tmp/f"branch{bid}_unit{unit}_conv{conv}.convonly.rk3588.fp16.rknn").write_bytes(b"")

class _Runtime:
    made=[]
    def __init__(self,*args):
        self.__class__.made.append(self); self.closed=0
    def inference(self,inputs): return [np.zeros((1,128,M.TILE),np.float32)]
    def release(self): self.closed+=1

def test_constructor_releases_each_registered_model_on_fourth_create(monkeypatch,tmp_path):
    _models(tmp_path); _Runtime.made=[]; count=[0]
    class FailRuntime(_Runtime):
        def __init__(self,*args):
            count[0]+=1
            if count[0]==4: raise RuntimeError("fourth create")
            super().__init__(*args)
    with pytest.raises(RuntimeError,match="fourth create"):
        M.PythonConvOnlyTail(tmp_path,tmp_path/"merge",[np.ones(128,np.float32)]*18,tmp_path/"lib",runtime_cls=FailRuntime)
    assert len(_Runtime.made)==3 and [x.closed for x in _Runtime.made]==[1,1,1]

def test_normal_constructor_has_19_contexts_and_idempotent_close(tmp_path):
    _models(tmp_path); _Runtime.made=[]
    tail=M.PythonConvOnlyTail(tmp_path,tmp_path/"merge",[np.ones(128,np.float32)]*18,tmp_path/"lib",runtime_cls=_Runtime)
    assert len(_Runtime.made)==19
    tail.close(); tail.close(); assert sum(x.closed for x in _Runtime.made)==19

def test_submit_failure_drains_already_running_branch(monkeypatch):
    started=threading.Event(); release=threading.Event(); finished=threading.Event()
    class Future:
        def __init__(self): self.done=False
        def result(self): started.set(); release.wait(2); self.done=True; return (np.zeros((1,128,4),np.float32),{"wall_ms":1})
        def cancel(self): return False
    class Executor:
        def __init__(self): self.calls=0; self.f=Future()
        def submit(self,*args):
            self.calls+=1
            if self.calls==1:return self.f
            raise RuntimeError("submit failed")
    tail=SimpleNamespace(_lock=threading.RLock(),_closed=False,max_n=32,models={},slopes=[np.ones((1,128,1),np.float32)]*18,executor=Executor(),merge=None)
    with pytest.raises(RuntimeError,match="submit failed"):
        # Exercise the implementation's submit/finally path.
        M.PythonConvOnlyTail.run(tail,np.zeros((1,128,4),np.float32),np.zeros((1,18,128),np.float32),np.zeros((1,18,128),np.float32))
    assert started.is_set(); release.set(); assert tail.executor.f.done is True

def test_tail_clock_includes_branch_time(monkeypatch):
    clock=[0.0]
    def tick(): value=clock[0]; clock[0]+=1.0; return value
    monkeypatch.setattr(M.time,"perf_counter",tick)
    def branch(*args): clock[0]+=5.0; return np.zeros((1,128,4),np.float32),{"wall_ms":5}
    monkeypatch.setattr(M,"run_branch",branch)
    class Merge:
        def inference(self,_): return [np.zeros((1,22,M.TILE),np.float32)]
    tail=SimpleNamespace(_lock=threading.RLock(),_closed=False,max_n=32,models={},slopes=[np.ones((1,128,1),np.float32)]*18,executor=__import__('concurrent.futures').futures.ThreadPoolExecutor(3),merge=Merge())
    try:
        _, info=M.PythonConvOnlyTail.run(tail,np.zeros((1,128,4),np.float32),np.zeros((1,18,128),np.float32),np.zeros((1,18,128),np.float32))
        assert info["tail_ms_including_merge"]>=5.0
    finally: tail.executor.shutdown()

def test_close_waits_until_running_call_releases(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    entered=threading.Event(); allow=threading.Event(); released=threading.Event()
    class Tail:
        _lock=threading.RLock(); _closed=False; executor=ThreadPoolExecutor(3); models={}; max_n=4; slopes=[np.ones((1,128,1),np.float32)]*18
        merge=type("Merge", (), {"inference": lambda self, _inputs: [np.zeros((1,22,M.TILE),np.float32)], "release": lambda self: None})()
        close=M.PythonConvOnlyTail.close
    tail=Tail()
    def run(*_): entered.set(); allow.wait(2); return np.zeros((1,128,4),np.float32),{"wall_ms":1}
    monkeypatch.setattr(M,"run_branch",run)
    # No merge is needed: the blocked branch demonstrates close's lifecycle lock.
    worker=threading.Thread(target=lambda: M.PythonConvOnlyTail.run(tail,np.zeros((1,128,4),np.float32),np.zeros((1,18,128),np.float32),np.zeros((1,18,128),np.float32)))
    original_shutdown=tail.executor.shutdown
    def shutdown(**kwargs): released.set(); return original_shutdown(**kwargs)
    tail.executor.shutdown=shutdown
    worker.start(); entered.wait(1); closer=threading.Thread(target=tail.close); closer.start(); time.sleep(.05); assert not released.is_set(); allow.set(); worker.join(2); closer.join(2); assert released.is_set()

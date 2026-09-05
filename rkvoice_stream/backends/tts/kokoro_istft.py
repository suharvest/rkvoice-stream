"""Validated Kokoro v1.0 post-generator CPU tail (production package)."""
from __future__ import annotations
import numpy as np

def conv_post_to_waveform(conv_post: np.ndarray) -> np.ndarray:
    q=np.asarray(conv_post)
    if q.dtype != np.float32 or q.ndim != 3 or q.shape[1] != 22 or q.shape[2] < 2 or not np.isfinite(q).all():
        raise ValueError("conv_post must be finite float32 [B,22,F]")
    f=q.shape[2]
    with np.errstate(over="ignore", invalid="ignore"):
        mag=np.exp(q[:,:11,:],dtype=np.float32)
    phase=np.sin(q[:,11:,:],dtype=np.float32)
    if not np.isfinite(mag).all() or not np.isfinite(phase).all(): raise FloatingPointError("non-finite polar values")
    spec=mag*(np.cos(phase,dtype=np.float32)+1j*np.sin(phase,dtype=np.float32))
    frames=np.asarray(np.fft.irfft(np.transpose(spec,(0,2,1)),n=20,axis=-1),np.float32)
    w=(.5-.5*np.cos(2*np.pi*np.arange(20,dtype=np.float32)/20)).astype(np.float32)
    out=np.zeros((q.shape[0],(f-1)*5+20),np.float32); starts=5*np.arange(f)
    for off in range(20): out[:,starts+off]+=frames[:,:,off]*w[off]
    out[:,1:20]/=np.maximum(w[1:]*w[1:],1e-8)
    out=np.asarray(out[:,10:-10],np.float32)
    if out.shape != (q.shape[0],5*(f-1)) or not np.isfinite(out).all(): raise FloatingPointError("iSTFT output ABI/nonfinite")
    return out


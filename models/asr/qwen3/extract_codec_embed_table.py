#!/usr/bin/env python3
"""Extract codec_embed RKNN weights into a numpy lookup table.

Runs the RKNN model for all vocab IDs (0..vocab_size-1) and saves the
output as a [vocab_size, 1024] float32 numpy array.

Usage:
    python extract_codec_embed_table.py /opt/tts/models
"""

import sys
import os
import numpy as np

def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/opt/tts/models"
    rknn_path = os.path.join(model_dir, "codec_embed.rknn")
    out_path = os.path.join(model_dir, "codec_embed_table.npy")

    from rknnlite.api import RKNNLite
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn(rknn_path)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

    # Probe vocab size: try common sizes
    vocab_size = 3072  # TALKER_VOCAB_SIZE from tts_service.py
    table = np.zeros((vocab_size, 1024), dtype=np.float32)

    print(f"Extracting {vocab_size} embeddings from {rknn_path}...")
    for i in range(vocab_size):
        inp = np.array([[i]], dtype=np.int64)
        out = rknn.inference(inputs=[inp])
        table[i] = np.array(out[0])[0, 0]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{vocab_size}")

    np.save(out_path, table)
    print(f"Saved {out_path}: {table.shape} ({table.nbytes / 1024 / 1024:.1f} MB)")
    rknn.release()


if __name__ == "__main__":
    main()

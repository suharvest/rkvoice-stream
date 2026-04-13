#!/usr/bin/env python3
"""Extract text_project RKNN weights into a numpy lookup table.

The text_project model is an embedding layer: token_id -> [1024] float32.
Extracts all vocab embeddings and saves as [vocab_size, 1024] numpy array.

This replaces the 606MB RKNN model with a ~590MB numpy file that can be
mmap'd at near-zero RSS cost.

Usage:
    python extract_text_project_table.py /opt/tts/models [vocab_size]
"""

import sys
import os
import numpy as np


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/opt/tts/models"
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 151936
    rknn_path = os.path.join(model_dir, "text_project.rknn")
    out_path = os.path.join(model_dir, "text_project_table.npy")

    from rknnlite.api import RKNNLite
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn(rknn_path)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

    # text_project input: [1, 128] int64, output: [1, 128, 1024] float32
    # We batch 128 token IDs per inference call
    batch_size = 128
    table = np.zeros((vocab_size, 1024), dtype=np.float32)

    print(f"Extracting {vocab_size} embeddings from {rknn_path}...")
    print(f"Batch size: {batch_size}, total batches: {(vocab_size + batch_size - 1) // batch_size}")

    for start in range(0, vocab_size, batch_size):
        end = min(start + batch_size, vocab_size)
        n = end - start

        inp = np.zeros((1, 128), dtype=np.int64)
        inp[0, :n] = np.arange(start, end)

        out = rknn.inference(inputs=[inp])
        result = np.array(out[0])  # [1, 128, 1024]
        table[start:end] = result[0, :n]

        if (start // batch_size + 1) % 100 == 0 or end == vocab_size:
            print(f"  {end}/{vocab_size} ({end * 100 // vocab_size}%)")

    np.save(out_path, table)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Saved {out_path}: {table.shape} ({size_mb:.1f} MB)")
    rknn.release()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate E2E reference data from Qwen3-TTS on GPU.
No hooks - clean run to verify the model works.
"""
import os
import json
import numpy as np
import torch

OUT_DIR = os.path.expanduser("~/qwen3-tts-export/e2e_ref")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_TEXT = "今天天气真不错"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
    from transformers import AutoTokenizer

    MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    print("Loading model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    )
    if device == "cuda":
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Build input text
    assistant_text = "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n".format(TEST_TEXT)
    input_ids = tokenizer(assistant_text, return_tensors="pt").input_ids.to(device)
    print("Input IDs shape:", input_ids.shape)

    print("Running model.generate() (no hooks)...")
    with torch.no_grad():
        try:
            talker_codes_list, talker_hidden_list = model.generate(
                input_ids=[input_ids],
                languages=["chinese"],
                non_streaming_mode=True,
                max_new_tokens=2048,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                repetition_penalty=1.05,
            )

            print("Generated {} code sequences".format(len(talker_codes_list)))
            for i, codes in enumerate(talker_codes_list):
                codes_np = codes.cpu().numpy()
                print("  Seq {}: {} tokens, shape={}".format(i, len(codes_np), codes_np.shape))
                print("  First 30: {}".format(codes_np[:30].tolist()))
                print("  Range: [{}, {}]".format(codes_np.min(), codes_np.max()))
                np.save(os.path.join(OUT_DIR, "codec_tokens.npy"), codes_np)

            # Decode to audio
            print("\nDecoding to audio...")
            codes_for_decode = [{"audio_codes": codes} for codes in talker_codes_list]
            wavs, fs = model.speech_tokenizer.decode(codes_for_decode)

            for i, wav in enumerate(wavs):
                import soundfile as sf
                wav_path = os.path.join(OUT_DIR, "reference.wav")
                sf.write(wav_path, wav, fs)
                print("Saved {}: {} samples, {:.2f}s @ {}Hz".format(
                    wav_path, len(wav), len(wav) / fs, fs))

            # Save metadata
            codes_np = talker_codes_list[0].cpu().numpy()
            meta = {
                "text": TEST_TEXT,
                "sample_rate": fs,
                "audio_duration_s": round(len(wavs[0]) / fs, 2),
                "n_codec_tokens": len(codes_np),
                "codec_tokens_all": codes_np.tolist(),
            }
            with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print("ERROR: {}".format(e))
            import traceback
            traceback.print_exc()

            # Try on CPU as fallback
            print("\nRetrying on CPU with float32...")
            model_cpu = Qwen3TTSForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=torch.float32
            )
            model_cpu.eval()
            input_ids_cpu = tokenizer(assistant_text, return_tensors="pt").input_ids

            talker_codes_list, _ = model_cpu.generate(
                input_ids=[input_ids_cpu],
                languages=["chinese"],
                non_streaming_mode=True,
                max_new_tokens=512,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
            )

            print("CPU generated {} code sequences".format(len(talker_codes_list)))
            for i, codes in enumerate(talker_codes_list):
                codes_np = codes.cpu().numpy()
                print("  Seq {}: {} tokens".format(i, len(codes_np)))
                print("  First 30: {}".format(codes_np[:30].tolist()))
                np.save(os.path.join(OUT_DIR, "codec_tokens.npy"), codes_np)

            # Decode
            codes_for_decode = [{"audio_codes": codes} for codes in talker_codes_list]
            wavs, fs = model_cpu.speech_tokenizer.decode(codes_for_decode)
            import soundfile as sf
            wav_path = os.path.join(OUT_DIR, "reference.wav")
            sf.write(wav_path, wavs[0], fs)
            print("Saved CPU audio:", wav_path)

            codes_np = talker_codes_list[0].cpu().numpy()
            meta = {
                "text": TEST_TEXT,
                "sample_rate": fs,
                "audio_duration_s": round(len(wavs[0]) / fs, 2),
                "n_codec_tokens": len(codes_np),
                "codec_tokens_all": codes_np.tolist(),
                "computed_on": "cpu_fp32",
            }
            with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDone. Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print("  {} ({:.1f} KB)".format(f, size / 1024))


if __name__ == "__main__":
    main()

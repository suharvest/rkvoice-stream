"""Kokoro/Kokora RKNN TTS backend.

This backend is intentionally small and model-file driven.  It is for Kokoro
RKNN exports that keep the sherpa-onnx style input contract:

  tokens: int64 [1, seq_len]
  style:  float32 [1, 256]
  speed:  float32 [1]

The full Kokoro graph is known to hit RKNN register limits on RK3576/RK3588
for common exports.  When a working split or bucketed RKNN artifact is
available, this backend gives the service a stable runtime entry point.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = int(os.environ.get("KOKORO_SAMPLE_RATE", "24000"))
SEQ_LEN = int(os.environ.get("KOKORO_SEQ_LEN", os.environ.get("KOKORA_SEQ_LEN", "128")))
STYLE_DIM = int(os.environ.get("KOKORO_STYLE_DIM", "256"))
MODEL_DIR = os.environ.get(
    "KOKORO_MODEL_DIR",
    os.environ.get("KOKORA_MODEL_DIR", "/opt/kokoro-rknn"),
)
MODEL_FILE = os.environ.get("KOKORO_MODEL", os.environ.get("KOKORA_MODEL", "kokoro.rknn"))
VOICE = os.environ.get("KOKORO_VOICE", os.environ.get("KOKORA_VOICE", "default"))
MODE = os.environ.get("KOKORO_RKNN_MODE", os.environ.get("KOKORA_RKNN_MODE", "auto")).lower()
PREFIX_ONNX = os.environ.get("KOKORO_PREFIX_ONNX", "kokoro-prefix-cpu.onnx")
FRONT_RKNN = os.environ.get("KOKORO_FRONT_RKNN", "rk3588/kokoro-decoder-front.rknn")
TAIL_ONNX = os.environ.get("KOKORO_TAIL_ONNX", "kokoro-generator-tail-cpu.onnx")
PREFIX_ORT_INTRA_OP = int(os.environ.get("KOKORO_PREFIX_ORT_INTRA_OP", "1"))
PREFIX_ORT_INTER_OP = int(os.environ.get("KOKORO_PREFIX_ORT_INTER_OP", "1"))
TAIL_ORT_INTRA_OP = int(os.environ.get("KOKORO_TAIL_ORT_INTRA_OP", "4"))
TAIL_ORT_INTER_OP = int(os.environ.get("KOKORO_TAIL_ORT_INTER_OP", "1"))
ORT_GRAPH_OPT = os.environ.get("KOKORO_ORT_GRAPH_OPT", "all").lower()
ORT_ENABLE_CPU_MEM_ARENA = os.environ.get("KOKORO_ORT_ENABLE_CPU_MEM_ARENA", "1").lower() not in {
    "0",
    "false",
    "no",
}
ORT_ENABLE_MEM_PATTERN = os.environ.get("KOKORO_ORT_ENABLE_MEM_PATTERN", "0").lower() not in {
    "0",
    "false",
    "no",
}
ORT_ENABLE_MEM_REUSE = os.environ.get("KOKORO_ORT_ENABLE_MEM_REUSE", "1").lower() not in {
    "0",
    "false",
    "no",
}

DECODER_INPUT = "/MatMul_1_output_0"
STYLE_SLICE = "/Slice_2_output_0"
FRONT_OUTPUT = "/decoder/decode.3/Mul_output_0"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;。！？；\n])\s*")


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _trim_silence(audio: np.ndarray, threshold: float = 0.005, frame_size: int = 512) -> np.ndarray:
    if audio.size < frame_size:
        return audio
    n_frames = audio.size // frame_size
    frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    keep = np.where(rms > threshold)[0]
    if keep.size == 0:
        return audio
    return audio[keep[0] * frame_size: (keep[-1] + 1) * frame_size]


class _KokoroTokenizer:
    """Minimal token mapper for RKNN smoke/runtime.

    The tokenizer accepts either a JSON symbol map or a sherpa-style tokens.txt.
    It is deliberately conservative: unknown characters are skipped unless an
    <unk> token exists.  Production-quality G2P should live in the exported
    model package and provide a matching tokens file.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.token_to_id: dict[str, int] = {}
        self.pad_id = 0
        self.bos_id: int | None = None
        self.eos_id: int | None = None
        self.unk_id: int | None = None

    def load(self) -> None:
        json_path = self.model_dir / "tokens.json"
        txt_path = self.model_dir / "tokens.txt"
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "token_to_id" in data:
                data = data["token_to_id"]
            self.token_to_id = {str(k): int(v) for k, v in data.items()}
        elif txt_path.exists():
            self.token_to_id = self._load_tokens_txt(txt_path)
        else:
            raise FileNotFoundError(
                f"No Kokoro token file found in {self.model_dir}; expected tokens.json or tokens.txt"
            )

        self.pad_id = self._first_id("<pad>", "<blank>", "_", default=0)
        self.bos_id = self._optional_id("<s>", "<bos>", "^")
        self.eos_id = self._optional_id("</s>", "<eos>", "$")
        self.unk_id = self._optional_id("<unk>", "UNK")

    @staticmethod
    def _load_tokens_txt(path: Path) -> dict[str, int]:
        mapping: dict[str, int] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
                    token = " ".join(parts[:-1])
                    idx = int(parts[-1])
                else:
                    token = parts[0]
                    idx = line_no
                mapping[token] = idx
        return mapping

    def _optional_id(self, *tokens: str) -> int | None:
        for token in tokens:
            if token in self.token_to_id:
                return self.token_to_id[token]
        return None

    def _first_id(self, *tokens: str, default: int) -> int:
        found = self._optional_id(*tokens)
        return default if found is None else found

    def encode(self, text: str, seq_len: int) -> tuple[np.ndarray, int]:
        ids: list[int] = []
        if self.bos_id is not None:
            ids.append(self.bos_id)

        for ch in text:
            if ch.isspace():
                token_id = self.token_to_id.get(" ", self.token_to_id.get("_"))
            else:
                token_id = self.token_to_id.get(ch)
            if token_id is None:
                token_id = self.unk_id
            if token_id is not None:
                ids.append(int(token_id))

        if self.eos_id is not None:
            ids.append(self.eos_id)

        actual = min(len(ids), seq_len)
        arr = np.full((1, seq_len), self.pad_id, dtype=np.int64)
        if actual:
            arr[0, :actual] = np.asarray(ids[:actual], dtype=np.int64)
        return arr, actual


class KokoroRKNNBackend:
    """Kokoro TTS using full RKNN or a CPU/RKNN/CPU hybrid split."""

    def __init__(self) -> None:
        self.model_dir = Path(MODEL_DIR)
        self.mode = MODE
        self.model_path = Path(MODEL_FILE)
        if not self.model_path.is_absolute():
            self.model_path = self.model_dir / self.model_path
        self.prefix_path = self._resolve_model_path(PREFIX_ONNX)
        self.front_path = self._resolve_model_path(FRONT_RKNN)
        self.tail_path = self._resolve_model_path(TAIL_ONNX)
        self.sample_rate = SAMPLE_RATE
        self.seq_len = SEQ_LEN
        self._rknn = None
        self._prefix_sess = None
        self._tail_sess = None
        self._tokenizer = _KokoroTokenizer(self.model_dir)
        self._style = np.zeros((1, STYLE_DIM), dtype=np.float32)
        self._ready = False

    def _resolve_model_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.model_dir / path

    @property
    def name(self) -> str:
        return "kokoro_rknn"

    def is_ready(self) -> bool:
        if not self._ready:
            return False
        if self.mode == "hybrid":
            return self._rknn is not None and self._prefix_sess is not None and self._tail_sess is not None
        return self._rknn is not None

    def preload(self) -> None:
        from rknnlite.api import RKNNLite

        self._tokenizer.load()
        self._style = self._load_style()

        if self.mode == "auto":
            if self.prefix_path.exists() and self.front_path.exists() and self.tail_path.exists():
                self.mode = "hybrid"
            else:
                self.mode = "full"

        if self.mode == "hybrid":
            self._preload_hybrid(RKNNLite)
            return
        if self.mode != "full":
            raise ValueError("KOKORO_RKNN_MODE must be one of: auto, full, hybrid")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Kokoro RKNN model not found: {self.model_path}")
        self._rknn = RKNNLite(verbose=False)
        ret = self._rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load Kokoro RKNN {self.model_path}: ret={ret}")
        ret = self._rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init Kokoro RKNN runtime: ret={ret}")
        self._ready = True
        logger.info("Loaded Kokoro RKNN model: %s voice=%s sr=%d", self.model_path, VOICE, self.sample_rate)

    def _preload_hybrid(self, rknn_lite_cls) -> None:
        import onnxruntime as ort

        missing = [
            str(path)
            for path in (self.prefix_path, self.front_path, self.tail_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Kokoro hybrid model file(s) not found: " + ", ".join(missing))

        self._prefix_sess = self._make_ort_session(
            ort,
            self.prefix_path,
            intra_op=PREFIX_ORT_INTRA_OP,
            inter_op=PREFIX_ORT_INTER_OP,
            graph_opt=ORT_GRAPH_OPT,
        )
        self._tail_sess = self._make_ort_session(
            ort,
            self.tail_path,
            intra_op=TAIL_ORT_INTRA_OP,
            inter_op=TAIL_ORT_INTER_OP,
            graph_opt=ORT_GRAPH_OPT,
        )
        self._rknn = rknn_lite_cls(verbose=False)
        ret = self._rknn.load_rknn(str(self.front_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load Kokoro decoder-front RKNN {self.front_path}: ret={ret}")
        ret = self._rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init Kokoro decoder-front RKNN runtime: ret={ret}")
        self._ready = True
        logger.info(
            "Loaded Kokoro hybrid: prefix=%s front=%s tail=%s voice=%s sr=%d ort_graph_opt=%s prefix_threads=%d/%d tail_threads=%d/%d ort_arena=%s ort_mem_pattern=%s ort_mem_reuse=%s",
            self.prefix_path,
            self.front_path,
            self.tail_path,
            VOICE,
            self.sample_rate,
            ORT_GRAPH_OPT,
            PREFIX_ORT_INTRA_OP,
            PREFIX_ORT_INTER_OP,
            TAIL_ORT_INTRA_OP,
            TAIL_ORT_INTER_OP,
            ORT_ENABLE_CPU_MEM_ARENA,
            ORT_ENABLE_MEM_PATTERN,
            ORT_ENABLE_MEM_REUSE,
        )

    @staticmethod
    def _make_ort_session(ort, path: Path, *, intra_op: int, inter_op: int, graph_opt: str):
        options = ort.SessionOptions()
        if intra_op > 0:
            options.intra_op_num_threads = intra_op
        if inter_op > 0:
            options.inter_op_num_threads = inter_op
        levels = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        if graph_opt not in levels:
            raise ValueError("KOKORO_ORT_GRAPH_OPT must be one of: disable, basic, extended, all")
        options.graph_optimization_level = levels[graph_opt]
        options.enable_cpu_mem_arena = ORT_ENABLE_CPU_MEM_ARENA
        options.enable_mem_pattern = ORT_ENABLE_MEM_PATTERN
        options.enable_mem_reuse = ORT_ENABLE_MEM_REUSE
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def _load_style(self) -> np.ndarray:
        candidates = [
            self.model_dir / f"{VOICE}.npy",
            self.model_dir / "style.npy",
            self.model_dir / "voices.npy",
        ]
        for path in candidates:
            if path.exists():
                arr = np.load(path).astype(np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[-1] != STYLE_DIM:
                    raise ValueError(f"Kokoro style vector {path} has shape {arr.shape}, expected (*, {STYLE_DIM})")
                return arr[:1].copy()
        logger.warning("No Kokoro style .npy found in %s; using zeros", self.model_dir)
        return np.zeros((1, STYLE_DIM), dtype=np.float32)

    def cleanup(self) -> None:
        if self._rknn is not None:
            try:
                self._rknn.release()
            except Exception:
                pass
        self._rknn = None
        self._prefix_sess = None
        self._tail_sess = None
        self._ready = False

    def _infer_segment(self, text: str, speed: float) -> tuple[np.ndarray, dict]:
        tokens, n_tokens = self._tokenizer.encode(text, self.seq_len)
        meta = {"num_tokens": n_tokens}
        if n_tokens == 0:
            return np.zeros(0, dtype=np.float32), meta

        if self.mode == "hybrid":
            return self._infer_segment_hybrid(tokens, n_tokens, speed)

        t0 = time.perf_counter()
        outputs = self._rknn.inference(
            inputs=[
                tokens,
                self._style,
                np.asarray([speed], dtype=np.float32),
            ]
        )
        meta["infer_ms"] = (time.perf_counter() - t0) * 1000
        if not outputs:
            return np.zeros(0, dtype=np.float32), meta

        audio = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        audio = _trim_silence(audio)
        meta["duration_s"] = audio.size / self.sample_rate
        if meta["duration_s"] > 0:
            meta["rtf"] = meta["infer_ms"] / 1000.0 / meta["duration_s"]
        return audio, meta

    def _infer_segment_hybrid(self, tokens: np.ndarray, n_tokens: int, speed: float) -> tuple[np.ndarray, dict]:
        meta = {"num_tokens": n_tokens}
        speed_arr = np.asarray([speed], dtype=np.float32)

        t0 = time.perf_counter()
        prefix_outputs = self._prefix_sess.run(
            None,
            self._build_ort_feed(self._prefix_sess, tokens=tokens, style=self._style, speed=speed_arr),
        )
        meta["prefix_ms"] = (time.perf_counter() - t0) * 1000

        decoder_input = self._select_output(self._prefix_sess, prefix_outputs, DECODER_INPUT, 0)
        style_slice = self._select_output(self._prefix_sess, prefix_outputs, STYLE_SLICE, 1)

        t0 = time.perf_counter()
        front_outputs = self._rknn.inference(inputs=[decoder_input, style_slice])
        meta["front_ms"] = (time.perf_counter() - t0) * 1000
        if not front_outputs:
            return np.zeros(0, dtype=np.float32), meta

        hidden = np.asarray(front_outputs[0], dtype=np.float32)
        t0 = time.perf_counter()
        tail_outputs = self._tail_sess.run(
            None,
            self._build_tail_feed(self._tail_sess, hidden=hidden, style_slice=style_slice),
        )
        meta["tail_ms"] = (time.perf_counter() - t0) * 1000
        meta["infer_ms"] = meta["prefix_ms"] + meta["front_ms"] + meta["tail_ms"]
        if not tail_outputs:
            return np.zeros(0, dtype=np.float32), meta

        audio = np.asarray(tail_outputs[0]).reshape(-1).astype(np.float32)
        audio = _trim_silence(audio)
        meta["duration_s"] = audio.size / self.sample_rate
        if meta["duration_s"] > 0:
            meta["rtf"] = meta["infer_ms"] / 1000.0 / meta["duration_s"]
        return audio, meta

    @staticmethod
    def _select_output(sess, outputs: list[np.ndarray], name: str, fallback_idx: int) -> np.ndarray:
        names = [item.name for item in sess.get_outputs()]
        if name in names:
            return np.asarray(outputs[names.index(name)], dtype=np.float32)
        return np.asarray(outputs[fallback_idx], dtype=np.float32)

    @staticmethod
    def _build_ort_feed(sess, *, tokens: np.ndarray, style: np.ndarray, speed: np.ndarray) -> dict[str, np.ndarray]:
        values = {"tokens": tokens, "style": style, "speed": speed}
        feed: dict[str, np.ndarray] = {}
        for item in sess.get_inputs():
            if item.name not in values:
                raise KeyError(f"Unsupported Kokoro prefix input: {item.name}")
            feed[item.name] = values[item.name]
        return feed

    @staticmethod
    def _build_tail_feed(sess, *, hidden: np.ndarray, style_slice: np.ndarray) -> dict[str, np.ndarray]:
        values = {FRONT_OUTPUT: hidden, STYLE_SLICE: style_slice}
        feed: dict[str, np.ndarray] = {}
        for item in sess.get_inputs():
            if item.name not in values:
                raise KeyError(f"Unsupported Kokoro tail input: {item.name}")
            feed[item.name] = values[item.name]
        return feed

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        import soundfile as sf

        if not self.is_ready():
            raise RuntimeError("KokoroRKNNBackend.preload() has not been called")

        effective_speed = float(speed if speed is not None else kwargs.get("kokoro_speed", 1.0))
        t_start = time.perf_counter()
        audio_parts: list[np.ndarray] = []
        agg = {"num_tokens": 0, "infer_ms": 0.0, "prefix_ms": 0.0, "front_ms": 0.0, "tail_ms": 0.0}

        for sentence in _split_sentences(text):
            audio, meta = self._infer_segment(sentence, effective_speed)
            if audio.size:
                audio_parts.append(audio)
            agg["num_tokens"] += int(meta.get("num_tokens", 0))
            for key in ("infer_ms", "prefix_ms", "front_ms", "tail_ms"):
                agg[key] += float(meta.get(key, 0.0))

        audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.95

        duration = audio.size / self.sample_rate
        inference_time = time.perf_counter() - t_start
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), {
            "duration": duration,
            "inference_time": inference_time,
            "rtf": inference_time / duration if duration > 0 else 0.0,
            "backend": self.name,
            "mode": self.mode,
            "voice": VOICE,
            **agg,
        }

    def synthesize_stream(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        if not self.is_ready():
            raise RuntimeError("KokoroRKNNBackend.preload() has not been called")
        effective_speed = float(speed if speed is not None else kwargs.get("kokoro_speed", 1.0))
        for sentence in _split_sentences(text):
            audio, meta = self._infer_segment(sentence, effective_speed)
            if audio.size == 0:
                continue
            peak = float(np.max(np.abs(audio)))
            if peak > 0:
                audio = audio / peak * 0.95
            yield audio, {"duration": audio.size / self.sample_rate, "backend": self.name, "mode": self.mode, **meta}

    def get_sample_rate(self) -> int:
        return self.sample_rate


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Kokoro RKNN TTS smoke-test")
    parser.add_argument("--text", "-t", default="こんにちは。", help="Input text")
    parser.add_argument("--output", "-o", default="/tmp/kokoro_rknn_test.wav")
    parser.add_argument("--speed", "-s", type=float, default=1.0)
    args = parser.parse_args()

    backend = KokoroRKNNBackend()
    backend.preload()
    wav_bytes, metadata = backend.synthesize(args.text, speed=args.speed)
    Path(args.output).write_bytes(wav_bytes)
    print(f"Saved {len(wav_bytes)} bytes to {args.output}")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    backend.cleanup()

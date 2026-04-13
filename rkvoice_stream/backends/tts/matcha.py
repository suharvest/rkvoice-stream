"""
RKNN TTS Backend for RK3576

完整流程：
文本 → 文本前端 (sherpa-onnx, CPU) → tokens → Matcha RKNN (NPU) → mel → Vocos RKNN (NPU) → ISTFT (CPU) → 音频

Matcha acoustic model is compiled with fixed output shape (probe-first surgery).
Available bucket models:
  - matcha-s64.rknn:  seq_len=80,  x_len=64,  ~599 mel frames, ~9.6s, 53MB, ~430ms
  - matcha-s140.rknn: seq_len=160, x_len=140, ~1278 mel frames, ~20s, 60MB, ~900ms

Vocos vocoder compiled with fixed TIME_FRAMES:
  - vocos-16khz-600.rknn: 600 frames input, ~26.9MB, ~80ms

性能 (s64 + vocos-600, typical 50-token sentence):
- 文本前端: <10ms (CPU 查表)
- Matcha RKNN: ~430ms (NPU, s64 bucket)
- Vocos RKNN: ~80ms (NPU, 600 frames)
- ISTFT: ~50ms (CPU)
- 总计: ~570ms for 7.6s audio, RTF ~0.07
"""

from __future__ import annotations

import os
import time
import numpy as np
from pathlib import Path
from typing import Optional

# 音频参数
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
MAX_SEQ_LEN = int(os.environ.get('MATCHA_MAX_PHONEMES', '64'))
# RKNN model compiled input sequence length (must match the shape used during
# rknn.build()).  All tensor inputs to the Matcha RKNN model are padded to
# this length so that byte sizes match the static-shape expectations.
# Default 80 matches the s64 bucket model (seq_len=80, x_len=64, ~599 mel frames).
MATCHA_MODEL_SEQ_LEN = int(os.environ.get('MATCHA_MODEL_SEQ_LEN', '80'))
# Vocos model compiled time frame dimension (must match vocos RKNN build).
VOCOS_FRAMES = int(os.environ.get('VOCOS_FRAMES', '600'))

# Split model constants (encoder + estimator with CPU FP32 ODE loop)
MEL_SIGMA = 5.446792
MEL_MEAN = -2.9521978
# ODE constants for split RKNN mode (not used in ORT mode).
# The default runtime step count is 1 (env MATCHA_ODE_STEPS=1) for best
# FP16 precision; N_ODE_STEPS=3 is only used for loading time_emb files.
ODE_DT = 1.0 / 3.0
N_ODE_STEPS = 3  # number of pre-computed time_emb files (always 3)
MAX_FRAMES = 600
TIME_EMB_DIM = 256
N_TIME_BLOCKS = 6


class RKNNMatchaVocoder:
    """RKNN 加速的 Matcha TTS 引擎"""

    def __init__(
        self,
        matcha_rknn_path: str,
        vocos_rknn_path: str,
        lexicon_path: str,
        tokens_path: str,
        data_dir: str,
    ):
        self.matcha_rknn_path = matcha_rknn_path
        self.vocos_rknn_path = vocos_rknn_path
        self.lexicon_path = lexicon_path
        self.tokens_path = tokens_path
        self.data_dir = data_dir

        # 加载后的模型
        self._matcha = None
        self._matcha_backend = None  # 'rknn', 'rknn_split', or 'ort'
        self._matcha_encoder = None   # split mode: encoder RKNN
        self._matcha_estimator = None  # split mode: estimator RKNN
        self._cstsin_refs = None      # ctypes refs for CstSin custom op (prevent GC)
        self._time_emb_steps = None   # split mode: [3, 6, 256] time embeddings
        self._vocos = None
        self._lexicon = None
        self._token_to_id = None

    def load(self):
        """加载所有模型和资源"""
        import logging
        log = logging.getLogger(__name__)

        from rknnlite.api import RKNNLite

        # 加载 lexicon
        self._lexicon = {}
        with open(self.lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self._lexicon[parts[0]] = parts[1:]

        # 加载 tokens
        self._token_to_id = {}
        with open(self.tokens_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 1:
                    # token ID = 行号 + 1 (1-indexed)
                    self._token_to_id[parts[0]] = i + 1

        # 加载 Matcha 声学模型
        # Priority: 1) split RKNN (best FP16 precision), 2) single RKNN, 3) ORT fallback
        matcha_dir = os.path.dirname(self.matcha_rknn_path)
        split_dir = os.environ.get('MATCHA_SPLIT_DIR',
                                   os.path.join(matcha_dir, 'matcha-split'))
        enc_path = os.path.join(split_dir, 'matcha-encoder-fp16.rknn')
        est_path = os.path.join(split_dir, 'matcha-estimator-fp16.rknn')
        te_path = os.path.join(split_dir, 'time_emb_step0.npy')

        use_ort = os.environ.get('MATCHA_USE_ORT', '').lower() in ('1', 'true', 'yes')

        # Try split RKNN mode first (best precision: ODE loop on CPU FP32)
        if not use_ort and os.path.exists(enc_path) and os.path.exists(est_path) and os.path.exists(te_path):
            try:
                self._matcha_encoder = RKNNLite(verbose=False)
                ret = self._matcha_encoder.load_rknn(enc_path)
                if ret != 0:
                    raise RuntimeError(f"load encoder ret={ret}")
                ret = self._matcha_encoder.init_runtime(core_mask=1)
                if ret != 0:
                    raise RuntimeError(f"init encoder runtime ret={ret}")

                self._matcha_estimator = RKNNLite(verbose=False)
                ret = self._matcha_estimator.load_rknn(est_path)
                if ret != 0:
                    raise RuntimeError(f"load estimator ret={ret}")
                ret = self._matcha_estimator.init_runtime(core_mask=1)
                if ret != 0:
                    raise RuntimeError(f"init estimator runtime ret={ret}")

                # Register custom CPU ops if the .so exists
                cstops_so = os.environ.get(
                    'CSTOPS_LIB', '/opt/tts/lib/libcstops.so')
                if os.path.exists(cstops_so):
                    from rkvoice_stream.backends.custom_ops.rknn_custom_ops import register_custom_ops
                    self._cstsin_refs = register_custom_ops(
                        self._matcha_estimator,
                        lib_path=cstops_so,
                    )
                    if self._cstsin_refs is None:
                        log.warning("Custom op registration failed")
                else:
                    self._cstsin_refs = None

                self._time_emb_steps = [
                    np.load(os.path.join(split_dir, f'time_emb_step{i}.npy'))
                    for i in range(N_ODE_STEPS)
                ]
                self._matcha_backend = 'rknn_split'
                log.info("Matcha loaded via split RKNN (encoder+estimator, CPU FP32 ODE)")
            except Exception as e:
                log.warning("Matcha split RKNN load failed (%s), trying single RKNN", e)
                for m in (self._matcha_encoder, self._matcha_estimator):
                    if m is not None:
                        try:
                            m.release()
                        except Exception:
                            pass
                self._matcha_encoder = self._matcha_estimator = self._time_emb_steps = None

        # Try single RKNN
        if self._matcha_backend is None and not use_ort and os.path.exists(self.matcha_rknn_path):
            try:
                self._matcha = RKNNLite(verbose=False)
                ret = self._matcha.load_rknn(self.matcha_rknn_path)
                if ret != 0:
                    raise RuntimeError(f"load_rknn ret={ret}")
                ret = self._matcha.init_runtime(core_mask=1)
                if ret != 0:
                    raise RuntimeError(f"init_runtime ret={ret}")
                self._matcha_backend = 'rknn'
                log.info("Matcha acoustic model loaded via RKNN (single model)")
            except Exception as e:
                log.warning("Matcha RKNN load failed (%s), trying ORT fallback", e)
                if self._matcha is not None:
                    try:
                        self._matcha.release()
                    except Exception:
                        pass
                self._matcha = None

        # ORT fallback
        if self._matcha_backend is None:
            matcha_onnx_path = self.matcha_rknn_path.replace('.rknn', '.onnx')
            if not os.path.exists(matcha_onnx_path):
                alt = os.path.join(matcha_dir, 'model-steps-3.onnx')
                if os.path.exists(alt):
                    matcha_onnx_path = alt
                else:
                    matcha_onnx_path = os.environ.get('MATCHA_ONNX_PATH', matcha_onnx_path)
            if os.path.exists(matcha_onnx_path):
                import onnxruntime as ort
                self._matcha = ort.InferenceSession(
                    matcha_onnx_path,
                    providers=['CPUExecutionProvider'],
                )
                self._matcha_backend = 'ort'
                log.info("Matcha acoustic model loaded via ORT (CPU): %s", matcha_onnx_path)

        if self._matcha_backend is None:
            raise RuntimeError(
                f"无法加载 Matcha 声学模型: split={split_dir}, "
                f"RKNN={self.matcha_rknn_path}"
            )

        # 加载 Vocos RKNN
        self._vocos = RKNNLite(verbose=False)
        ret = self._vocos.load_rknn(self.vocos_rknn_path)
        if ret != 0:
            raise RuntimeError(f"加载 Vocos RKNN 失败: ret={ret}")
        ret = self._vocos.init_runtime(core_mask=1)  # NPU_CORE_0 only; CORE_1 reserved for ASR encoder
        if ret != 0:
            raise RuntimeError(f"初始化 Vocos RKNN 运行时失败: ret={ret}")

    def release(self):
        """释放资源"""
        for m in (self._matcha, self._matcha_encoder, self._matcha_estimator):
            if m is not None:
                try:
                    m.release()
                except Exception:
                    pass
        self._matcha = self._matcha_encoder = self._matcha_estimator = None
        self._cstsin_refs = None
        self._time_emb_steps = None
        self._matcha_backend = None
        if self._vocos:
            try:
                self._vocos.release()
            except Exception:
                pass
            self._vocos = None

    def _phonemize_english(self, text: str) -> list[str]:
        """
        Use espeak-ng to convert English text to IPA phonemes.

        Falls back to empty list if espeak-ng is not available.
        """
        import subprocess
        import logging
        log = logging.getLogger(__name__)

        try:
            # Use espeak-ng with the data_dir if available
            cmd = ["espeak-ng", "--ipa", "-v", "en-us", "-q", "--", text]
            env = os.environ.copy()
            if self.data_dir and os.path.isdir(self.data_dir):
                env["ESPEAK_DATA_PATH"] = self.data_dir
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5, env=env,
            )
            if result.returncode != 0:
                log.warning("espeak-ng failed (rc=%d): %s", result.returncode, result.stderr.strip())
                return []
            ipa = result.stdout.strip()
            if not ipa:
                return []
            # Split IPA string into individual phoneme characters/tokens.
            # espeak-ng outputs IPA with spaces between words and stress marks.
            # Each IPA character that exists in our token table is a valid phoneme.
            phonemes = []
            for ch in ipa:
                if ch in (' ', '\n', '\t'):
                    continue
                phonemes.append(ch)
            return phonemes
        except FileNotFoundError:
            log.warning("espeak-ng not found — English text will be skipped")
            return []
        except subprocess.TimeoutExpired:
            log.warning("espeak-ng timed out")
            return []

    def text_to_tokens(self, text: str) -> list[int]:
        """
        将文本转换为 token IDs

        中文：lexicon 查表 → phonemes → token IDs
        英文：espeak-ng IPA → token IDs
        混合文本：按语言分段处理
        """
        import re
        tokens = []

        # Split text into Chinese and non-Chinese (English/punctuation) segments
        # Each segment is (is_chinese: bool, text: str)
        segments = re.findall(r'[\u4e00-\u9fff]+|[A-Za-z][A-Za-z\' ]*[A-Za-z]|[A-Za-z]|[^\u4e00-\u9fffA-Za-z]+', text)

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            # Check if segment is Chinese
            if re.match(r'^[\u4e00-\u9fff]+$', seg):
                # Chinese: use lexicon lookup
                tokens.extend(self._chinese_to_tokens(seg))
            elif re.match(r'^[A-Za-z]', seg):
                # English: use espeak-ng phonemization
                phonemes = self._phonemize_english(seg)
                for p in phonemes:
                    if p in self._token_to_id:
                        tokens.append(self._token_to_id[p])
            # else: punctuation/whitespace — skip

        return tokens

    def _chinese_to_tokens(self, text: str) -> list[int]:
        """Convert Chinese text to token IDs via lexicon lookup."""
        tokens = []
        i = 0
        while i < len(text):
            # 尝试匹配最长词
            found = False
            for length in range(min(4, len(text) - i), 0, -1):
                word = text[i:i+length]
                if word in self._lexicon:
                    phonemes = self._lexicon[word]
                    for p in phonemes:
                        if p in self._token_to_id:
                            tokens.append(self._token_to_id[p])
                    i += length
                    found = True
                    break

            if not found:
                # 单字处理
                char = text[i]
                if char in self._lexicon:
                    phonemes = self._lexicon[char]
                    for p in phonemes:
                        if p in self._token_to_id:
                            tokens.append(self._token_to_id[p])
                i += 1

        return tokens

    def run_matcha(
        self,
        tokens: list[int],
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        运行 Matcha RKNN 声学模型

        Args:
            tokens: 音素 token IDs
            noise_scale: 噪声缩放因子
            length_scale: 时长缩放因子

        Returns:
            mel: Mel 频谱图 [1, 80, T]
            mel_frames: 有效帧数
        """
        num_tokens = len(tokens)
        x_length = np.array([num_tokens], dtype=np.int64)
        noise_scale_arr = np.array([noise_scale], dtype=np.float32)
        length_scale_arr = np.array([length_scale], dtype=np.float32)

        if self._matcha_backend == 'rknn_split':
            # Split mode: encoder (NPU) + estimator (NPU) + ODE loop (CPU FP32)
            tokens_padded = np.zeros((1, MATCHA_MODEL_SEQ_LEN), dtype=np.int64)
            tokens_padded[0, :num_tokens] = tokens
            enc_out = self._matcha_encoder.inference(
                inputs=[tokens_padded, x_length, noise_scale_arr, length_scale_arr]
            )
            mu = enc_out[0]     # [1, 80, 600]
            mask = enc_out[1]   # [1, 1, 600]
            z = enc_out[2]      # [1, 80, 600] (z0 = noise * noise_scale)

            # ODE loop on CPU (FP32 precision).
            # 1-step Euler (dt=1.0) is preferred on RK3576: fewer FP16
            # accumulation errors and 2.6x faster than 3-step.
            n_steps = int(os.environ.get('MATCHA_ODE_STEPS', '1'))
            dt = np.float32(1.0 / n_steps)
            for step in range(n_steps):
                te = self._time_emb_steps[min(step, len(self._time_emb_steps) - 1)]
                feeds = [z, mu, mask]
                for i in range(N_TIME_BLOCKS):
                    feeds.append(te[i].reshape(1, TIME_EMB_DIM, 1).astype(np.float32))
                v = self._matcha_estimator.inference(inputs=feeds)[0]
                z = z + dt * v

            # Denormalize
            mel = z * np.float32(MEL_SIGMA) + np.float32(MEL_MEAN)

        elif self._matcha_backend == 'rknn':
            tokens_padded = np.zeros((1, MATCHA_MODEL_SEQ_LEN), dtype=np.int64)
            tokens_padded[0, :num_tokens] = tokens
            mel = self._matcha.inference(
                inputs=[tokens_padded, x_length, noise_scale_arr, length_scale_arr]
            )[0]
        else:
            # ORT fallback — dynamic shapes, no padding needed
            tokens_padded = np.zeros((1, max(num_tokens, 1)), dtype=np.int64)
            tokens_padded[0, :num_tokens] = tokens
            mel = self._matcha.run(
                None,
                {
                    'x': tokens_padded,
                    'x_length': x_length,
                    'noise_scale': noise_scale_arr,
                    'length_scale': length_scale_arr,
                },
            )[0]

        # Determine valid mel frames.
        T = mel.shape[2]
        if self._matcha_backend == 'ort':
            # ORT produces dynamic output — all frames are valid.
            mel_frames = T
        else:
            # RKNN produces fixed-size output (e.g., 600 frames for split, 599 for single).
            # Estimate valid frames from token count. Calibrated against ORT:
            #   n=1→65, n=5→114, n=9→150, n=14→226, n=17→253
            # Linear fit: 11.9 * n + 51, with 20% safety margin to avoid truncation.
            est = int((11.9 * num_tokens + 51) * length_scale * 1.2 + 0.5)
            mel_frames = min(est, T)
            # Clamp to ORT-observed range as safety measure.
            mel = np.clip(mel, -25.0, 8.0)

        return mel, mel_frames

    def run_vocos(self, mel: np.ndarray, mel_frames: int) -> np.ndarray:
        """
        运行 Vocos RKNN 声码器

        Args:
            mel: Mel 频谱图 [1, 80, T]
            mel_frames: 有效帧数

        Returns:
            audio: 音频样本
        """
        # Pad mel to Vocos compiled input size
        mel_padded = np.zeros((1, 80, VOCOS_FRAMES), dtype=np.float32)
        use_frames = min(mel_frames, VOCOS_FRAMES, mel.shape[2])
        mel_padded[:, :, :use_frames] = mel[:, :, :use_frames]

        # 推理
        outputs = self._vocos.inference(inputs=[mel_padded])

        # 提取 STFT 分量
        mag = outputs[0][0]  # [513, T]
        x = outputs[1][0]    # cos 分量
        y = outputs[2][0]    # sin 分量

        # ISTFT
        audio = self._istft(mag, x, y)

        # 裁剪到正确长度
        audio = audio[:mel_frames * HOP_LENGTH]

        return audio

    def _istft(
        self,
        mag: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        逆短时傅里叶变换

        Args:
            mag: 幅度谱 [513, T]
            x: 余弦分量 (实部)
            y: 正弦分量 (虚部)

        Returns:
            audio: 重建的音频
        """
        # 重建复数频谱
        complex_spec = mag * (x + 1j * y)

        n_frames = complex_spec.shape[1]
        output_len = (n_frames - 1) * HOP_LENGTH + N_FFT

        audio = np.zeros(output_len, dtype=np.float32)
        window = np.hanning(N_FFT)

        # 重叠相加
        for i in range(n_frames):
            frame = np.fft.irfft(complex_spec[:, i], n=N_FFT) * window
            start = i * HOP_LENGTH
            audio[start:start + N_FFT] += frame

        # 归一化
        window_sum = np.zeros(output_len, dtype=np.float32)
        for i in range(n_frames):
            start = i * HOP_LENGTH
            window_sum[start:start + N_FFT] += window ** 2

        audio = audio / np.maximum(window_sum, 1e-8)

        return audio

    def _split_text(self, text: str) -> list[str]:
        """将文本按句子分割，确保每段不超过 MAX_SEQ_LEN 个音素。"""
        import re
        # 按句末标点分割
        segments = re.split(r'([。！？；!?;])', text)
        # 将标点重新附加到前一段
        result = []
        for i in range(0, len(segments), 2):
            seg = segments[i]
            if i + 1 < len(segments):
                seg += segments[i + 1]
            seg = seg.strip()
            if seg:
                result.append(seg)
        if not result:
            return [text]

        # 对仍超出 MAX_SEQ_LEN 的段，按逗号进一步拆分
        final = []
        for seg in result:
            tokens = self.text_to_tokens(seg)
            if len(tokens) <= MAX_SEQ_LEN:
                final.append(seg)
            else:
                # 按逗号拆分
                sub_segs = re.split(r'([，,])', seg)
                sub_result = []
                for j in range(0, len(sub_segs), 2):
                    s = sub_segs[j]
                    if j + 1 < len(sub_segs):
                        s += sub_segs[j + 1]
                    s = s.strip()
                    if s:
                        sub_result.append(s)
                final.extend(sub_result if sub_result else [seg])
        return final

    @staticmethod
    def _smooth_mel(mel: np.ndarray) -> np.ndarray:
        """Fix FP16 energy anomalies in RKNN mel via adaptive per-frame correction.

        RKNN FP16 estimator produces localized energy dips (40% of expected) and
        spikes (200%) at individual frames. Instead of blanket smoothing (which
        blurs good frames), we detect anomalous frames by comparing per-frame
        energy against a local median, then blend only those frames with their
        neighbors.

        Args:
            mel: [1, 80, T] mel spectrogram

        Returns:
            Corrected mel with same shape.
        """
        m = mel[0]  # [80, T]
        T = m.shape[1]
        if T < 5:
            return mel

        # Per-frame energy
        energy = np.mean(m ** 2, axis=0)  # [T]

        # Local median energy (window=5)
        pad = 2
        e_padded = np.pad(energy, pad, mode='reflect')
        # Sliding window median via sorted approach
        local_med = np.array([
            np.median(e_padded[i:i + 5]) for i in range(T)
        ])

        # Detect anomalous frames: energy ratio vs local median
        ratio = energy / (local_med + 1e-8)
        # Anomaly = frame where energy < 50% or > 180% of local median
        anomaly = (ratio < 0.5) | (ratio > 1.8)
        n_anomaly = np.sum(anomaly)
        if n_anomaly == 0:
            return mel

        # Blend anomalous frames with average of their 2 neighbors
        result = m.copy()
        for t in range(T):
            if anomaly[t]:
                left = max(0, t - 1)
                right = min(T - 1, t + 1)
                if left == t:
                    result[:, t] = m[:, right]
                elif right == t:
                    result[:, t] = m[:, left]
                else:
                    result[:, t] = (m[:, left] + m[:, right]) * 0.5

        return result[np.newaxis]

    def _synthesize_segment(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
    ) -> tuple[np.ndarray, dict]:
        """合成单个文本段（不超过 MAX_SEQ_LEN 音素）。"""
        metadata = {}

        # Step 1: 文本 → tokens
        t0 = time.perf_counter()
        tokens = self.text_to_tokens(text)
        metadata['text_frontend_ms'] = (time.perf_counter() - t0) * 1000
        metadata['num_tokens'] = len(tokens)

        if len(tokens) == 0:
            return np.zeros(0, dtype=np.float32), metadata

        # 截断超长 tokens
        tokens = tokens[:MAX_SEQ_LEN]

        # Step 2: Matcha RKNN
        t0 = time.perf_counter()
        mel, mel_frames = self.run_matcha(tokens, noise_scale, 1.0 / speed)
        metadata['matcha_ms'] = (time.perf_counter() - t0) * 1000

        # Step 3: Vocos RKNN
        t0 = time.perf_counter()
        audio = self.run_vocos(mel, mel_frames)
        metadata['vocos_ms'] = (time.perf_counter() - t0) * 1000

        metadata['duration_s'] = len(audio) / SAMPLE_RATE
        metadata['total_ms'] = sum(v for k, v in metadata.items() if k.endswith('_ms'))
        if metadata['duration_s'] > 0:
            metadata['rtf'] = metadata['total_ms'] / 1000 / metadata['duration_s']

        return audio.astype(np.float32), metadata

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
    ) -> tuple[np.ndarray, dict]:
        """
        合成语音，自动分句处理超长文本

        Args:
            text: 输入文本 (中文)
            speed: 语速 (1.0 = 正常)
            noise_scale: 噪声强度

        Returns:
            audio: 音频样本 (float32, [-1, 1])
            metadata: 元数据 (耗时等)
        """
        segments = self._split_text(text)
        all_audio = []
        total_text_frontend_ms = 0.0
        total_matcha_ms = 0.0
        total_vocos_ms = 0.0
        total_num_tokens = 0

        for seg in segments:
            audio_seg, meta_seg = self._synthesize_segment(seg, speed, noise_scale)
            if len(audio_seg) > 0:
                all_audio.append(audio_seg)
            total_text_frontend_ms += meta_seg.get('text_frontend_ms', 0.0)
            total_matcha_ms += meta_seg.get('matcha_ms', 0.0)
            total_vocos_ms += meta_seg.get('vocos_ms', 0.0)
            total_num_tokens += meta_seg.get('num_tokens', 0)

        audio = np.concatenate(all_audio) if all_audio else np.zeros(0, dtype=np.float32)

        # 归一化 (guard against empty audio)
        if len(audio) > 0 and np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95

        metadata = {
            'num_tokens': total_num_tokens,
            'text_frontend_ms': total_text_frontend_ms,
            'matcha_ms': total_matcha_ms,
            'vocos_ms': total_vocos_ms,
        }
        metadata['duration_s'] = len(audio) / SAMPLE_RATE
        metadata['total_ms'] = sum(v for k, v in metadata.items() if k.endswith('_ms'))
        if metadata['duration_s'] > 0:
            metadata['rtf'] = metadata['total_ms'] / 1000 / metadata['duration_s']

        return audio.astype(np.float32), metadata


def create_rknn_tts_backend(model_dir: str = None) -> RKNNMatchaVocoder:
    """
    创建 RKNN TTS 后端

    Args:
        model_dir: 模型目录，默认从环境变量获取
    """
    if model_dir is None:
        model_dir = os.environ.get('TTS_MODEL_DIR', '/home/cat/models')

    model_dir = Path(model_dir)

    matcha_name = os.environ.get('MATCHA_MODEL', 'matcha-s64.rknn')
    vocos_name = os.environ.get('VOCOS_MODEL', 'vocos-16khz-600.rknn')

    return RKNNMatchaVocoder(
        matcha_rknn_path=str(model_dir / matcha_name),
        vocos_rknn_path=str(model_dir / vocos_name),
        lexicon_path=str(model_dir / 'matcha-icefall-zh-en' / 'lexicon.txt'),
        tokens_path=str(model_dir / 'matcha-icefall-zh-en' / 'tokens.txt'),
        data_dir=str(model_dir / 'matcha-icefall-zh-en' / 'espeak-ng-data'),
    )


class MatchaRKNNBackend:
    """TTSBackend wrapper around RKNNMatchaVocoder.

    Select via TTS_BACKEND=matcha_rknn.

    Note: intentionally duck-typed (not inheriting TTSBackend) to avoid
    importing tts_backend at module level. The synthesize_stream() fallback
    is provided explicitly below.
    """

    def __init__(self) -> None:
        self._engine: Optional[RKNNMatchaVocoder] = None

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "matcha_rknn"

    def is_ready(self) -> bool:
        """Return True if the engine is loaded and ready."""
        return self._engine is not None and self._engine._matcha_backend is not None

    def preload(self) -> None:
        """Create and load RKNNMatchaVocoder. Called once at startup."""
        self._engine = create_rknn_tts_backend()
        self._engine.load()

    def get_sample_rate(self) -> int:
        """Return audio sample rate in Hz."""
        return SAMPLE_RATE

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Synthesize text to WAV bytes.

        Args:
            text: Input text (Chinese / mixed Chinese-English).
            speaker_id: Ignored (Matcha model has a single speaker).
            speed: Speech rate multiplier (1.0 = normal). Defaults to 1.0.
            pitch_shift: Ignored (not supported by this backend).
            **kwargs: Forwarded to engine.synthesize() (e.g. noise_scale).

        Returns:
            wav_bytes: PCM audio encoded as a WAV file.
            metadata: Dict with keys ``duration``, ``inference_time``, ``rtf``
                      plus per-stage timing from the engine.
        """
        import io
        import soundfile as sf

        if self._engine is None:
            raise RuntimeError("MatchaRKNNBackend.preload() has not been called")

        t_start = time.perf_counter()
        audio, engine_meta = self._engine.synthesize(
            text,
            speed=speed if speed is not None else 1.0,
            **{k: v for k, v in kwargs.items() if k in ("noise_scale",)},
        )
        inference_time = time.perf_counter() - t_start

        # Guard: if no audio was produced (e.g. text yielded no phonemes),
        # return a short silence instead of crashing downstream.
        if len(audio) == 0:
            import logging
            logging.getLogger(__name__).warning(
                "No audio produced for text: %r — returning 0.1s silence", text
            )
            audio = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)

        # Encode float32 audio → WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        duration = engine_meta.get("duration_s", len(audio) / SAMPLE_RATE)
        rtf = engine_meta.get("rtf", inference_time / duration if duration > 0 else 0.0)

        metadata = {
            "duration": duration,
            "inference_time": inference_time,
            "rtf": rtf,
            **engine_meta,
        }
        return wav_bytes, metadata

    def synthesize_stream(self, text, speaker_id=0, speed=None, pitch_shift=None, **kwargs):
        """Yield (audio_float32_chunk, metadata). Non-streaming fallback."""
        import io as _io
        import soundfile as sf

        wav_bytes, meta = self.synthesize(
            text=text, speaker_id=speaker_id, speed=speed,
            pitch_shift=pitch_shift, **kwargs,
        )
        buf = _io.BytesIO(wav_bytes)
        audio, _ = sf.read(buf, dtype="float32")
        yield audio, meta

    def cleanup(self) -> None:
        """Release RKNN resources."""
        if self._engine is not None:
            self._engine.release()
            self._engine = None


# 命令行测试
if __name__ == '__main__':
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description='RKNN TTS 测试')
    parser.add_argument('--text', '-t', default='你好世界', help='输入文本')
    parser.add_argument('--output', '-o', default='/tmp/rknn_tts.wav', help='输出文件')
    parser.add_argument('--speed', '-s', type=float, default=1.0, help='语速')
    args = parser.parse_args()

    print(f"输入: {args.text}")
    print("\n加载模型...")

    engine = create_rknn_tts_backend()
    engine.load()
    print("模型加载完成")

    print("\n合成中...")
    audio, meta = engine.synthesize(args.text, speed=args.speed)

    print(f"\n结果:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    if len(audio) > 0:
        sf.write(args.output, audio, SAMPLE_RATE)
        print(f"\n保存: {args.output}")

    engine.release()
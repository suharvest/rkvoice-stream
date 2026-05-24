"""MOSS-TTS-Nano RKNN backend.

This backend wraps a native RKNN worker process using the same JSONL streaming
contract as the Jetson MOSS worker. The Python layer is intentionally strict:
it validates the artifact bundle before preload so a production service cannot
silently start with missing or stale engines.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import soundfile as sf

from rkvoice_stream.engine.tts import TTSBackend

logger = logging.getLogger(__name__)


class MossArtifactError(RuntimeError):
    """Raised when a MOSS RKNN artifact bundle is not production-valid."""


class _WorkerDeadError(RuntimeError):
    """Raised when the worker exits or cannot be reached."""


class _WorkerRequestError(RuntimeError):
    """Raised for structured per-request worker failures."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def validate_moss_artifacts(
    model_dir: str | Path,
    manifest_name: str = "moss-rknn-manifest.json",
    *,
    require_production_default: bool = False,
) -> dict[str, Any]:
    """Validate a MOSS RKNN bundle and return the parsed manifest."""

    root = Path(model_dir)
    manifest_path = root / manifest_name
    if not manifest_path.exists():
        raise MossArtifactError(f"Missing MOSS manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MossArtifactError(f"Invalid JSON manifest: {manifest_path}: {exc}") from exc

    if manifest.get("model_id") not in {"moss-tts-nano", "moss-tts-nano-rknn"}:
        raise MossArtifactError(f"Unexpected MOSS model_id: {manifest.get('model_id')!r}")
    target = str(manifest.get("target_platform", "")).lower()
    if target not in {"rk3576", "rk3588"}:
        raise MossArtifactError(f"target_platform must be rk3576 or rk3588, got {target!r}")
    if int(manifest.get("sample_rate", 0)) <= 0:
        raise MossArtifactError("manifest sample_rate must be positive")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise MossArtifactError("manifest artifacts must be a non-empty list")

    for item in artifacts:
        if not isinstance(item, dict):
            raise MossArtifactError(f"artifact entry must be object, got {item!r}")
        required = bool(item.get("required", True))
        rel = item.get("path")
        if not rel:
            raise MossArtifactError(f"artifact missing path: {item!r}")
        path = root / str(rel)
        if not path.exists():
            if required:
                raise MossArtifactError(f"Missing required MOSS artifact: {path}")
            continue
        if path.is_dir():
            raise MossArtifactError(f"Artifact path is a directory: {path}")
        expected_size = item.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise MossArtifactError(
                f"Size mismatch for {path}: got {path.stat().st_size}, expected {expected_size}"
            )
        expected_sha = item.get("sha256")
        if expected_sha:
            actual_sha = _sha256_file(path)
            if actual_sha.lower() != str(expected_sha).lower():
                raise MossArtifactError(f"sha256 mismatch for {path}: got {actual_sha}, expected {expected_sha}")

    gates = manifest.get("production_gates", {})
    if gates:
        ttfa = gates.get("max_ttfa_ms")
        rtf = gates.get("max_rtf")
        cer = gates.get("max_asr_cer")
        if ttfa is not None and float(ttfa) <= 0:
            raise MossArtifactError("production_gates.max_ttfa_ms must be positive")
        if rtf is not None and float(rtf) <= 0:
            raise MossArtifactError("production_gates.max_rtf must be positive")
        if cer is not None and not (0 <= float(cer) <= 1):
            raise MossArtifactError("production_gates.max_asr_cer must be in [0, 1]")

    quality = manifest.get("quality_status", {})
    if quality:
        if not isinstance(quality, dict):
            raise MossArtifactError("quality_status must be an object")
        if quality.get("production_default") is True:
            evidence = quality.get("production_evidence")
            if not isinstance(evidence, dict) or evidence.get("passed") is not True:
                raise MossArtifactError("quality_status.production_default=true requires production_evidence.passed=true")
            checks = evidence.get("checks", {})
            required_checks = {"artifact_manifest", "service_streaming", "backend_stage", "roundtrip_quality"}
            if not isinstance(checks, dict) or any(checks.get(check) is not True for check in required_checks):
                raise MossArtifactError(
                    "quality_status.production_default=true requires production_evidence checks for "
                    "artifact_manifest, service_streaming, backend_stage, and roundtrip_quality"
                )

    if require_production_default and (not isinstance(quality, dict) or quality.get("production_default") is not True):
        raise MossArtifactError(
            "MOSS_RKNN_REQUIRE_PRODUCTION_DEFAULT=1 requires "
            "quality_status.production_default=true with full production evidence"
        )

    return manifest


class MossRKNNBackend(TTSBackend):
    """MOSS-TTS-Nano worker backend for Rockchip RKNN artifacts."""

    supports_streaming = True

    _CONTROL_TIMEOUT_S = 30.0
    _REQUEST_TIMEOUT_S = 60.0
    _SHUTDOWN_TIMEOUT_S = 5.0

    def __init__(self):
        self._model_dir = Path(os.environ.get("MOSS_RKNN_MODEL_DIR", os.environ.get("MODEL_DIR", "/opt/tts/models/moss")))
        self._manifest_name = os.environ.get("MOSS_RKNN_MANIFEST", "moss-rknn-manifest.json")
        self._worker_bin = os.environ.get("MOSS_RKNN_WORKER_BIN", "/opt/rkvoice-workers/moss_rknn_worker")
        self._sample_rate = int(os.environ.get("MOSS_RKNN_SAMPLE_RATE", "48000"))
        self._channels = int(os.environ.get("MOSS_RKNN_CHANNELS", "2"))
        self._max_seq_len = int(os.environ.get("MOSS_RKNN_MAX_SEQ_LEN", "1024"))
        self._chunk_frames = int(os.environ.get("MOSS_RKNN_CHUNK_FRAMES", "4"))
        self._require_production_default = _env_flag("MOSS_RKNN_REQUIRE_PRODUCTION_DEFAULT")
        self._manifest: dict[str, Any] | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._proc_lock = threading.Lock()
        self._queues_lock = threading.Lock()
        self._request_queues: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._control_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._thread_local = threading.local()

    @property
    def name(self) -> str:
        return "moss_rknn"

    def is_ready(self) -> bool:
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    def preload(self) -> None:
        self._manifest = validate_moss_artifacts(
            self._model_dir,
            self._manifest_name,
            require_production_default=self._require_production_default,
        )
        self._sample_rate = int(self._manifest.get("sample_rate", self._sample_rate))
        self._channels = int(self._manifest.get("channels", self._channels))
        if self._channels <= 0:
            raise MossArtifactError(f"MOSS_RKNN_CHANNELS must be positive, got {self._channels}")

        worker = Path(self._worker_bin)
        if not worker.exists():
            raise FileNotFoundError(f"MOSS RKNN worker binary not found: {worker}")
        if not os.access(worker, os.X_OK):
            raise PermissionError(f"MOSS RKNN worker is not executable: {worker}")

        with self._proc_lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            self._terminate_locked()
            self._control_queue = queue.Queue()
            with self._queues_lock:
                self._request_queues.clear()

            cmd = [
                self._worker_bin,
                f"--model-dir={self._model_dir}",
                f"--manifest={self._model_dir / self._manifest_name}",
                f"--max-seq-len={self._max_seq_len}",
            ]
            logger.info("Starting MOSS RKNN worker: %s", " ".join(cmd))
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=False,
            )
            self._stdout_thread = threading.Thread(target=self._stdout_reader, args=(self._proc,), daemon=True)
            self._stderr_thread = threading.Thread(target=self._stderr_drain, args=(self._proc,), daemon=True)
            self._stdout_thread.start()
            self._stderr_thread.start()

        deadline = time.monotonic() + self._CONTROL_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.cleanup()
                raise TimeoutError("Timed out waiting for MOSS RKNN worker_ready event")
            try:
                event = self._control_queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                with self._proc_lock:
                    proc = self._proc
                    returncode = proc.poll() if proc is not None else None
                if returncode is not None:
                    raise RuntimeError(f"MOSS RKNN worker exited during preload with code {returncode}")
                continue
            if event.get("event") == "worker_ready":
                logger.info("MOSS RKNN worker ready: %s", event)
                return
            if event.get("event") == "worker_exit":
                raise RuntimeError(f"MOSS RKNN worker exited during preload: {event}")

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs: Any,
    ) -> tuple[bytes, dict]:
        start = time.monotonic()
        chunks = list(self.synthesize_stream(text, speaker_id=speaker_id, speed=speed, pitch_shift=pitch_shift, **kwargs))
        if chunks:
            audio = np.concatenate([c for c, _ in chunks]).astype(np.float32, copy=False)
            meta = dict(getattr(self._thread_local, "last_stream_metadata", {}) or chunks[-1][1])
        else:
            audio = np.zeros((0,), dtype=np.float32)
            meta = {}
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, self._sample_rate, format="WAV", subtype="PCM_16")
        elapsed = time.monotonic() - start
        duration = len(audio) / float(self._sample_rate)
        meta.setdefault("sample_rate", self._sample_rate)
        meta.setdefault("channels", self._channels)
        meta["duration"] = duration
        meta["inference_time"] = elapsed
        meta["rtf"] = elapsed / duration if duration > 0 else None
        meta["wall_ms"] = int(round(elapsed * 1000.0))
        return wav_io.getvalue(), meta

    def synthesize_stream(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs: Any,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        if not self.is_ready():
            raise RuntimeError("MOSS RKNN backend not loaded; call preload() first")

        request_id = uuid.uuid4().hex
        request = {
            "id": request_id,
            "text": text,
            "stream": True,
            "chunk_transport": "base64",
            "chunk_format": "pcm_s16le",
            "chunk_frames": int(kwargs.get("chunk_frames", self._chunk_frames)),
        }
        if speed is not None:
            request["speed"] = float(speed)

        request_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        first_chunk_ms: float | None = None
        start = time.monotonic()
        self._register_request_queue(request_id, request_queue)
        try:
            self._send_request(request)
            while True:
                try:
                    event = request_queue.get(timeout=self._REQUEST_TIMEOUT_S)
                except queue.Empty as exc:
                    raise TimeoutError(f"Timed out waiting for MOSS RKNN chunk for request {request_id}") from exc
                kind = event.get("event")
                if kind == "ready":
                    continue
                if kind == "chunk":
                    pcm = self._decode_pcm_event(event, request_id)
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.monotonic() - start) * 1000.0
                    if pcm.size:
                        yield pcm, {"chunk_index": event.get("chunk_index"), "ttfa_ms": int(round(first_chunk_ms))}
                    continue
                if kind == "done":
                    meta = {
                        "ttfa_ms": event.get("ttfa_ms", int(round(first_chunk_ms)) if first_chunk_ms is not None else None),
                        "wall_ms": event.get("wall_ms", int(round((time.monotonic() - start) * 1000.0))),
                        "total_frames": event.get("total_frames"),
                        "sample_rate": self._sample_rate,
                        "channels": self._channels,
                    }
                    self._thread_local.last_stream_metadata = meta
                    return
                if kind == "error":
                    raise _WorkerRequestError(event.get("message", "unknown MOSS RKNN worker error"))
                if kind == "worker_exit":
                    raise _WorkerDeadError(f"MOSS RKNN worker exited during request {request_id}: {event}")
        finally:
            self._forget_request_queue(request_id, request_queue)

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def cleanup(self) -> None:
        with self._proc_lock:
            self._terminate_locked()
        with self._queues_lock:
            self._request_queues.clear()

    def _decode_pcm_event(self, event: dict[str, Any], request_id: str) -> np.ndarray:
        data = event.get("audio_b64") or event.get("data")
        if not isinstance(data, str):
            raise _WorkerRequestError(f"MOSS RKNN chunk missing base64 data for request {request_id}")
        raw = base64.b64decode(data, validate=True)
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if self._channels > 1 and pcm.size:
            pcm = pcm.reshape(-1, self._channels)
        return pcm

    def _send_request(self, request: dict[str, Any]) -> None:
        line = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        with self._proc_lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                raise _WorkerDeadError("MOSS RKNN worker is not running")
            proc.stdin.write(line)
            proc.stdin.flush()

    def _register_request_queue(self, request_id: str, request_queue: queue.Queue[dict[str, Any]]) -> None:
        with self._queues_lock:
            self._request_queues[request_id] = request_queue

    def _forget_request_queue(self, request_id: str, request_queue: queue.Queue[dict[str, Any]]) -> None:
        with self._queues_lock:
            if self._request_queues.get(request_id) is request_queue:
                self._request_queues.pop(request_id, None)

    def _stdout_reader(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stdout is None:
            self._publish_worker_exit(proc, "stdout unavailable")
            return
        try:
            while True:
                raw = proc.stdout.readline()
                if raw == b"":
                    self._publish_worker_exit(proc, "stdout eof")
                    return
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON MOSS RKNN stdout: %s", line)
                    continue
                if isinstance(event, dict):
                    self._route_stdout_event(event)
        except Exception as exc:
            logger.error("MOSS RKNN stdout reader failed: %s", exc)
            self._publish_worker_exit(proc, f"stdout reader failed: {exc}")

    def _stderr_drain(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stderr is None:
            return
        try:
            while True:
                raw = proc.stderr.readline()
                if raw == b"":
                    return
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    logger.debug("MOSS RKNN stderr: %s", line)
        except Exception:
            return

    def _route_stdout_event(self, event: dict[str, Any]) -> None:
        if event.get("event") == "worker_ready":
            self._control_queue.put(event)
            return
        request_id = event.get("id")
        if isinstance(request_id, str) and request_id:
            with self._queues_lock:
                request_queue = self._request_queues.get(request_id)
            if request_queue is not None:
                request_queue.put(event)
            return
        logger.debug("Dropping MOSS RKNN event without request id: %s", event)

    def _publish_worker_exit(self, proc: subprocess.Popen[bytes], reason: str) -> None:
        event = {"event": "worker_exit", "returncode": proc.poll(), "message": reason}
        self._control_queue.put(event)
        with self._queues_lock:
            queues = list(self._request_queues.values())
        for request_queue in queues:
            request_queue.put(event)

    def _terminate_locked(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=self._SHUTDOWN_TIMEOUT_S)
            return
        except subprocess.TimeoutExpired:
            pass
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)

"""RK1828 C++ worker subprocess manager (RKNN3 / PCIe coprocessor).

Generic manager for the on-device RK1828 C++ demo binaries running in
*server mode*: the binary loads the model once (Init) then loops reading
requests on stdin and emitting results on stdout.  This module is the Python
client of that protocol.

Phase 1 (Qwen3-TTS) protocol (unchanged from the existing C++ server mode):
  - stdin:  one UTF-8 text line per request (terminated by ``\\n``).
  - stdout: a stream of length-prefixed binary frames:
        [uint32 LE len]
          * len == 0xFFFFFFFF        -> end-of-utterance sentinel
          * otherwise                -> ``len`` bytes of int16 LE PCM audio
  - stderr: free-form diagnostics (drained to logging on a background thread).

Concurrency: RK1828 is a single PCIe device.  All requests through a single
worker are serialized by ``threading.Lock`` (independent of the host RKNN2
``get_npu_lock`` used by RK3576/88 backends).

The manager is intentionally model-agnostic so Phase 2 (Gemma-4 AudioLLM)
can reuse spawn / lifecycle / stderr handling with its own (versioned)
protocol layered on top.

Phase 2 (Gemma-4 AudioLLM) protocol (spec §5; a *different* binary, NOT the
TTS worker — the two protocols never share a process):
  - handshake: the binary prints ``READY <proto_version>`` on stderr once Init
    (LLM + audio encoder load) completes; the manager records the negotiated
    version.
  - stdin:  one JSON request line per turn, e.g.
        {"audio_ref": "/path/to.wav", "prompt": "...", "max_new_tokens": 256}
    (UTF-8, terminated by ``\\n``). ``audio_ref`` is a file path *or* a
    ``pcm://`` reference the binary knows how to resolve.
  - stdout: a stream of length-prefixed *text* token frames:
        [uint32 LE len]
          * len == 0xFFFFFFFE        -> end-of-stream (EOS) sentinel
          * otherwise                -> ``len`` bytes of UTF-8 text (one token /
                                        token-group)
    This deliberately uses a *distinct* EOS value (0xFFFFFFFE) from the TTS
    end-of-utterance sentinel (0xFFFFFFFF) so the two protocols can never be
    confused.
"""

from __future__ import annotations

import json
import logging
import struct
import subprocess
import threading
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

# Sentinel marking the end of one utterance in the TTS (Phase 1) stdout stream.
END_OF_UTTERANCE = 0xFFFFFFFF

# Sentinel marking end-of-stream in the AudioLLM (Phase 2) text-token stream.
# Distinct from END_OF_UTTERANCE so the two protocols are never confused.
END_OF_STREAM = 0xFFFFFFFE

# Protocol version the Python client speaks for the AudioLLM text protocol.
AUDIO_LLM_PROTOCOL_VERSION = 1

# Length prefix is a 32-bit little-endian unsigned int.
_LEN_STRUCT = struct.Struct("<I")
_LEN_BYTES = _LEN_STRUCT.size  # 4


class WorkerCrashError(RuntimeError):
    """Raised when the worker process dies mid-request (stdout EOF)."""


class ProtocolMismatchError(RuntimeError):
    """Raised when the worker handshake reports an unsupported protocol version."""


class RKNN3Worker:
    """Manage one RK1828 C++ demo subprocess in server mode."""

    def __init__(
        self,
        binary_path: str,
        model_dir: str,
        ref_speaker: str = "girl_base",
        device_id: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        ready_timeout_s: float = 120.0,
    ) -> None:
        self._binary_path = binary_path
        self._model_dir = model_dir
        self._ref_speaker = ref_speaker
        self._device_id = device_id
        self._extra_args = list(extra_args) if extra_args else []
        self._ready_timeout_s = ready_timeout_s

        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ready = False

    # ── argv construction ────────────────────────────────────────

    def _build_args(self) -> List[str]:
        """Build the subprocess argv.

        The existing TTS demo is invoked positionally::

            rknn_qwen3_tts_demo <model_dir> <ref_speaker> -

        where the trailing ``-`` selects server mode (read stdin loop).  The
        device id is passed as ``--device-id <id>`` per spec §6 (the C++ side
        is being regularised to accept this flag).  ``extra_args`` allows mock
        workers / future binaries to receive additional positional args.
        """
        args: List[str] = [self._binary_path, self._model_dir, self._ref_speaker]
        if self._device_id:
            args += ["--device-id", str(self._device_id)]
        args += self._extra_args
        args.append("-")  # server mode sentinel (stdin loop)
        return args

    # ── lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the worker and wait for the ready handshake.

        Ready handshake: the binary prints a line containing ``READY`` on
        stderr once Init (decoder + talker load) completes.  We also treat an
        early process exit as a failed start.
        """
        if self._proc is not None and self._proc.poll() is None:
            return  # already running

        args = self._build_args()
        logger.info("Spawning RK1828 worker: %s", " ".join(args))
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        ready_event = threading.Event()
        self._start_stderr_pump(ready_event)

        if not ready_event.wait(timeout=self._ready_timeout_s):
            # Either Init is slow or the binary never signals ready.  If the
            # process is still alive we optimistically mark ready (some
            # binaries may not emit an explicit token); if it died, fail.
            if self._proc.poll() is not None:
                self._ready = False
                raise WorkerCrashError(
                    f"RK1828 worker exited during startup (rc={self._proc.returncode})"
                )
            logger.warning(
                "RK1828 worker ready handshake not seen in %.0fs; "
                "assuming ready (process alive)",
                self._ready_timeout_s,
            )

        self._ready = self._proc.poll() is None
        if not self._ready:
            raise WorkerCrashError(
                f"RK1828 worker not alive after start (rc={self._proc.returncode})"
            )
        logger.info("RK1828 worker ready")

    def _start_stderr_pump(self, ready_event: threading.Event) -> None:
        """Drain stderr to logging on a daemon thread; set ready_event on READY."""

        def _pump() -> None:
            assert self._proc is not None and self._proc.stderr is not None
            for raw in iter(self._proc.stderr.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                if "READY" in line:
                    ready_event.set()
                logger.debug("[rk1828-worker] %s", line)

        self._stderr_thread = threading.Thread(
            target=_pump, name="rk1828-stderr", daemon=True
        )
        self._stderr_thread.start()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def is_ready(self) -> bool:
        return self._ready and self.is_alive()

    def stop(self) -> None:
        """Stop the worker: close stdin (EOF), wait, then kill as a fallback."""
        self._ready = False
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()  # EOF -> server loop exits
                except Exception:
                    pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("RK1828 worker did not exit on EOF; killing")
                proc.kill()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    logger.error("RK1828 worker unresponsive to kill")
        finally:
            self._proc = None

    # ── request / response ───────────────────────────────────────

    def _read_exact(self, n: int) -> bytes:
        """Read exactly ``n`` bytes from stdout; raise WorkerCrashError on EOF.

        A short read (EOF before ``n`` bytes) means the worker crashed or
        closed the pipe mid-frame — surfaced as a crash, never silently.
        """
        assert self._proc is not None and self._proc.stdout is not None
        buf = bytearray()
        stdout = self._proc.stdout
        while len(buf) < n:
            chunk = stdout.read(n - len(buf))
            if not chunk:
                self._ready = False
                raise WorkerCrashError(
                    f"RK1828 worker stdout EOF (wanted {n} bytes, got {len(buf)})"
                )
            buf += chunk
        return bytes(buf)

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Send one text line and yield int16 PCM chunks until end-of-utterance.

        Yields raw ``bytes`` (int16 LE PCM).  Serialized via the worker lock so
        concurrent callers cannot interleave on the single device.
        """
        if not self.is_ready():
            raise WorkerCrashError("RK1828 worker is not ready")

        with self._lock:
            assert self._proc is not None and self._proc.stdin is not None
            line = (text.replace("\n", " ").strip() + "\n").encode("utf-8")
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._ready = False
                raise WorkerCrashError(f"RK1828 worker stdin write failed: {exc}")

            while True:
                length = _LEN_STRUCT.unpack(self._read_exact(_LEN_BYTES))[0]
                if length == END_OF_UTTERANCE:
                    return
                if length == 0:
                    # Empty (non-sentinel) frame: nothing to emit, keep reading.
                    continue
                yield self._read_exact(length)

    def synthesize(self, text: str) -> bytes:
        """Send one text line and return the full concatenated int16 PCM bytes."""
        return b"".join(self.synthesize_stream(text))


class AudioLLMWorker:
    """Manage one RK1828 AudioLLM (Gemma-4) C++ demo subprocess in server mode.

    Phase 2 protocol (spec §5): JSON request line in, length-prefixed UTF-8 text
    token frames out, ``END_OF_STREAM`` (0xFFFFFFFE) sentinel per turn.  The
    ready handshake also negotiates a protocol version (``READY <version>``).

    This is a *separate* binary / protocol from :class:`RKNN3Worker` (TTS); it
    reuses the spawn / lifecycle / stderr-pump / framed-read *patterns* but never
    shares a process or sentinel with the TTS worker.
    """

    def __init__(
        self,
        binary_path: str,
        model_dir: str,
        device_id: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        ready_timeout_s: float = 180.0,
        protocol_version: int = AUDIO_LLM_PROTOCOL_VERSION,
    ) -> None:
        self._binary_path = binary_path
        self._model_dir = model_dir
        self._device_id = device_id
        self._extra_args = list(extra_args) if extra_args else []
        self._ready_timeout_s = ready_timeout_s
        self._client_protocol_version = protocol_version

        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ready = False
        self._worker_protocol_version: Optional[int] = None

    # ── argv construction ────────────────────────────────────────

    def _build_args(self) -> List[str]:
        """Build the subprocess argv for the gemma4 server-mode binary.

        Mirrors the TTS worker shape: ``<binary> <model_dir> [--device-id <id>]
        [extra...] -`` where the trailing ``-`` selects server mode (stdin loop).
        """
        args: List[str] = [self._binary_path, self._model_dir]
        if self._device_id:
            args += ["--device-id", str(self._device_id)]
        args += self._extra_args
        args.append("-")  # server mode sentinel (stdin loop)
        return args

    # ── lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the worker and wait for the ``READY <version>`` handshake.

        The negotiated protocol version is validated against the client's
        supported version; a mismatch raises :class:`ProtocolMismatchError`.
        """
        if self._proc is not None and self._proc.poll() is None:
            return  # already running

        args = self._build_args()
        logger.info("Spawning RK1828 AudioLLM worker: %s", " ".join(args))
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        ready_event = threading.Event()
        self._start_stderr_pump(ready_event)

        if not ready_event.wait(timeout=self._ready_timeout_s):
            if self._proc.poll() is not None:
                self._ready = False
                raise WorkerCrashError(
                    f"RK1828 AudioLLM worker exited during startup (rc={self._proc.returncode})"
                )
            logger.warning(
                "RK1828 AudioLLM worker ready handshake not seen in %.0fs; "
                "assuming ready (process alive)",
                self._ready_timeout_s,
            )

        self._ready = self._proc.poll() is None
        if not self._ready:
            raise WorkerCrashError(
                f"RK1828 AudioLLM worker not alive after start (rc={self._proc.returncode})"
            )

        # Validate the negotiated protocol version (if the worker reported one).
        if (
            self._worker_protocol_version is not None
            and self._worker_protocol_version != self._client_protocol_version
        ):
            self._ready = False
            self.stop()
            raise ProtocolMismatchError(
                f"RK1828 AudioLLM worker protocol v{self._worker_protocol_version} "
                f"!= client v{self._client_protocol_version}"
            )
        logger.info(
            "RK1828 AudioLLM worker ready (protocol v%s)",
            self._worker_protocol_version
            if self._worker_protocol_version is not None
            else self._client_protocol_version,
        )

    def _start_stderr_pump(self, ready_event: threading.Event) -> None:
        """Drain stderr to logging; parse ``READY [<version>]`` to set ready."""

        def _pump() -> None:
            assert self._proc is not None and self._proc.stderr is not None
            for raw in iter(self._proc.stderr.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                if "READY" in line:
                    self._parse_ready(line)
                    ready_event.set()
                logger.debug("[rk1828-audiollm-worker] %s", line)

        self._stderr_thread = threading.Thread(
            target=_pump, name="rk1828-audiollm-stderr", daemon=True
        )
        self._stderr_thread.start()

    def _parse_ready(self, line: str) -> None:
        """Parse the version out of a ``READY <version>`` handshake line."""
        for tok in line.split():
            if tok == "READY":
                continue
            v = tok.lstrip("vV")
            try:
                self._worker_protocol_version = int(v)
            except ValueError:
                continue
            break

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def is_ready(self) -> bool:
        return self._ready and self.is_alive()

    @property
    def protocol_version(self) -> Optional[int]:
        return self._worker_protocol_version

    def stop(self) -> None:
        """Stop the worker: close stdin (EOF), wait, then kill as a fallback."""
        self._ready = False
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()  # EOF -> server loop exits
                except Exception:
                    pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("RK1828 AudioLLM worker did not exit on EOF; killing")
                proc.kill()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    logger.error("RK1828 AudioLLM worker unresponsive to kill")
        finally:
            self._proc = None

    # ── request / response ───────────────────────────────────────

    def _read_exact(self, n: int) -> bytes:
        """Read exactly ``n`` bytes from stdout; raise WorkerCrashError on EOF."""
        assert self._proc is not None and self._proc.stdout is not None
        buf = bytearray()
        stdout = self._proc.stdout
        while len(buf) < n:
            chunk = stdout.read(n - len(buf))
            if not chunk:
                self._ready = False
                raise WorkerCrashError(
                    f"RK1828 AudioLLM worker stdout EOF (wanted {n} bytes, got {len(buf)})"
                )
            buf += chunk
        return bytes(buf)

    def generate_stream(
        self,
        audio_ref: str,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Send one JSON request and yield UTF-8 text tokens until EOS.

        Serialized via the worker lock so concurrent callers cannot interleave
        on the single device.
        """
        if not self.is_ready():
            raise WorkerCrashError("RK1828 AudioLLM worker is not ready")

        request: dict = {"audio_ref": audio_ref}
        if prompt is not None:
            request["prompt"] = prompt
        if max_new_tokens is not None:
            request["max_new_tokens"] = int(max_new_tokens)

        with self._lock:
            assert self._proc is not None and self._proc.stdin is not None
            line = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._ready = False
                raise WorkerCrashError(f"RK1828 AudioLLM worker stdin write failed: {exc}")

            while True:
                length = _LEN_STRUCT.unpack(self._read_exact(_LEN_BYTES))[0]
                if length == END_OF_STREAM:
                    return
                if length == 0:
                    # Empty (non-sentinel) frame: nothing to emit, keep reading.
                    continue
                yield self._read_exact(length).decode("utf-8", errors="replace")

    def generate(
        self,
        audio_ref: str,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Send one request and return the full concatenated text."""
        return "".join(self.generate_stream(audio_ref, prompt, max_new_tokens))

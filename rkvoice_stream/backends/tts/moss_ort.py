"""MOSS-TTS-Nano ONNX Runtime streaming backend.

This is the RK3576 production fallback while RKNN subgraphs are still being
split and stabilized. It preloads ONNX Runtime sessions and streams 80 ms
stereo chunks through:

    prefill -> fixed-frame sampler -> codec decode

If ``sentencepiece`` is unavailable, arbitrary text is rejected unless
``MOSS_ORT_ALLOW_DETERMINISTIC_FALLBACK=1`` is set for device smoke tests.
"""

from __future__ import annotations

import io
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import soundfile as sf

from rkvoice_stream.engine.tts import TTSBackend

logger = logging.getLogger(__name__)


class MossORTArtifactError(RuntimeError):
    """Raised when a MOSS ORT artifact bundle is not production-valid."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_moss_ort_artifacts(require_streaming_codec: bool = True) -> list[str]:
    artifacts = [
        "tokenizer.model",
        "tts_browser_onnx_meta.json",
        "codec_browser_onnx_meta.json",
        "moss_tts_prefill.onnx",
        "moss_tts_decode_step.onnx",
        "moss_tts_local_fixed_sampled_frame.onnx",
        "moss_tts_global_shared.data",
        "moss_tts_local_shared.data",
        "moss_audio_tokenizer_decode_full.onnx",
        "moss_audio_tokenizer_decode_shared.data",
    ]
    if require_streaming_codec:
        artifacts.append("moss_audio_tokenizer_decode_step.onnx")
    return artifacts


def default_moss_hybrid_artifacts(seq_len: int = 320, target: str = "rk3576") -> list[str]:
    artifacts = [
        f"moss_embedding_prefix.s{seq_len}.onnx",
        f"moss_final_norm.s{seq_len}.onnx",
    ]
    for layer in range(12):
        artifacts.extend(
            [
                f"moss_block{layer}_attn_residual.s{seq_len}.onnx",
                f"moss_block{layer}_ln2_mlp.s{seq_len}.fp16.{target}.rknn",
            ]
        )
    return artifacts


def default_moss_hybrid_mlp_only_artifacts(seq_len: int = 320, target: str = "rk3576") -> list[str]:
    artifacts = [
        f"moss_embedding_prefix.s{seq_len}.onnx",
        f"moss_final_norm.s{seq_len}.onnx",
    ]
    for layer in range(12):
        artifacts.extend(
            [
                f"moss_block{layer}_attn_residual.s{seq_len}.onnx",
                f"moss_block{layer}_ln2_mlp.s{seq_len}.onnx",
                f"moss_block{layer}_ln2.s{seq_len}.onnx",
                f"moss_block{layer}_mlp.s{seq_len}.fp16.{target}.rknn",
            ]
        )
    return artifacts


def default_moss_hybrid_fc_split_artifacts(seq_len: int = 320, target: str = "rk3576", split: str = "fc_out_only") -> list[str]:
    artifacts = [
        f"moss_embedding_prefix.s{seq_len}.onnx",
        f"moss_final_norm.s{seq_len}.onnx",
    ]
    for layer in range(12):
        artifacts.extend(
            [
                f"moss_block{layer}_attn_residual.s{seq_len}.onnx",
                f"moss_block{layer}_ln2_mlp.s{seq_len}.onnx",
                f"moss_block{layer}_ln2.s{seq_len}.onnx",
            ]
        )
        if split == "fc_in_act_only":
            artifacts.extend(
                [
                    f"moss_block{layer}_fc_in_act.s{seq_len}.fp16.{target}.rknn",
                    f"moss_block{layer}_fc_out.s{seq_len}.onnx",
                ]
            )
        elif split == "fc_out_only":
            artifacts.extend(
                [
                    f"moss_block{layer}_fc_in_act.s{seq_len}.onnx",
                    f"moss_block{layer}_fc_out.s{seq_len}.fp16.{target}.rknn",
                ]
            )
        else:
            raise ValueError(f"unsupported MOSS fc split: {split}")
    return artifacts


def default_moss_hybrid_ln1_cattn_artifacts(seq_len: int = 320, target: str = "rk3576") -> list[str]:
    return _required_hybrid_artifacts_for_split(seq_len, target, "ln1_cattn")


def _parse_hybrid_layers(raw: str | None) -> set[int]:
    text = str(raw or "all").strip().lower()
    if text in {"", "all"}:
        return set(range(12))
    if text == "none":
        return set()
    layers: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            layers.update(range(int(start_s), int(end_s) + 1))
        else:
            layers.add(int(part))
    invalid = sorted(layer for layer in layers if layer < 0 or layer > 11)
    if invalid:
        raise ValueError(f"invalid MOSS hybrid layer ids: {invalid}")
    return layers


def _validate_artifact_entries(root: Path, artifacts: Any, context: str) -> set[str]:
    if not isinstance(artifacts, list) or not artifacts:
        raise MossORTArtifactError(f"{context} artifacts must be a non-empty list")

    paths_seen: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            raise MossORTArtifactError(f"{context} artifact entry must be object, got {item!r}")
        required = bool(item.get("required", True))
        rel = item.get("path")
        if not rel:
            raise MossORTArtifactError(f"{context} artifact missing path: {item!r}")
        rel_path = Path(str(rel))
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise MossORTArtifactError(f"{context} artifact path must be relative and stay inside model_dir: {rel!r}")
        paths_seen.add(str(rel_path))
        path = root / rel_path
        if not path.exists():
            if required:
                raise MossORTArtifactError(f"Missing required {context} artifact: {path}")
            continue
        if path.is_dir():
            raise MossORTArtifactError(f"{context} artifact path is a directory: {path}")
        expected_size = item.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise MossORTArtifactError(
                f"Size mismatch for {path}: got {path.stat().st_size}, expected {expected_size}"
            )
        expected_sha = item.get("sha256")
        if expected_sha:
            actual_sha = _sha256_file(path)
            if actual_sha.lower() != str(expected_sha).lower():
                raise MossORTArtifactError(f"sha256 mismatch for {path}: got {actual_sha}, expected {expected_sha}")
    return paths_seen


def validate_moss_ort_artifacts(
    model_dir: str | Path,
    manifest_name: str = "moss-ort-manifest.json",
) -> dict[str, Any]:
    """Validate a MOSS ONNX Runtime bundle and return the parsed manifest."""

    root = Path(model_dir)
    manifest_path = root / manifest_name
    if not manifest_path.exists():
        raise MossORTArtifactError(f"Missing MOSS ORT manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MossORTArtifactError(f"Invalid JSON manifest: {manifest_path}: {exc}") from exc

    if manifest.get("model_id") not in {"moss-tts-nano-onnx", "moss-tts-nano-ort"}:
        raise MossORTArtifactError(f"Unexpected MOSS ORT model_id: {manifest.get('model_id')!r}")
    target = str(manifest.get("target_platform", "")).lower()
    if target and target not in {"rk3576", "rk3588", "generic"}:
        raise MossORTArtifactError(f"target_platform must be rk3576, rk3588, or generic, got {target!r}")
    if int(manifest.get("sample_rate", 0)) <= 0:
        raise MossORTArtifactError("manifest sample_rate must be positive")
    if int(manifest.get("channels", 0)) <= 0:
        raise MossORTArtifactError("manifest channels must be positive")

    paths_seen = _validate_artifact_entries(root, manifest.get("artifacts"), "MOSS ORT")

    if bool(manifest.get("streaming_required", True)):
        for rel in default_moss_ort_artifacts(require_streaming_codec=True):
            if rel not in paths_seen:
                raise MossORTArtifactError(f"streaming manifest missing required artifact entry: {rel}")

    gates = manifest.get("production_gates", {})
    if gates:
        for key in (
            "max_tts_first_payload_ms",
            "max_dialogue_first_payload_ms",
            "max_tts_wall_ms",
            "max_dialogue_wall_ms",
        ):
            value = gates.get(key)
            if value is not None and float(value) <= 0:
                raise MossORTArtifactError(f"production_gates.{key} must be positive")
        for key in ("max_avg_cer", "max_cer"):
            value = gates.get(key)
            if value is not None and not (0 <= float(value) <= 1):
                raise MossORTArtifactError(f"production_gates.{key} must be in [0, 1]")
        rms = gates.get("min_rms")
        if rms is not None and float(rms) < 0:
            raise MossORTArtifactError("production_gates.min_rms must be non-negative")

    return manifest


def _normalize_hybrid_split(value: Any) -> str:
    raw = str(value or "prefill_ln2_mlp").strip()
    aliases = {
        "prefill_ln2_mlp": "ln2_mlp",
        "ln2_mlp": "ln2_mlp",
        "prefill_mlp_only": "mlp_only",
        "mlp_only": "mlp_only",
        "prefill_fc_in_act_only": "fc_in_act_only",
        "fc_in_act_only": "fc_in_act_only",
        "prefill_fc_out_only": "fc_out_only",
        "fc_out_only": "fc_out_only",
        "prefill_ln1_cattn": "ln1_cattn",
        "ln1_cattn": "ln1_cattn",
    }
    if raw not in aliases:
        raise MossORTArtifactError(f"Unexpected hybrid split: {raw!r}")
    return aliases[raw]


def _required_hybrid_artifacts_for_split(
    seq_len: int,
    target: str,
    split: str,
    layers: set[int] | None = None,
    split_root: str = "artifact_dir",
) -> list[str]:
    return [path for _, path in _required_hybrid_artifact_entries_for_split(seq_len, target, split, layers, split_root)]


def _required_hybrid_artifact_entries_for_split(
    seq_len: int,
    target: str,
    split: str,
    layers: set[int] | None = None,
    split_root: str = "artifact_dir",
) -> list[tuple[str, str]]:
    selected_layers = set(range(12)) if layers is None else set(layers)
    artifacts: list[tuple[str, str]] = [
        ("artifact_dir", f"moss_embedding_prefix.s{seq_len}.onnx"),
        ("artifact_dir", f"moss_final_norm.s{seq_len}.onnx"),
    ]
    for layer in range(12):
        if split == "ln1_cattn" and layer in selected_layers:
            artifacts.append((split_root, f"moss_block{layer}_attn_after_cattn.s{seq_len}.onnx"))
        else:
            artifacts.append(("artifact_dir", f"moss_block{layer}_attn_residual.s{seq_len}.onnx"))
        if layer not in selected_layers:
            artifacts.append(("artifact_dir", f"moss_block{layer}_ln2_mlp.s{seq_len}.onnx"))
            continue
        if split == "ln2_mlp":
            artifacts.append((split_root, f"moss_block{layer}_ln2_mlp.s{seq_len}.fp16.{target}.rknn"))
        elif split == "ln1_cattn":
            artifacts.extend(
                [
                    (split_root, f"moss_block{layer}_ln1_cattn.s{seq_len}.fp16.{target}.rknn"),
                    (split_root, f"moss_block{layer}_ln2_mlp.s{seq_len}.fp16.{target}.rknn"),
                ]
            )
        elif split == "mlp_only":
            artifacts.extend(
                [
                    (split_root, f"moss_block{layer}_ln2.s{seq_len}.onnx"),
                    (split_root, f"moss_block{layer}_mlp.s{seq_len}.fp16.{target}.rknn"),
                ]
            )
        elif split == "fc_in_act_only":
            artifacts.extend(
                [
                    (split_root, f"moss_block{layer}_ln2.s{seq_len}.onnx"),
                    (split_root, f"moss_block{layer}_fc_in_act.s{seq_len}.fp16.{target}.rknn"),
                    (split_root, f"moss_block{layer}_fc_out.s{seq_len}.onnx"),
                ]
            )
        elif split == "fc_out_only":
            artifacts.extend(
                [
                    (split_root, f"moss_block{layer}_ln2.s{seq_len}.onnx"),
                    (split_root, f"moss_block{layer}_fc_in_act.s{seq_len}.onnx"),
                    (split_root, f"moss_block{layer}_fc_out.s{seq_len}.fp16.{target}.rknn"),
                ]
            )
        else:
            raise MossORTArtifactError(f"Unexpected hybrid split: {split!r}")
    return artifacts


def _validate_artifact_entries_for_roots(
    roots: dict[str, Path],
    artifacts: Any,
    context: str,
) -> set[tuple[str, str]]:
    if not isinstance(artifacts, list) or not artifacts:
        raise MossORTArtifactError(f"{context} artifacts must be a non-empty list")

    paths_seen: set[tuple[str, str]] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            raise MossORTArtifactError(f"{context} artifact entry must be object, got {item!r}")
        required = bool(item.get("required", True))
        rel = item.get("path")
        if not rel:
            raise MossORTArtifactError(f"{context} artifact missing path: {item!r}")
        root_key = str(item.get("root", "artifact_dir"))
        if root_key not in roots:
            raise MossORTArtifactError(f"{context} artifact has unknown root {root_key!r}: {item!r}")
        rel_path = Path(str(rel))
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise MossORTArtifactError(f"{context} artifact path must be relative and stay inside model_dir: {rel!r}")
        paths_seen.add((root_key, str(rel_path)))
        path = roots[root_key] / rel_path
        if not path.exists():
            if required:
                raise MossORTArtifactError(f"Missing required {context} artifact: {path}")
            continue
        if path.is_dir():
            raise MossORTArtifactError(f"{context} artifact path is a directory: {path}")
        expected_size = item.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise MossORTArtifactError(
                f"Size mismatch for {path}: got {path.stat().st_size}, expected {expected_size}"
            )
        expected_sha = item.get("sha256")
        if expected_sha:
            actual_sha = _sha256_file(path)
            if actual_sha.lower() != str(expected_sha).lower():
                raise MossORTArtifactError(f"sha256 mismatch for {path}: got {actual_sha}, expected {expected_sha}")
    return paths_seen


def validate_moss_hybrid_artifacts(
    artifact_dir: str | Path,
    manifest_name: str = "moss-hybrid-manifest.json",
    seq_len: int | None = None,
    target: str = "rk3576",
    split: str | None = None,
    layers: set[int] | None = None,
    rknn_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a MOSS hybrid ORT+RKNN prefill bundle."""

    root = Path(artifact_dir)
    manifest_path = root / manifest_name
    if not manifest_path.exists():
        raise MossORTArtifactError(f"Missing MOSS hybrid manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MossORTArtifactError(f"Invalid JSON manifest: {manifest_path}: {exc}") from exc

    if manifest.get("model_id") != "moss-tts-nano-hybrid-rknn":
        raise MossORTArtifactError(f"Unexpected MOSS hybrid model_id: {manifest.get('model_id')!r}")
    manifest_target = str(manifest.get("target_platform", "")).lower()
    if manifest_target not in {"rk3576", "rk3588"}:
        raise MossORTArtifactError(f"target_platform must be rk3576 or rk3588, got {manifest_target!r}")
    if target and manifest_target != target:
        raise MossORTArtifactError(f"target_platform mismatch: got {manifest_target!r}, expected {target!r}")
    manifest_seq_len = int(manifest.get("seq_len", 0))
    if manifest_seq_len <= 0:
        raise MossORTArtifactError("manifest seq_len must be positive")
    if seq_len is not None and manifest_seq_len != int(seq_len):
        raise MossORTArtifactError(f"seq_len mismatch: got {manifest_seq_len}, expected {seq_len}")
    manifest_split = _normalize_hybrid_split(manifest.get("split"))
    if split is not None and manifest_split != _normalize_hybrid_split(split):
        raise MossORTArtifactError(f"hybrid split mismatch: got {manifest_split!r}, expected {split!r}")
    manifest_layers_raw = manifest.get("rknn_layers")
    if manifest_layers_raw is None:
        manifest_layers = set(range(12))
    elif isinstance(manifest_layers_raw, list):
        manifest_layers = {int(item) for item in manifest_layers_raw}
    else:
        raise MossORTArtifactError("hybrid manifest rknn_layers must be a list when present")
    invalid_layers = sorted(layer for layer in manifest_layers if layer < 0 or layer > 11)
    if invalid_layers:
        raise MossORTArtifactError(f"hybrid manifest rknn_layers out of range: {invalid_layers}")
    if layers is not None and manifest_layers != set(layers):
        raise MossORTArtifactError(
            f"hybrid rknn_layers mismatch: got {sorted(manifest_layers)}, expected {sorted(layers)}"
        )

    split_root = "rknn_dir" if rknn_dir is not None else "artifact_dir"
    paths_seen = _validate_artifact_entries_for_roots(
        {
            "artifact_dir": root,
            "rknn_dir": Path(rknn_dir) if rknn_dir is not None else root,
        },
        manifest.get("artifacts"),
        "MOSS hybrid",
    )
    for root_key, rel in _required_hybrid_artifact_entries_for_split(
        manifest_seq_len,
        manifest_target,
        manifest_split,
        manifest_layers,
        split_root,
    ):
        if (root_key, rel) not in paths_seen:
            label = f"{root_key}:{rel}" if root_key != "artifact_dir" else rel
            raise MossORTArtifactError(f"hybrid manifest missing required artifact entry: {label}")

    gates = manifest.get("quality_status", {})
    if gates.get("production_default") is True:
        raise MossORTArtifactError("hybrid manifest must not declare production_default=true until ASR gates pass")

    return manifest


def _truthy(value: str | None) -> bool:
    return str(value or "").strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in {None, ""} else int(default)


def _attention_input_name(layer: int) -> str:
    if layer == 0:
        return "/Add_15_output_0"
    return f"/Mul_{22 + (layer - 1) * 6}_output_0"


def _layer_suffix(layer: int) -> str:
    return "" if layer == 0 else f"_{layer}"


def _trim_kv_to_length(value: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim >= 3 and arr.shape[2] >= length:
        return arr[:, :, :length, ...].astype(np.float32, copy=False)
    if arr.ndim >= 2 and arr.shape[1] >= length:
        return arr[:, :length, ...].astype(np.float32, copy=False)
    return arr


class _HybridRknnSession:
    def __init__(self, path: Path) -> None:
        from rknnlite.api import RKNNLite

        self.path = path
        self._rknn = RKNNLite(verbose=False)
        ret = self._rknn.load_rknn(str(path))
        if ret != 0:
            raise RuntimeError(f"load_rknn returned {ret}: {path}")
        ret = self._rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime returned {ret}: {path}")

    def run(self, hidden: np.ndarray) -> np.ndarray:
        outputs = self._rknn.inference(inputs=[hidden.astype(np.float32, copy=False)])
        if outputs is None:
            raise RuntimeError(f"RKNN inference returned None: {self.path}")
        return np.asarray(outputs[0], dtype=np.float32)

    def release(self) -> None:
        self._rknn.release()


class _HybridPrefillSession:
    """Composed MOSS prefill from ORT CPU slices and stable RKNN islands."""

    def __init__(
        self,
        artifact_dir: Path,
        seq_len: int,
        threads: int,
        ort_module: Any,
        manifest_name: str | None = None,
        split: str = "ln2_mlp",
        layers: set[int] | None = None,
        rknn_dir: Path | None = None,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.rknn_dir = rknn_dir or artifact_dir
        self.seq_len = seq_len
        self._threads = threads
        self._ort = ort_module
        self._manifest_name = manifest_name or "moss-hybrid-manifest.json"
        self._split = split
        self._rknn_layers = set(range(12)) if layers is None else set(layers)
        self._embedding = None
        self._final_norm = None
        self._attention: dict[int, Any] = {}
        self._attention_suffix: dict[int, Any] = {}
        self._ln1_cattn_rknn: dict[int, _HybridRknnSession] = {}
        self._mlp_rknn: dict[int, _HybridRknnSession] = {}
        self._mlp_ort: dict[int, Any] = {}
        self._ln2: dict[int, Any] = {}
        self._fc_in_act_ort: dict[int, Any] = {}
        self._fc_out_ort: dict[int, Any] = {}
        if self._split not in {"ln2_mlp", "ln1_cattn", "mlp_only", "fc_in_act_only", "fc_out_only"}:
            raise ValueError(f"Unsupported MOSS hybrid split: {self._split}")
        self._validate_artifacts()

    def _validate_artifacts(self) -> None:
        manifest_path = self.artifact_dir / self._manifest_name
        if "MOSS_ORT_HYBRID_MANIFEST" in os.environ or manifest_path.exists():
            validate_moss_hybrid_artifacts(
                self.artifact_dir,
                self._manifest_name,
                seq_len=self.seq_len,
                target="rk3576",
                split=self._split,
                layers=self._rknn_layers,
                rknn_dir=self.rknn_dir if self.rknn_dir != self.artifact_dir else None,
            )
            return
        if self._split in {"mlp_only", "fc_in_act_only", "fc_out_only"}:
            required = [
                f"moss_embedding_prefix.s{self.seq_len}.onnx",
                f"moss_final_norm.s{self.seq_len}.onnx",
                *[f"moss_block{layer}_attn_residual.s{self.seq_len}.onnx" for layer in range(12)],
                *[
                    f"moss_block{layer}_ln2_mlp.s{self.seq_len}.onnx"
                    for layer in range(12)
                    if layer not in self._rknn_layers
                ],
            ]
            missing = [name for name in required if not (self.artifact_dir / name).exists()]
            missing.extend(
                str(Path(self.rknn_dir) / f"moss_block{layer}_ln2.s{self.seq_len}.onnx")
                for layer in self._rknn_layers
                if not (Path(self.rknn_dir) / f"moss_block{layer}_ln2.s{self.seq_len}.onnx").exists()
            )
            if self._split == "mlp_only":
                missing.extend(
                    str(Path(self.rknn_dir) / f"moss_block{layer}_mlp.s{self.seq_len}.fp16.rk3576.rknn")
                    for layer in self._rknn_layers
                    if not (Path(self.rknn_dir) / f"moss_block{layer}_mlp.s{self.seq_len}.fp16.rk3576.rknn").exists()
                )
            elif self._split == "fc_in_act_only":
                missing.extend(
                    str(Path(self.rknn_dir) / f"moss_block{layer}_fc_in_act.s{self.seq_len}.fp16.rk3576.rknn")
                    for layer in self._rknn_layers
                    if not (Path(self.rknn_dir) / f"moss_block{layer}_fc_in_act.s{self.seq_len}.fp16.rk3576.rknn").exists()
                )
                missing.extend(
                    str(Path(self.rknn_dir) / f"moss_block{layer}_fc_out.s{self.seq_len}.onnx")
                    for layer in self._rknn_layers
                    if not (Path(self.rknn_dir) / f"moss_block{layer}_fc_out.s{self.seq_len}.onnx").exists()
                )
            else:
                missing.extend(
                    str(Path(self.rknn_dir) / f"moss_block{layer}_fc_in_act.s{self.seq_len}.onnx")
                    for layer in self._rknn_layers
                    if not (Path(self.rknn_dir) / f"moss_block{layer}_fc_in_act.s{self.seq_len}.onnx").exists()
                )
                missing.extend(
                    str(Path(self.rknn_dir) / f"moss_block{layer}_fc_out.s{self.seq_len}.fp16.rk3576.rknn")
                    for layer in self._rknn_layers
                    if not (Path(self.rknn_dir) / f"moss_block{layer}_fc_out.s{self.seq_len}.fp16.rk3576.rknn").exists()
                )
        elif self._split == "ln1_cattn":
            required = [
                f"moss_embedding_prefix.s{self.seq_len}.onnx",
                f"moss_final_norm.s{self.seq_len}.onnx",
                *[
                    f"moss_block{layer}_attn_residual.s{self.seq_len}.onnx"
                    for layer in range(12)
                    if layer not in self._rknn_layers
                ],
                *[
                    f"moss_block{layer}_ln2_mlp.s{self.seq_len}.onnx"
                    for layer in range(12)
                    if layer not in self._rknn_layers
                ],
            ]
            missing = [name for name in required if not (self.artifact_dir / name).exists()]
            missing.extend(
                str(Path(self.rknn_dir) / f"moss_block{layer}_attn_after_cattn.s{self.seq_len}.onnx")
                for layer in self._rknn_layers
                if not (Path(self.rknn_dir) / f"moss_block{layer}_attn_after_cattn.s{self.seq_len}.onnx").exists()
            )
            missing.extend(
                str(Path(self.rknn_dir) / f"moss_block{layer}_ln1_cattn.s{self.seq_len}.fp16.rk3576.rknn")
                for layer in self._rknn_layers
                if not (Path(self.rknn_dir) / f"moss_block{layer}_ln1_cattn.s{self.seq_len}.fp16.rk3576.rknn").exists()
            )
            missing.extend(
                str(Path(self.rknn_dir) / f"moss_block{layer}_ln2_mlp.s{self.seq_len}.fp16.rk3576.rknn")
                for layer in self._rknn_layers
                if not (Path(self.rknn_dir) / f"moss_block{layer}_ln2_mlp.s{self.seq_len}.fp16.rk3576.rknn").exists()
            )
        else:
            required = [
                f"moss_embedding_prefix.s{self.seq_len}.onnx",
                f"moss_final_norm.s{self.seq_len}.onnx",
                *[f"moss_block{layer}_attn_residual.s{self.seq_len}.onnx" for layer in range(12)],
                *[
                    f"moss_block{layer}_ln2_mlp.s{self.seq_len}.onnx"
                    for layer in range(12)
                    if layer not in self._rknn_layers
                ],
            ]
            missing = [name for name in required if not (self.artifact_dir / name).exists()]
            missing.extend(
                str(Path(self.rknn_dir) / f"moss_block{layer}_ln2_mlp.s{self.seq_len}.fp16.rk3576.rknn")
                for layer in self._rknn_layers
                if not (Path(self.rknn_dir) / f"moss_block{layer}_ln2_mlp.s{self.seq_len}.fp16.rk3576.rknn").exists()
            )
        if missing:
            raise FileNotFoundError(f"MOSS hybrid prefill artifacts missing: {missing}")

    def _make_ort_session(self, path: Path) -> Any:
        opts = self._ort.SessionOptions()
        opts.intra_op_num_threads = self._threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return self._ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])

    def preload(self) -> None:
        self._embedding = self._make_ort_session(self.artifact_dir / f"moss_embedding_prefix.s{self.seq_len}.onnx")
        self._final_norm = self._make_ort_session(self.artifact_dir / f"moss_final_norm.s{self.seq_len}.onnx")
        if self._split == "ln1_cattn":
            self._attention = {
                layer: self._make_ort_session(self.artifact_dir / f"moss_block{layer}_attn_residual.s{self.seq_len}.onnx")
                for layer in range(12)
                if layer not in self._rknn_layers
            }
            self._attention_suffix = {
                layer: self._make_ort_session(self.rknn_dir / f"moss_block{layer}_attn_after_cattn.s{self.seq_len}.onnx")
                for layer in self._rknn_layers
            }
            self._ln1_cattn_rknn = {
                layer: _HybridRknnSession(self.rknn_dir / f"moss_block{layer}_ln1_cattn.s{self.seq_len}.fp16.rk3576.rknn")
                for layer in self._rknn_layers
            }
        else:
            self._attention = {
                layer: self._make_ort_session(self.artifact_dir / f"moss_block{layer}_attn_residual.s{self.seq_len}.onnx")
                for layer in range(12)
            }
        self._mlp_ort = {
            layer: self._make_ort_session(self.artifact_dir / f"moss_block{layer}_ln2_mlp.s{self.seq_len}.onnx")
            for layer in range(12)
            if layer not in self._rknn_layers
        }
        if self._split in {"mlp_only", "fc_in_act_only", "fc_out_only"}:
            self._ln2 = {
                layer: self._make_ort_session(self.rknn_dir / f"moss_block{layer}_ln2.s{self.seq_len}.onnx")
                for layer in self._rknn_layers
            }
        if self._split == "mlp_only":
            self._mlp_rknn = {
                layer: _HybridRknnSession(self.rknn_dir / f"moss_block{layer}_mlp.s{self.seq_len}.fp16.rk3576.rknn")
                for layer in self._rknn_layers
            }
        elif self._split == "fc_in_act_only":
            self._fc_out_ort = {
                layer: self._make_ort_session(self.rknn_dir / f"moss_block{layer}_fc_out.s{self.seq_len}.onnx")
                for layer in self._rknn_layers
            }
            self._mlp_rknn = {
                layer: _HybridRknnSession(self.rknn_dir / f"moss_block{layer}_fc_in_act.s{self.seq_len}.fp16.rk3576.rknn")
                for layer in self._rknn_layers
            }
        elif self._split == "fc_out_only":
            self._fc_in_act_ort = {
                layer: self._make_ort_session(self.rknn_dir / f"moss_block{layer}_fc_in_act.s{self.seq_len}.onnx")
                for layer in self._rknn_layers
            }
            self._mlp_rknn = {
                layer: _HybridRknnSession(self.rknn_dir / f"moss_block{layer}_fc_out.s{self.seq_len}.fp16.rk3576.rknn")
                for layer in self._rknn_layers
            }
        else:
            self._mlp_rknn = {
                layer: _HybridRknnSession(self.rknn_dir / f"moss_block{layer}_ln2_mlp.s{self.seq_len}.fp16.rk3576.rknn")
                for layer in self._rknn_layers
            }

    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        if self._embedding is None or self._final_norm is None:
            raise RuntimeError("MOSS hybrid prefill is not preloaded")
        actual_len = int(input_ids.shape[1])
        if actual_len > self.seq_len:
            raise ValueError(f"Hybrid prefill seq_len {self.seq_len} cannot fit input length {actual_len}")
        padded_ids = np.full((1, self.seq_len, 17), 1024, dtype=np.int32)
        padded_ids[:, :actual_len, :] = input_ids.astype(np.int32, copy=False)
        padded_mask = np.zeros((1, self.seq_len), dtype=np.int32)
        padded_mask[:, :actual_len] = attention_mask.astype(np.int32, copy=False)
        mask3 = padded_mask[:, :, None].astype(np.float32)
        timings: dict[str, Any] = {"layers": []}

        t0 = time.perf_counter()
        hidden = self._embedding.run(None, {"input_ids": padded_ids})[0].astype(np.float32, copy=False)
        timings["embedding_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        kv_cache: dict[str, np.ndarray] = {}
        for layer in range(12):
            layer_start = time.perf_counter()
            attn_start = time.perf_counter()
            if self._split == "ln1_cattn" and layer in self._rknn_layers:
                qkv = self._ln1_cattn_rknn[layer].run(np.asarray(hidden * mask3, dtype=np.float32))
                suffix_inputs = {
                    f"/c_attn{_layer_suffix(layer)}/Add_output_0": qkv,
                    _attention_input_name(layer): hidden,
                    "attention_mask": padded_mask,
                }
                attn_residual, key, value = self._attention_suffix[layer].run(None, suffix_inputs)
                attention_kind = "rknn_ln1_cattn_suffix_ort"
            else:
                attn_residual, key, value = self._attention[layer].run(
                    None,
                    {_attention_input_name(layer): hidden, "attention_mask": padded_mask},
                )
                attention_kind = "ort_attn_residual"
            attn_ms = (time.perf_counter() - attn_start) * 1000.0
            mlp_start = time.perf_counter()
            if layer in self._rknn_layers:
                if self._split == "mlp_only":
                    ln2_input_name = self._ln2[layer].get_inputs()[0].name
                    normalized = self._ln2[layer].run(None, {ln2_input_name: np.asarray(attn_residual, dtype=np.float32)})[0]
                    mlp_out = self._mlp_rknn[layer].run(np.asarray(normalized, dtype=np.float32))
                    mlp_kind = "rknn_mlp_only"
                elif self._split == "fc_in_act_only":
                    ln2_input_name = self._ln2[layer].get_inputs()[0].name
                    normalized = self._ln2[layer].run(None, {ln2_input_name: np.asarray(attn_residual, dtype=np.float32)})[0]
                    activation = self._mlp_rknn[layer].run(np.asarray(normalized, dtype=np.float32))
                    fc_out_input_name = self._fc_out_ort[layer].get_inputs()[0].name
                    mlp_out = self._fc_out_ort[layer].run(None, {fc_out_input_name: np.asarray(activation, dtype=np.float32)})[0]
                    mlp_kind = "rknn_fc_in_act_only"
                elif self._split == "fc_out_only":
                    ln2_input_name = self._ln2[layer].get_inputs()[0].name
                    normalized = self._ln2[layer].run(None, {ln2_input_name: np.asarray(attn_residual, dtype=np.float32)})[0]
                    fc_in_input_name = self._fc_in_act_ort[layer].get_inputs()[0].name
                    activation = self._fc_in_act_ort[layer].run(None, {fc_in_input_name: np.asarray(normalized, dtype=np.float32)})[0]
                    mlp_out = self._mlp_rknn[layer].run(np.asarray(activation, dtype=np.float32))
                    mlp_kind = "rknn_fc_out_only"
                else:
                    mlp_out = self._mlp_rknn[layer].run(np.asarray(attn_residual, dtype=np.float32))
                    mlp_kind = "rknn_ln2_mlp"
            else:
                input_name = self._mlp_ort[layer].get_inputs()[0].name
                mlp_out = self._mlp_ort[layer].run(None, {input_name: np.asarray(attn_residual, dtype=np.float32)})[0]
                mlp_kind = "ort_ln2_mlp"
            mlp_ms = (time.perf_counter() - mlp_start) * 1000.0
            hidden = (np.asarray(attn_residual, dtype=np.float32) + mlp_out) * mask3
            kv_cache[f"present_key_{layer}"] = _trim_kv_to_length(key, actual_len)
            kv_cache[f"present_value_{layer}"] = _trim_kv_to_length(value, actual_len)
            timings["layers"].append(
                {
                    "layer": layer,
                    "attention_kind": attention_kind,
                    "attention_ms": round(attn_ms, 3),
                    "mlp_kind": mlp_kind,
                    "mlp_ms": round(mlp_ms, 3),
                    "layer_ms": round((time.perf_counter() - layer_start) * 1000.0, 3),
                }
            )

        final_start = time.perf_counter()
        ln_f = self._final_norm.run(None, {"/Mul_88_output_0": hidden})[0]
        global_hidden = np.asarray(ln_f, dtype=np.float32) * mask3
        timings["final_norm_ms"] = round((time.perf_counter() - final_start) * 1000.0, 3)
        timings["hybrid_prefill_ms"] = round(
            timings["embedding_ms"]
            + sum(float(item["layer_ms"]) for item in timings["layers"])
            + timings["final_norm_ms"],
            3,
        )
        return global_hidden, kv_cache, timings

    def release(self) -> None:
        for session in self._mlp_rknn.values():
            try:
                session.release()
            except Exception:
                logger.exception("Failed to release MOSS hybrid RKNN session")
        self._mlp_rknn = {}
        self._mlp_ort = {}
        self._attention_suffix = {}
        for session in self._ln1_cattn_rknn.values():
            try:
                session.release()
            except Exception:
                logger.exception("Failed to release MOSS ln1_cattn RKNN session")
        self._ln1_cattn_rknn = {}
        self._ln2 = {}
        self._fc_in_act_ort = {}
        self._fc_out_ort = {}
        self._attention = {}
        self._embedding = None
        self._final_norm = None


class _CodecStreamingDecodeSession:
    def __init__(self, codec_meta: dict[str, Any], session: Any) -> None:
        self._codec_meta = codec_meta
        self._session = session
        streaming = codec_meta.get("streaming_decode", {})
        self._transformer_specs = list(streaming.get("transformer_offsets", []))
        self._attention_specs = list(streaming.get("attention_caches", []))
        self._state_feeds: dict[str, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        self._state_feeds = {}
        for spec in self._transformer_specs:
            self._state_feeds[str(spec["input_name"])] = np.zeros(tuple(spec["shape"]), dtype=np.int32)
        for spec in self._attention_specs:
            self._state_feeds[str(spec["offset_input_name"])] = np.zeros(tuple(spec["offset_shape"]), dtype=np.int32)
            self._state_feeds[str(spec["cached_keys_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            self._state_feeds[str(spec["cached_values_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            self._state_feeds[str(spec["cached_positions_input_name"])] = np.full(tuple(spec["positions_shape"]), -1, dtype=np.int32)

    def run_frames(self, frames: list[list[int]]) -> tuple[np.ndarray, int]:
        num_quantizers = int(self._codec_meta["codec_config"]["num_quantizers"])
        audio_codes = np.zeros((1, len(frames), num_quantizers), dtype=np.int32)
        for frame_index, frame in enumerate(frames):
            for channel_index in range(num_quantizers):
                audio_codes[0, frame_index, channel_index] = int(frame[channel_index] if channel_index < len(frame) else 0)
        feeds: dict[str, np.ndarray] = {
            "audio_codes": audio_codes,
            "audio_code_lengths": np.asarray([len(frames)], dtype=np.int32),
            **self._state_feeds,
        }
        outputs = self._session.run(None, feeds)
        output_names = [output.name for output in self._session.get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        for spec in self._transformer_specs:
            self._state_feeds[str(spec["input_name"])] = named_outputs[str(spec["output_name"])]
        for spec in self._attention_specs:
            self._state_feeds[str(spec["offset_input_name"])] = named_outputs[str(spec["offset_output_name"])]
            self._state_feeds[str(spec["cached_keys_input_name"])] = named_outputs[str(spec["cached_keys_output_name"])]
            self._state_feeds[str(spec["cached_values_input_name"])] = named_outputs[str(spec["cached_values_output_name"])]
            self._state_feeds[str(spec["cached_positions_input_name"])] = named_outputs[str(spec["cached_positions_output_name"])]
        return named_outputs["audio"], int(named_outputs["audio_lengths"].reshape(-1)[0])


class MossORTBackend(TTSBackend):
    """Low-latency MOSS streaming backend using ONNX Runtime CPU EP."""

    supports_streaming = True

    def __init__(self) -> None:
        self._model_dir = Path(os.environ.get("MOSS_ORT_MODEL_DIR", os.environ.get("MODEL_DIR", "/opt/tts/models/moss-tts-nano-onnx")))
        self._sample_rate = int(os.environ.get("MOSS_ORT_SAMPLE_RATE", "48000"))
        self._channels = int(os.environ.get("MOSS_ORT_CHANNELS", "2"))
        self._threads = int(os.environ.get("MOSS_ORT_THREADS", "4"))
        self._prefill_threads = _env_int("MOSS_ORT_PREFILL_THREADS", self._threads)
        self._decode_threads = _env_int("MOSS_ORT_DECODE_THREADS", self._threads)
        self._sampler_threads = _env_int("MOSS_ORT_SAMPLER_THREADS", self._threads)
        self._codec_threads = _env_int("MOSS_ORT_CODEC_THREADS", self._threads)
        self._codec_batch_frames = max(1, int(os.environ.get("MOSS_ORT_CODEC_BATCH_FRAMES", "1")))
        self._codec_async = _truthy(os.environ.get("MOSS_ORT_CODEC_ASYNC"))
        self._load_full_codec = _truthy(os.environ.get("MOSS_ORT_LOAD_FULL_CODEC"))
        self._prefill_seq = int(os.environ.get("MOSS_ORT_PREFILL_SEQ", "0"))
        self._max_new_frames = int(os.environ.get("MOSS_ORT_MAX_NEW_FRAMES", "8"))
        self._voice = os.environ.get("MOSS_ORT_VOICE", "Junhao")
        self._warmup_text = os.environ.get("MOSS_ORT_WARMUP_TEXT", "")
        self._allow_probe_fallback = os.environ.get("MOSS_ORT_ALLOW_DETERMINISTIC_FALLBACK", "0") in {"1", "true", "TRUE", "yes"}
        self._hybrid_enabled = _truthy(os.environ.get("MOSS_ORT_HYBRID_RKNN"))
        self._hybrid_strict = _truthy(os.environ.get("MOSS_ORT_HYBRID_STRICT"))
        self._hybrid_dir = Path(os.environ.get("MOSS_ORT_HYBRID_DIR", os.environ.get("MOSS_ORT_HYBRID_MODEL_DIR", "")) or self._model_dir)
        self._hybrid_rknn_dir = Path(os.environ.get("MOSS_ORT_HYBRID_RKNN_DIR", "")) if os.environ.get("MOSS_ORT_HYBRID_RKNN_DIR") else self._hybrid_dir
        self._hybrid_seq_len = int(os.environ.get("MOSS_ORT_HYBRID_SEQ_LEN", "320"))
        self._hybrid_manifest_name = os.environ.get("MOSS_ORT_HYBRID_MANIFEST", "moss-hybrid-manifest.json")
        self._hybrid_split = os.environ.get("MOSS_ORT_HYBRID_SPLIT", "ln2_mlp")
        self._hybrid_layers = _parse_hybrid_layers(os.environ.get("MOSS_ORT_HYBRID_LAYERS", "all"))
        self._manifest_name = os.environ.get("MOSS_ORT_MANIFEST", "moss-ort-manifest.json")
        self._rng_seed = int(os.environ.get("MOSS_ORT_SEED", "1234"))
        self._rng = np.random.default_rng(self._rng_seed)
        self._rng_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stream_requests = 0
        self._stream_completed = 0
        self._stream_errors = 0
        self._stream_active = 0
        self._stream_chunks = 0
        self._last_stream_error: str | None = None
        self._last_stream_error_time: float | None = None
        self._prefill = None
        self._decode = None
        self._sampler = None
        self._codec = None
        self._codec_stream = None
        self._codec_stream_session: _CodecStreamingDecodeSession | None = None
        self._sp = None
        self._voice_prefix_cache: tuple[int, dict[str, np.ndarray]] | None = None
        self._manifest: dict[str, Any] = {}
        self._artifact_manifest: dict[str, Any] | None = None
        self._artifact_manifest_sha256: str | None = None
        self._tts_meta: dict[str, Any] = {}
        self._codec_meta: dict[str, Any] = {}
        self._config: dict[str, Any] = {}
        self._hybrid_prefill: _HybridPrefillSession | None = None
        self._ready = False

    @property
    def name(self) -> str:
        return "moss_ort"

    def is_ready(self) -> bool:
        return self._ready

    def runtime_info(self) -> dict[str, Any]:
        manifest = self._artifact_manifest or {}
        artifacts = manifest.get("artifacts") or []
        return {
            "backend": self.name,
            "ready": self._ready,
            "model_dir": str(self._model_dir),
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "supports_streaming": self.supports_streaming,
            "profile": {
                "voice": self._voice,
                "seed": self._rng_seed,
                "threads": self._threads,
                "session_threads": {
                    "prefill": self._prefill_threads,
                    "decode": self._decode_threads,
                    "sampler": self._sampler_threads,
                    "codec": self._codec_threads,
                },
                "prefill_seq": self._prefill_seq,
                "max_new_frames": self._max_new_frames,
                "codec_streaming": self._codec_stream_session is not None,
                "codec_full_loaded": self._codec is not None,
                "codec_batch_frames": self._codec_batch_frames,
                "codec_async": self._codec_async,
                "cache_voice_prefix": False,
            },
            "manifest": {
                "name": self._manifest_name,
                "validated": self._artifact_manifest is not None,
                "sha256": self._artifact_manifest_sha256,
                "model_id": manifest.get("model_id"),
                "target_platform": manifest.get("target_platform"),
                "required_artifacts": len([a for a in artifacts if a.get("required", True)]),
                "streaming_required": manifest.get("streaming_required"),
            },
            "hybrid": {
                "enabled": self._hybrid_enabled,
                "strict": self._hybrid_strict,
                "dir": str(self._hybrid_dir),
                "rknn_dir": str(self._hybrid_rknn_dir),
                "seq_len": self._hybrid_seq_len,
                "split": self._hybrid_split,
                "layers": sorted(self._hybrid_layers),
                "manifest": self._hybrid_manifest_name if self._hybrid_enabled else None,
            },
            "streaming_stats": self._streaming_stats(),
        }

    def _streaming_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "requests": self._stream_requests,
                "completed": self._stream_completed,
                "errors": self._stream_errors,
                "active": self._stream_active,
                "chunks": self._stream_chunks,
                "last_error": self._last_stream_error,
                "last_error_time": self._last_stream_error_time,
            }

    @staticmethod
    def _make_session_options(ort_module: Any, threads: int) -> Any:
        opts = ort_module.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    def preload(self) -> None:
        import onnxruntime as ort

        manifest_path = self._model_dir / self._manifest_name
        if "MOSS_ORT_MANIFEST" in os.environ or manifest_path.exists():
            manifest = validate_moss_ort_artifacts(self._model_dir, self._manifest_name)
            self._artifact_manifest = manifest
            self._artifact_manifest_sha256 = _sha256_file(manifest_path)
            self._sample_rate = int(manifest.get("sample_rate", self._sample_rate))
            self._channels = int(manifest.get("channels", self._channels))
        codec_streaming_requested = os.environ.get("MOSS_ORT_CODEC_STREAMING", "1") not in {"0", "false", "FALSE", "no"}
        required = default_moss_ort_artifacts(require_streaming_codec=codec_streaming_requested)
        missing = [name for name in required if not (self._model_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"MOSS ORT model dir missing required files: {missing}")

        self._tts_meta = json.loads((self._model_dir / "tts_browser_onnx_meta.json").read_text(encoding="utf-8"))
        self._codec_meta = json.loads((self._model_dir / "codec_browser_onnx_meta.json").read_text(encoding="utf-8"))
        self._config = dict(self._tts_meta.get("model_config") or {})
        manifest_path = self._model_dir / "browser_poc_manifest.json"
        if manifest_path.exists():
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._config.update(self._manifest.get("tts_config") or {})
            defaults = self._manifest.get("generation_defaults") or {}
            self._max_new_frames = int(os.environ.get("MOSS_ORT_MAX_NEW_FRAMES", defaults.get("max_new_frames", self._max_new_frames)))
            if "MOSS_ORT_VOICE" not in os.environ:
                voices = self._manifest.get("builtin_voices") or []
                if voices:
                    self._voice = str(max(voices, key=lambda v: len(v.get("prompt_audio_codes", []))).get("voice", self._voice))

        providers = ["CPUExecutionProvider"]

        start = time.perf_counter()
        self._prefill = ort.InferenceSession(
            str(self._model_dir / "moss_tts_prefill.onnx"),
            sess_options=self._make_session_options(ort, self._prefill_threads),
            providers=providers,
        )
        self._decode = ort.InferenceSession(
            str(self._model_dir / "moss_tts_decode_step.onnx"),
            sess_options=self._make_session_options(ort, self._decode_threads),
            providers=providers,
        )
        self._sampler = ort.InferenceSession(
            str(self._model_dir / "moss_tts_local_fixed_sampled_frame.onnx"),
            sess_options=self._make_session_options(ort, self._sampler_threads),
            providers=providers,
        )
        codec_step = self._model_dir / "moss_audio_tokenizer_decode_step.onnx"
        if codec_step.exists() and codec_streaming_requested:
            self._codec_stream = ort.InferenceSession(
                str(codec_step),
                sess_options=self._make_session_options(ort, self._codec_threads),
                providers=providers,
            )
            self._codec_stream_session = _CodecStreamingDecodeSession(self._codec_meta, self._codec_stream)
        if self._load_full_codec or self._codec_stream_session is None:
            self._codec = ort.InferenceSession(
                str(self._model_dir / "moss_audio_tokenizer_decode_full.onnx"),
                sess_options=self._make_session_options(ort, self._codec_threads),
                providers=providers,
            )
        if self._hybrid_enabled:
            hybrid_start = time.perf_counter()
            self._hybrid_prefill = _HybridPrefillSession(
                self._hybrid_dir,
                self._hybrid_seq_len,
                self._prefill_threads,
                ort,
                self._hybrid_manifest_name,
                split=self._hybrid_split,
                layers=self._hybrid_layers,
                rknn_dir=self._hybrid_rknn_dir,
            )
            self._hybrid_prefill.preload()
            logger.info(
                "MOSS hybrid RKNN prefill ready in %.0fms artifact_dir=%s seq_len=%s",
                (time.perf_counter() - hybrid_start) * 1000.0,
                self._hybrid_dir,
                self._hybrid_seq_len,
            )
        self._load_tokenizer()
        if os.environ.get("MOSS_ORT_CACHE_VOICE_PREFIX", "0") in {"1", "true", "TRUE", "yes"}:
            raise RuntimeError(
                "MOSS_ORT_CACHE_VOICE_PREFIX is disabled: prefix KV cache is not "
                "equivalent to full prefill for this ONNX export"
            )
        if self._warmup_text.strip():
            self._warmup_sessions(self._warmup_text)
        self._rng = np.random.default_rng(self._rng_seed)
        self._ready = True
        logger.info("MOSS ORT backend ready in %.0fms model_dir=%s", (time.perf_counter() - start) * 1000.0, self._model_dir)

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs: Any,
    ) -> tuple[bytes, dict]:
        start = time.perf_counter()
        chunks: list[np.ndarray] = []
        last_meta: dict[str, Any] = {}
        for chunk, meta in self.synthesize_stream(text, speaker_id=speaker_id, speed=speed, pitch_shift=pitch_shift, **kwargs):
            chunks.append(chunk)
            last_meta = dict(meta)
        audio = np.concatenate(chunks, axis=0).astype(np.float32, copy=False) if chunks else np.zeros((0, self._channels), dtype=np.float32)
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, self._sample_rate, format="WAV", subtype="PCM_16")
        elapsed = time.perf_counter() - start
        duration = audio.shape[0] / float(self._sample_rate)
        last_meta.setdefault("sample_rate", self._sample_rate)
        last_meta.setdefault("channels", self._channels)
        last_meta["duration"] = duration
        last_meta["inference_time"] = elapsed
        last_meta["wall_ms"] = int(round(elapsed * 1000.0))
        last_meta["rtf"] = elapsed / duration if duration > 0 else None
        return wav_io.getvalue(), last_meta

    def synthesize_stream(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs: Any,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        if not self.is_ready():
            raise RuntimeError("MOSS ORT backend not loaded; call preload() first")

        with self._stats_lock:
            self._stream_requests += 1
            self._stream_active += 1
        max_new_frames = int(kwargs.get("max_new_frames", self._max_new_frames))
        rng = np.random.default_rng(int(kwargs["seed"])) if "seed" in kwargs else self._rng
        lock = threading.Lock() if "seed" in kwargs else self._rng_lock
        completed = False
        try:
            with lock:
                for chunk, meta in self._synthesize_stream_locked(
                    text,
                    rng,
                    max_new_frames,
                ):
                    with self._stats_lock:
                        self._stream_chunks += 1
                    yield chunk, meta
            completed = True
        except Exception as exc:
            with self._stats_lock:
                self._stream_errors += 1
                self._last_stream_error = str(exc)
                self._last_stream_error_time = time.time()
            raise
        finally:
            with self._stats_lock:
                self._stream_active = max(0, self._stream_active - 1)
                if completed:
                    self._stream_completed += 1

    def _synthesize_stream_locked(
        self,
        text: str,
        rng: np.random.Generator,
        max_new_frames: int,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        start = time.perf_counter()
        if self._voice_prefix_cache is not None and self._prefill_seq <= 0 and self._sp is not None:
            past_len, kv_cache = self._clone_voice_prefix_cache()
            current_hidden = None
            for row in self._build_text_suffix_rows(text):
                current_hidden, kv_cache = self._decode_step(row.reshape(1, 1, 17), kv_cache, past_len)
                past_len += 1
            if current_hidden is None:
                raise RuntimeError("MOSS ORT text suffix produced no rows")
            prefill_ms = (time.perf_counter() - start) * 1000.0
            mode = "text_cached_prefix"
        else:
            input_ids, mode = self._build_prefill_rows(text)
            attention_mask = (input_ids[:, :, 0] != self._pad_token_id()).astype(np.int32)
            if not attention_mask.any():
                attention_mask[:] = 1
            prefill_names = self._tts_meta["onnx"]["prefill_output_names"]
            hybrid_meta: dict[str, Any] | None = None
            if self._hybrid_prefill is not None:
                try:
                    global_hidden, kv_cache, hybrid_timings = self._hybrid_prefill.run(input_ids, attention_mask)
                    prefill_outputs = None
                    hybrid_meta = {"hybrid": hybrid_timings}
                    mode = "text_hybrid_rknn" if mode == "text" else f"{mode}_hybrid_rknn"
                except Exception:
                    if self._hybrid_strict:
                        raise
                    logger.exception("MOSS hybrid prefill failed; falling back to full ORT prefill")
                    prefill_outputs = self._prefill.run(prefill_names, {"input_ids": input_ids, "attention_mask": attention_mask})
                    global_hidden = prefill_outputs[0]
                    kv_cache = {
                        prefill_names[i]: prefill_outputs[i]
                        for i in range(1, len(prefill_names))
                    }
                    mode = f"{mode}_hybrid_fallback_ort"
            else:
                prefill_outputs = self._prefill.run(prefill_names, {"input_ids": input_ids, "attention_mask": attention_mask})
                global_hidden = prefill_outputs[0]
                kv_cache = {
                    prefill_names[i]: prefill_outputs[i]
                    for i in range(1, len(prefill_names))
                }
            prefill_ms = (time.perf_counter() - start) * 1000.0
            past_len = int(attention_mask.sum())
            current_hidden = global_hidden[:, past_len - 1, :].astype(np.float32, copy=False)
        repetition_seen_mask = np.zeros((1, 16, 1024), dtype=np.int32)

        first_chunk_ms: float | None = None
        last_frame_tokens: np.ndarray | None = None
        generated_frames: list[np.ndarray] = []
        emitted_samples = 0
        if self._codec_stream_session is not None:
            self._codec_stream_session.reset()
        codec_executor: ThreadPoolExecutor | None = None
        codec_future: Future[tuple[np.ndarray, int, float]] | None = None
        codec_future_meta: dict[str, Any] | None = None
        pending_codec_frames: list[list[int]] = []
        pending_start_index = 0
        if self._codec_stream_session is not None and self._codec_async and self._codec_batch_frames > 1:
            codec_executor = ThreadPoolExecutor(max_workers=1)

        def _run_codec_frames(frames: list[list[int]]) -> tuple[np.ndarray, int, float]:
            codec_start = time.perf_counter()
            audio, audio_len = self._codec_stream_session.run_frames(frames)
            return audio, audio_len, (time.perf_counter() - codec_start) * 1000.0

        def _finish_codec_future(*, block: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
            nonlocal codec_future, codec_future_meta
            if codec_future is None or codec_future_meta is None:
                return None
            if not block and not codec_future.done():
                return None
            wait_start = time.perf_counter()
            audio, audio_len, codec_ms = codec_future.result()
            meta = dict(codec_future_meta)
            meta["codec_ms"] = round(codec_ms, 3)
            meta["codec_wait_ms"] = (time.perf_counter() - wait_start) * 1000.0
            codec_future = None
            codec_future_meta = None
            return audio, np.array([audio_len], dtype=np.int32), meta

        def _yield_finished_codec(*, block: bool) -> tuple[np.ndarray, dict[str, Any]] | None:
            finished = _finish_codec_future(block=block)
            if finished is None:
                return None
            audio, audio_lengths, meta = finished
            full_audio = self._normalize_audio(audio, audio_lengths)
            return full_audio, meta

        try:
          for frame_index in range(max_new_frames):
            ready_codec = _yield_finished_codec(block=False)
            if ready_codec is not None:
                chunk, meta = ready_codec
                if chunk.size:
                    yield chunk, meta
            decode_ms: float | None = None
            if frame_index > 0:
                assert last_frame_tokens is not None
                row = self._make_audio_row(last_frame_tokens).reshape(1, 1, 17)
                decode_start = time.perf_counter()
                current_hidden, kv_cache = self._decode_step(row, kv_cache, past_len)
                decode_ms = (time.perf_counter() - decode_start) * 1000.0
                past_len += 1

            sampler_start = time.perf_counter()
            should_continue, frame_token_ids = self._sampler.run(
                None,
                {
                    "global_hidden": current_hidden.astype(np.float32, copy=False),
                    "repetition_seen_mask": repetition_seen_mask,
                    "assistant_random_u": np.asarray(
                        [min(0.99999994, max(0.0, float(rng.random())))],
                        dtype=np.float32,
                    ),
                    "audio_random_u": np.asarray(
                        [[min(0.99999994, max(0.0, float(rng.random()))) for _ in range(16)]],
                        dtype=np.float32,
                    ),
                },
            )
            sampler_ms = (time.perf_counter() - sampler_start) * 1000.0
            frame = np.asarray(frame_token_ids, dtype=np.int32).reshape(1, 16)
            for i, token_id in enumerate(frame[0]):
                if 0 <= token_id < repetition_seen_mask.shape[2]:
                    repetition_seen_mask[0, i, token_id] = 1
            last_frame_tokens = frame[0]
            generated_frames.append(frame[0].astype(np.int32, copy=False))

            if self._codec_stream_session is not None:
                if not pending_codec_frames:
                    pending_start_index = frame_index
                pending_codec_frames.append(frame[0].astype(np.int32, copy=False).tolist())
                target_batch = 1 if frame_index == 0 else self._codec_batch_frames
                should_flush = (
                    len(pending_codec_frames) >= target_batch
                    or frame_index + 1 >= max_new_frames
                    or int(np.asarray(should_continue).reshape(-1)[0]) == 0
                )
                if not should_flush:
                    if int(np.asarray(should_continue).reshape(-1)[0]) == 0:
                        return
                    continue
                codec_batch_frames = len(pending_codec_frames)
                chunk_start_index = pending_start_index
                if codec_executor is not None and frame_index > 0:
                    ready_codec = _yield_finished_codec(block=True)
                    if ready_codec is not None:
                        chunk, meta = ready_codec
                        if chunk.size:
                            yield chunk, meta
                    codec_start = time.perf_counter()
                    codec_future = codec_executor.submit(_run_codec_frames, list(pending_codec_frames))
                    codec_future_meta = {
                        "backend": self.name,
                        "mode": mode,
                        "chunk_index": frame_index,
                        "chunk_start_index": chunk_start_index,
                        "codec_batch_frames": codec_batch_frames,
                        "ttfa_ms": int(round(first_chunk_ms or 0.0)),
                        "sample_rate": self._sample_rate,
                        "channels": self._channels,
                        "prefill_ms": None,
                        "decode_ms": round(decode_ms, 3) if decode_ms is not None else None,
                        "sampler_ms": round(sampler_ms, 3),
                        "codec_ms": None,
                        "codec_submit_ms": round((time.perf_counter() - codec_start) * 1000.0, 3),
                        "codec_async": True,
                    }
                    pending_codec_frames = []
                    if int(np.asarray(should_continue).reshape(-1)[0]) == 0:
                        ready_codec = _yield_finished_codec(block=True)
                        if ready_codec is not None:
                            chunk, meta = ready_codec
                            if chunk.size:
                                yield chunk, meta
                        return
                    continue
                codec_start = time.perf_counter()
                audio, audio_len = self._codec_stream_session.run_frames(pending_codec_frames)
                audio_lengths = np.array([audio_len], dtype=np.int32)
                pending_codec_frames = []
            else:
                codec_start = time.perf_counter()
                audio, audio_lengths = self._codec.run(
                    None,
                    {
                        "audio_codes": np.stack(generated_frames, axis=0).reshape(1, len(generated_frames), 16),
                        "audio_code_lengths": np.array([len(generated_frames)], dtype=np.int32),
                    },
                )
                codec_batch_frames = 1
                chunk_start_index = frame_index
            codec_ms = (time.perf_counter() - codec_start) * 1000.0
            full_audio = self._normalize_audio(audio, audio_lengths)
            if self._codec_stream_session is not None:
                chunk = full_audio
            else:
                chunk = full_audio[emitted_samples:]
                emitted_samples = int(full_audio.shape[0])
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - start) * 1000.0
            meta = {
                "backend": self.name,
                "mode": mode,
                "chunk_index": frame_index,
                "chunk_start_index": chunk_start_index,
                "codec_batch_frames": codec_batch_frames,
                "ttfa_ms": int(round(first_chunk_ms)),
                "sample_rate": self._sample_rate,
                "channels": self._channels,
                "prefill_ms": round(prefill_ms, 3) if frame_index == 0 else None,
                "decode_ms": round(decode_ms, 3) if decode_ms is not None else None,
                "sampler_ms": round(sampler_ms, 3),
                "codec_ms": round(codec_ms, 3),
            }
            if frame_index == 0 and "hybrid_meta" in locals() and hybrid_meta:
                meta.update(hybrid_meta)
            if chunk.size:
                yield chunk, meta
            if int(np.asarray(should_continue).reshape(-1)[0]) == 0:
                return
          while True:
            ready_codec = _yield_finished_codec(block=True)
            if ready_codec is None:
                break
            chunk, meta = ready_codec
            if chunk.size:
                yield chunk, meta
        finally:
            if codec_executor is not None:
                codec_executor.shutdown(wait=True)

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def cleanup(self) -> None:
        self._prefill = None
        self._decode = None
        self._sampler = None
        self._codec = None
        self._codec_stream = None
        self._codec_stream_session = None
        if self._hybrid_prefill is not None:
            self._hybrid_prefill.release()
        self._hybrid_prefill = None
        self._voice_prefix_cache = None
        self._ready = False

    def _load_tokenizer(self) -> None:
        try:
            import sentencepiece as spm
        except ImportError:
            self._sp = None
            if not self._allow_probe_fallback:
                logger.warning("sentencepiece unavailable; MOSS ORT will reject arbitrary text")
            return
        sp = spm.SentencePieceProcessor()
        if not sp.Load(str(self._model_dir / "tokenizer.model")):
            raise RuntimeError(f"Failed to load tokenizer: {self._model_dir / 'tokenizer.model'}")
        self._sp = sp

    def _warmup_sessions(self, text: str) -> None:
        if self._sp is None:
            logger.warning("Skipping MOSS ORT warmup because sentencepiece is unavailable")
            return
        input_ids, _mode = self._build_prefill_rows(text)
        attention_mask = (input_ids[:, :, 0] != self._pad_token_id()).astype(np.int32)
        if not attention_mask.any():
            attention_mask[:] = 1
        prefill_names = self._tts_meta["onnx"]["prefill_output_names"]
        if self._hybrid_prefill is not None and input_ids.shape[1] <= self._hybrid_seq_len:
            global_hidden, kv_cache, _hybrid_timings = self._hybrid_prefill.run(input_ids, attention_mask)
        else:
            prefill_outputs = self._prefill.run(prefill_names, {"input_ids": input_ids, "attention_mask": attention_mask})
            global_hidden = prefill_outputs[0]
            kv_cache = {prefill_names[i]: prefill_outputs[i] for i in range(1, len(prefill_names))}
        past_len = int(attention_mask.sum())
        current_hidden = global_hidden[:, past_len - 1, :].astype(np.float32, copy=False)

        should_continue, frame_token_ids = self._sampler.run(
            None,
            {
                "global_hidden": current_hidden,
                "repetition_seen_mask": np.zeros((1, 16, 1024), dtype=np.int32),
                "assistant_random_u": np.asarray([0.5], dtype=np.float32),
                "audio_random_u": np.full((1, 16), 0.5, dtype=np.float32),
            },
        )
        frame = np.asarray(frame_token_ids, dtype=np.int32).reshape(1, 16)
        row = self._make_audio_row(frame[0]).reshape(1, 1, 17)
        self._decode_step(row, kv_cache, past_len)

        if self._codec_stream_session is not None:
            self._codec_stream_session.reset()
            self._codec_stream_session.run_frames([frame[0].astype(np.int32, copy=False).tolist()])
            self._codec_stream_session.reset()
        else:
            self._codec.run(
                None,
                {
                    "audio_codes": frame.reshape(1, 1, 16).astype(np.int32, copy=False),
                    "audio_code_lengths": np.array([1], dtype=np.int32),
                },
            )
        if int(np.asarray(should_continue).reshape(-1)[0]) not in (0, 1):
            logger.warning("MOSS ORT warmup returned unexpected should_continue=%s", should_continue)

    def _prepare_voice_prefix_cache(self) -> None:
        rows = self._build_voice_prefix_rows()
        if not rows:
            self._voice_prefix_cache = None
            return
        input_ids = np.stack(rows, axis=0).reshape(1, len(rows), 17).astype(np.int32, copy=False)
        attention_mask = np.ones((1, len(rows)), dtype=np.int32)
        names = self._tts_meta["onnx"]["prefill_output_names"]
        outputs = self._prefill.run(names, {"input_ids": input_ids, "attention_mask": attention_mask})
        self._voice_prefix_cache = (
            int(attention_mask.sum()),
            {names[i]: outputs[i] for i in range(1, len(names))},
        )

    def _clone_voice_prefix_cache(self) -> tuple[int, dict[str, np.ndarray]]:
        if self._voice_prefix_cache is None:
            raise RuntimeError("MOSS ORT voice prefix cache is not initialized")
        past_len, cache = self._voice_prefix_cache
        return past_len, dict(cache)

    def _build_voice_prefix_rows(self) -> list[np.ndarray]:
        templates = self._manifest.get("prompt_templates") or {}
        rows: list[np.ndarray] = []
        rows.extend(self._make_text_row(int(t)) for t in templates.get("user_prompt_prefix_token_ids", []))
        rows.append(self._make_text_row(self._audio_start_token_id()))
        rows.extend(
            self._make_audio_row(np.asarray(frame, dtype=np.int32), slot_token_id=self._audio_user_slot_token_id())
            for frame in self._voice_prompt_rows()
        )
        rows.append(self._make_text_row(self._audio_end_token_id()))
        rows.extend(self._make_text_row(int(t)) for t in templates.get("user_prompt_after_reference_token_ids", []))
        return rows

    def _build_text_suffix_rows(self, text: str) -> list[np.ndarray]:
        if self._sp is None:
            raise RuntimeError("sentencepiece is required for MOSS ORT text synthesis")
        text_token_ids = [int(i) for i in self._sp.EncodeAsIds(text.strip() or " ")]
        templates = self._manifest.get("prompt_templates") or {}
        rows: list[np.ndarray] = []
        rows.extend(self._make_text_row(int(t)) for t in text_token_ids)
        rows.extend(self._make_text_row(int(t)) for t in templates.get("assistant_prompt_prefix_token_ids", []))
        rows.append(self._make_text_row(self._audio_start_token_id()))
        return rows

    def _build_prefill_rows(self, text: str) -> tuple[np.ndarray, str]:
        if self._sp is None:
            if not self._allow_probe_fallback:
                raise RuntimeError("sentencepiece is required for MOSS ORT text synthesis")
            probe_seq = self._prefill_seq if self._prefill_seq > 0 else 32
            ids = np.zeros((1, min(probe_seq, 32), 17), dtype=np.int32)
            ids[:, :, 0] = 1
            return ids, "deterministic_probe"

        rows = self._build_voice_prefix_rows()
        rows.extend(self._build_text_suffix_rows(text))
        if not rows:
            rows = [self._make_text_row(1)]
        if self._prefill_seq > 0 and len(rows) > self._prefill_seq:
            rows = rows[-self._prefill_seq :]
        return np.stack(rows, axis=0).reshape(1, len(rows), 17).astype(np.int32, copy=False), "text"

    def _voice_prompt_rows(self) -> list[np.ndarray]:
        for voice in self._manifest.get("builtin_voices", []):
            if str(voice.get("voice", "")).lower() == self._voice.lower():
                return [np.asarray(frame, dtype=np.int32) for frame in voice.get("prompt_audio_codes", [])]
        return []

    def _make_text_row(self, token_id: int) -> np.ndarray:
        row = np.full((17,), self._audio_pad_token_id(), dtype=np.int32)
        row[0] = int(token_id)
        return row

    def _make_audio_row(self, frame_tokens: np.ndarray, slot_token_id: int | None = None) -> np.ndarray:
        row = np.empty((17,), dtype=np.int32)
        row[0] = int(self._audio_assistant_slot_token_id() if slot_token_id is None else slot_token_id)
        row[1:] = np.asarray(frame_tokens, dtype=np.int32).reshape(16)
        return row

    def _decode_step(self, row: np.ndarray, kv_cache: dict[str, np.ndarray], past_len: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        inputs: dict[str, np.ndarray] = {
            "input_ids": row.astype(np.int32, copy=False),
            "past_valid_lengths": np.array([past_len], dtype=np.int32),
        }
        for layer in range(12):
            inputs[f"past_key_{layer}"] = kv_cache[f"present_key_{layer}"].astype(np.float16, copy=False)
            inputs[f"past_value_{layer}"] = kv_cache[f"present_value_{layer}"].astype(np.float16, copy=False)
        output_names = self._tts_meta["onnx"]["decode_output_names"]
        outputs = self._decode.run(output_names, inputs)
        next_hidden = outputs[0]
        next_cache = dict(kv_cache)
        for i, name in enumerate(output_names[1:], start=1):
            next_cache[name] = outputs[i]
        return next_hidden.reshape(1, -1).astype(np.float32, copy=False), next_cache

    def _normalize_audio(self, audio: np.ndarray, audio_lengths: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == self._channels:
            arr = np.transpose(arr[0], (1, 0))
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n = int(np.asarray(audio_lengths).reshape(-1)[0]) if np.asarray(audio_lengths).size else arr.shape[0]
        return arr[:n].astype(np.float32, copy=False)

    def _audio_pad_token_id(self) -> int:
        return int(self._config.get("audio_pad_token_id", 1024))

    def _pad_token_id(self) -> int:
        return int(self._config.get("pad_token_id", 3))

    def _audio_start_token_id(self) -> int:
        return int(self._config.get("audio_start_token_id", 6))

    def _audio_end_token_id(self) -> int:
        return int(self._config.get("audio_end_token_id", 7))

    def _audio_user_slot_token_id(self) -> int:
        return int(self._config.get("audio_user_slot_token_id", 8))

    def _audio_assistant_slot_token_id(self) -> int:
        return int(self._config.get("audio_assistant_slot_token_id", 9))

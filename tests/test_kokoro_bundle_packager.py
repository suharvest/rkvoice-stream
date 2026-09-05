import os
import stat
from pathlib import Path

import pytest

import tools.package_kokoro_convonly_bundle as packager


def stage(root: Path, platform="rk3588"):
    (root / "frontend").mkdir(parents=True)
    (root / "prefix").mkdir()
    (root / "tail").mkdir()
    for rel in ("frontend/frontend.manifest.json", "prefix/build-manifest.json",
                "tail/library.so", f"tail/merge-official-v9.{platform}.fp16.rknn"):
        (root / rel).write_bytes(b"fixture")


def test_packager_records_dynamic_cpu_generator(tmp_path):
    stage(tmp_path)
    cpu = tmp_path / "generator-dynamic-hardened.onnx"
    with cpu.open("wb") as stream:
        stream.truncate(packager.CPU_GENERATOR_SIZE)
    real_sha = packager.sha256
    packager.sha256 = lambda path: packager.GENERATOR_SOURCE_SHA256 if path == cpu else real_sha(path)
    try:
        manifest = packager.build_manifest(tmp_path, "rk3588")
    finally:
        packager.sha256 = real_sha
    assert manifest["cpu_generator"] == packager.CPU_GENERATOR
    assert any(row["path"] == manifest["cpu_generator"] for row in manifest["files"])


def test_packager_rejects_cpu_generator_wrong_size(tmp_path):
    stage(tmp_path)
    (tmp_path / packager.CPU_GENERATOR).write_bytes(b"wrong-size")
    with pytest.raises(ValueError, match="size mismatch"):
        packager.build_manifest(tmp_path, "rk3588")


def test_packager_rejects_cpu_generator_wrong_hash(tmp_path):
    stage(tmp_path)
    cpu = tmp_path / packager.CPU_GENERATOR
    with cpu.open("wb") as stream:
        stream.truncate(packager.CPU_GENERATOR_SIZE)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        packager.build_manifest(tmp_path, "rk3588")


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_packager_rejects_symlink_and_special_entries(tmp_path, kind):
    stage(tmp_path)
    target = tmp_path / "tail" / kind
    if kind == "symlink":
        target.symlink_to(tmp_path / "tail" / "library.so")
    else:
        os.mkfifo(target)
    with pytest.raises(ValueError, match="symlink|special"):
        packager.build_manifest(tmp_path, "rk3588")

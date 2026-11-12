"""Immutable run manifests and cryptographic artifact binding."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repository: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout
    return {"commit": commit, "clean": not bool(status.strip()), "status": status.splitlines()}


def artifact_hashes(root: Path, exclude=()) -> dict:
    excluded = {str(Path(path)) for path in exclude}
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and str(path.relative_to(root)) not in excluded:
            result[str(path.relative_to(root))] = sha256_file(path)
    return result


def dependency_versions():
    names = (
        "huggingface-hub", "matplotlib", "rasterio", "scikit-image",
        "scikit-learn", "scipy", "timm", "torchvision",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def start_run_manifest(output_dir, command, input_paths, require_clean_git=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    if any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {output_dir}")

    repository = Path(__file__).resolve().parents[1]
    state = git_state(repository)
    if require_clean_git and not state["clean"]:
        raise RuntimeError("Publication runs require a clean Git worktree")
    inputs = {}
    for name, value in input_paths.items():
        path = Path(value)
        if path.is_file():
            inputs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        elif path.is_dir():
            inputs[name] = {
                "path": str(path.resolve()),
                "files": artifact_hashes(path),
            }
        else:
            raise FileNotFoundError(f"Run input {name} does not exist: {path}")

    manifest = {
        "schema_version": "1.0.0",
        "status": "started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "git": state,
        "inputs": inputs,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "cudnn": torch.backends.cudnn.version(),
            "dependencies": dependency_versions(),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def _write_json_atomic(path: Path, payload: dict):
    """Write JSON via a temporary file and rename, so a crash mid-write
    cannot destroy an existing manifest (the run's only provenance record)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(path)


def complete_run_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "started":
        raise RuntimeError(
            f"Refusing to complete manifest with status "
            f"{manifest.get('status')!r}: {manifest_path}"
        )
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["outputs"] = artifact_hashes(
        manifest_path.parent, exclude=[manifest_path.name]
    )
    _write_json_atomic(manifest_path, manifest)

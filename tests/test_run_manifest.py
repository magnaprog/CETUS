import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from cetus_sar import reproducibility


def test_concurrent_starts_preserve_first_published_manifest(tmp_path, monkeypatch):
    entered = Event()
    resume = Event()
    original_hash = reproducibility.sha256_file
    slow_input = tmp_path / "slow"
    slow_input.write_bytes(b"first input")
    output = tmp_path / "run"

    def delayed_hash(path):
        if path == slow_input:
            entered.set()
            assert resume.wait(timeout=20)
        return original_hash(path)

    monkeypatch.setattr(reproducibility, "sha256_file", delayed_hash)
    start = reproducibility.start_run_manifest
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(start, output, ["slow"], {"slow": slow_input}, False)
        try:
            assert entered.wait(timeout=20)
            manifest = start(output, ["fast"], {}, require_clean_git=False)
            result = output / "result"
            result.write_bytes(b"fast result")
            reproducibility.complete_run_manifest(manifest)
        finally:
            resume.set()
        with pytest.raises(FileExistsError):
            first.result(timeout=20)

    payload = json.loads(manifest.read_text())
    assert payload["command"] == ["fast"]
    assert payload["status"] == "complete"
    assert payload["outputs"] == {"result": original_hash(result)}
    assert set(output.iterdir()) == {manifest, result}


def test_failed_manifest_completion_can_retry(tmp_path, monkeypatch):
    manifest = reproducibility.start_run_manifest(tmp_path, ["probe"], {}, False)
    previous = manifest.read_bytes()

    def fail_hash(*args, **kwargs):
        raise OSError("read failed")

    with monkeypatch.context() as patch:
        patch.setattr(reproducibility, "artifact_hashes", fail_hash)
        with pytest.raises(OSError, match="read failed"):
            reproducibility.complete_run_manifest(manifest)
    assert list(tmp_path.iterdir()) == [manifest]
    assert manifest.read_bytes() == previous
    reproducibility.complete_run_manifest(manifest)
    assert json.loads(manifest.read_text())["outputs"] == {}

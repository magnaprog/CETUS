"""Synthetic acceptance records and real fake workers; no scientific fitting."""
import copy
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import launch_fitting_sensitivity as launcher


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells():
    return [
        dict(id=f"{model}-fold-{fold}-{'standardized' if standard else 'raw'}-{penalty}",
             model=model, fold=fold, standardize=standard, regularization=penalty)
        for model in ("dinov2", "dofa", "croma", "random_init")
        for fold in range(5)
        for standard in (False, True)
        for penalty in ("none", "l2")
    ]


@pytest.fixture
def plan(tmp_path):
    return dict(
        cells=cells(), output_root=str(tmp_path / "outputs"),
        feature_root=str(tmp_path / "unused-features"), producer_commit="synthetic-producer",
        source_sha256={"synthetic-worker.py": "synthetic-source"},
        python_version="synthetic-python", package_versions={"synthetic": "1"},
        runtime_identity={"synthetic-runtime": "bound"},
    )


def worker_records(plan, cell, plan_hash, timestamp="2026-11-06T00:00:01+00:00"):
    output = Path(plan["output_root"]) / cell["id"]
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / "fitted_arrays.npz"
    # Acceptance checks opaque bytes/hashes. Numerical NPZ replay is separate.
    arrays.write_bytes(b"synthetic array payload; no scientific data\n")
    started = dict(
        cell=cell, plan_sha256=plan_hash, producer_commit=plan["producer_commit"],
        source_sha256=plan["source_sha256"],
        runtime={k: plan[k] for k in ("python_version", "package_versions", "runtime_identity")},
        started_utc=timestamp,
    )
    write(output / "started.json", started)
    write(output / "result.json", dict(
        started, status="finite_fit", arrays_sha256=digest(arrays),
        finished_utc="2026-11-06T00:00:02+00:00",
    ))
    return output


def completed_cell(plan, cell, plan_hash="synthetic-plan"):
    output = worker_records(plan, cell, plan_hash)
    launch = Path(plan["output_root"]) / "launches" / cell["id"]
    launch.mkdir(parents=True, exist_ok=True)
    log = launch / "process.log"
    log.write_bytes(b"synthetic worker completed\n")
    started = dict(cell=cell, plan_sha256=plan_hash,
                   started_utc="2026-11-06T00:00:00+00:00",
                   command=["synthetic-worker", "--cell", cell["id"]], cwd="synthetic-cwd")
    write(launch / "started.json", started)
    write(launch / "exit.json", dict(
        started, status="exited", returncode=0, pid=12345,
        finished_utc="2026-11-06T00:00:03+00:00", log_sha256=digest(log),
    ))
    return output, launch


def test_accept_cell_binds_result_exit_and_arrays(plan):
    cell = plan["cells"][0]
    output, launch = completed_cell(plan, cell)
    record = launcher.accept_cell(plan, cell, "synthetic-plan")
    assert record == dict(cell=cell["id"], result_sha256=digest(output / "result.json"),
                          exit_sha256=digest(launch / "exit.json"),
                          arrays_sha256=digest(output / "fitted_arrays.npz"))


@pytest.mark.parametrize("missing", ["result", "worker_started", "exit", "arrays"])
def test_accept_cell_rejects_missing_evidence(plan, missing):
    cell = plan["cells"][0]
    output, launch = completed_cell(plan, cell)
    {"result": output / "result.json", "worker_started": output / "started.json",
     "exit": launch / "exit.json", "arrays": output / "fitted_arrays.npz"}[missing].unlink()
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        launcher.accept_cell(plan, cell, "synthetic-plan")


@pytest.mark.parametrize("returncode,status", [
    (9, "exited"), (-signal.SIGTERM, "exited"), (0, "launcher_interrupted"),
    (None, "launch_failed"),
])
def test_accept_cell_rejects_unsuccessful_process(plan, returncode, status):
    cell = plan["cells"][0]
    _, launch = completed_cell(plan, cell)
    receipt = read(launch / "exit.json")
    receipt.update(returncode=returncode, status=status)
    write(launch / "exit.json", receipt)
    with pytest.raises(ValueError):
        launcher.accept_cell(plan, cell, "synthetic-plan")


@pytest.mark.parametrize("where,field", [
    ("result", "cell"), ("result", "plan_sha256"), ("result", "producer_commit"),
    ("result", "source_sha256"), ("result", "runtime"), ("result", "status"),
    ("worker_started", "cell"), ("worker_started", "plan_sha256"),
    ("worker_started", "producer_commit"), ("worker_started", "source_sha256"),
    ("worker_started", "runtime"), ("exit", "cell"), ("exit", "plan_sha256"),
])
def test_accept_cell_rejects_stale_identity(plan, where, field):
    cell = plan["cells"][0]
    output, launch = completed_cell(plan, cell)
    path = {"result": output / "result.json", "worker_started": output / "started.json",
            "exit": launch / "exit.json"}[where]
    record = read(path)
    record[field] = plan["cells"][1] if field == "cell" else "stale"
    write(path, record)
    with pytest.raises(ValueError):
        launcher.accept_cell(plan, cell, "synthetic-plan")


@pytest.mark.parametrize("fault", ["failed_receipt", "arrays_changed", "malformed_result", "nan_result"])
def test_accept_cell_rejects_failed_or_corrupt_result(plan, fault):
    cell = plan["cells"][0]
    output, _ = completed_cell(plan, cell)
    if fault == "failed_receipt":
        write(output / "failed.json", {"status": "failed"})
    elif fault == "arrays_changed":
        (output / "fitted_arrays.npz").write_bytes(b"changed")
    elif fault == "malformed_result":
        (output / "result.json").write_text("{incomplete")
    else:
        record = read(output / "result.json")
        record["objective"] = float("nan")
        write(output / "result.json", record)
    with pytest.raises(ValueError):
        launcher.accept_cell(plan, cell, "synthetic-plan")


@pytest.mark.parametrize("fault", ["missing_log", "changed_log", "missing_launch_started",
                                  "stale_launch_started"])
def test_accept_cell_rejects_missing_or_stale_process_evidence(plan, fault):
    cell = plan["cells"][0]
    _, launch = completed_cell(plan, cell)
    if fault == "missing_log":
        (launch / "process.log").unlink()
    elif fault == "changed_log":
        (launch / "process.log").write_bytes(b"changed after terminal receipt")
    elif fault == "missing_launch_started":
        (launch / "started.json").unlink()
    else:
        record = read(launch / "started.json")
        record["plan_sha256"] = "stale"
        write(launch / "started.json", record)
    with pytest.raises((ValueError, FileNotFoundError)):
        launcher.accept_cell(plan, cell, "synthetic-plan")


def test_accept_all_requires_exactly_80_cells_and_process_records(plan):
    for cell in plan["cells"]:
        completed_cell(plan, cell)
    accepted = launcher.accept_all(plan, "synthetic-plan")
    assert len(accepted) == 80
    assert {r["cell"] for r in accepted} == {c["id"] for c in plan["cells"]}


@pytest.mark.parametrize("fault", ["79", "81", "duplicate", "wrong_setting", "bool_fold"])
def test_accept_all_rejects_invalid_declared_coverage_before_io(plan, fault, monkeypatch):
    broken = copy.deepcopy(plan)
    if fault == "79":
        broken["cells"].pop()
    elif fault == "81":
        broken["cells"].append(dict(broken["cells"][0], id="unexpected"))
    elif fault == "duplicate":
        broken["cells"][-1] = broken["cells"][0]
    elif fault == "wrong_setting":
        broken["cells"][0]["regularization"] = "undeclared"
    else:
        broken["cells"][0]["fold"] = False
    monkeypatch.setattr(launcher, "accept_cell", lambda *a: pytest.fail("must reject before acceptance"))
    with pytest.raises(ValueError):
        launcher.accept_all(broken, "synthetic-plan")


@pytest.mark.parametrize("fault", ["missing_cell", "extra_cell", "missing_launch", "extra_launch"])
def test_accept_all_rejects_directory_coverage_mismatch(plan, fault, tmp_path):
    for cell in plan["cells"]:
        completed_cell(plan, cell)
    root = Path(plan["output_root"])
    if fault == "missing_cell":
        (root / plan["cells"][0]["id"]).rename(tmp_path / "removed-cell")
    elif fault == "extra_cell":
        (root / "unexpected").mkdir()
    elif fault == "missing_launch":
        (root / "launches" / plan["cells"][0]["id"]).rename(tmp_path / "removed-launch")
    else:
        (root / "launches" / "unexpected").mkdir()
    with pytest.raises(ValueError):
        launcher.accept_all(plan, "synthetic-plan")


FAKE_WORKER = r'''
import argparse, hashlib, json, os, signal, time
from datetime import datetime, timezone
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--plan"); p.add_argument("--plan-sha256"); p.add_argument("--cell")
a=p.parse_args(); plan=json.loads(Path(a.plan).read_text())
cell=next(c for c in plan["cells"] if c["id"]==a.cell)
mode=plan.get("fake_modes",{}).get(a.cell,"success")
launch=Path(plan["output_root"])/"launches"/a.cell
def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+"\n")
if mode in ("sleep","ignore_term"):
    if mode=="ignore_term": signal.signal(signal.SIGTERM,signal.SIG_IGN)
    write(launch/"ready.json",{"pid":os.getpid()})
    while True: time.sleep(1)
if mode=="nonzero":
    print("injected nonzero worker",flush=True)
    raise SystemExit(7)
if mode=="noop": raise SystemExit(0)
out=Path(plan["output_root"])/a.cell
out.mkdir(parents=True,exist_ok=False)
now=lambda:datetime.now(timezone.utc).isoformat()
started=dict(cell=cell,plan_sha256=a.plan_sha256,producer_commit=plan["producer_commit"],
    source_sha256=plan["source_sha256"],started_utc=now(),
    runtime={k:plan[k] for k in ("python_version","package_versions","runtime_identity")})
write(out/"started.json",started)
payload=out/"fitted_arrays.npz";payload.write_bytes(b"synthetic worker bytes")
if mode!="missing_result":
    write(out/"result.json",dict(started,status="finite_fit",finished_utc=now(),
        arrays_sha256=hashlib.sha256(payload.read_bytes()).hexdigest()))
print("synthetic worker finished",flush=True)
'''


@pytest.fixture
def fake_source(plan, tmp_path, monkeypatch):
    root = tmp_path / "fake-source"
    script = root / "scripts" / "run_fitting_sensitivity.py"
    script.parent.mkdir(parents=True)
    script.write_text(FAKE_WORKER)
    monkeypatch.setattr(launcher, "ROOT", root)
    plan_path = tmp_path / "plan.json"
    return root, plan_path


def seal(plan, path):
    write(path, plan)
    return digest(path)


def test_real_fake_workers_succeed_and_replay_process_logs(plan, fake_source):
    _, path = fake_source
    plan_hash = seal(plan, path)
    selected = plan["cells"][:4]
    terminal = launcher.run_cells(path, plan, plan_hash, selected, workers=2)
    assert len(terminal) == 4
    for cell in selected:
        launch = Path(plan["output_root"]) / "launches" / cell["id"]
        receipt = read(launch / "exit.json")
        assert receipt["returncode"] == 0 and receipt["status"] == "exited"
        assert receipt["log_sha256"] == digest(launch / "process.log")
        assert receipt["pid"] > 0
        assert launcher.accept_cell(plan, cell, plan_hash)["cell"] == cell["id"]


@pytest.mark.parametrize("mode", ["nonzero", "missing_result"])
def test_real_failed_or_incomplete_worker_cannot_be_accepted(plan, fake_source, mode):
    _, path = fake_source
    cell = plan["cells"][0]
    plan["fake_modes"] = {cell["id"]: mode}
    plan_hash = seal(plan, path)
    terminal = launcher.run_cells(path, plan, plan_hash, [cell], workers=1)
    assert terminal[0]["returncode"] == (7 if mode == "nonzero" else 0)
    with pytest.raises((ValueError, FileNotFoundError)):
        launcher.accept_cell(plan, cell, plan_hash)


def test_fresh_zero_exit_cannot_accept_preexisting_same_plan_result(plan, fake_source):
    _, path = fake_source
    cell = plan["cells"][0]
    plan["fake_modes"] = {cell["id"]: "noop"}
    plan_hash = seal(plan, path)
    worker_records(plan, cell, plan_hash, timestamp="2020-01-01T00:00:00+00:00")
    with pytest.raises((ValueError, FileExistsError)):
        launcher.run_cells(path, plan, plan_hash, [cell], workers=1)
        launcher.accept_cell(plan, cell, plan_hash)


@pytest.mark.parametrize("workers", [0, 5])
def test_invalid_worker_limit_creates_no_evidence(plan, fake_source, workers):
    _, path = fake_source
    plan_hash = seal(plan, path)
    with pytest.raises(ValueError):
        launcher.run_cells(path, plan, plan_hash, plan["cells"][:1], workers)
    assert not Path(plan["output_root"]).exists()


def test_existing_launch_is_not_overwritten_or_reexecuted(plan, fake_source, monkeypatch):
    _, path = fake_source
    plan_hash = seal(plan, path)
    cell = plan["cells"][0]
    launcher.run_cells(path, plan, plan_hash, [cell], workers=1)
    root = Path(plan["output_root"])
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("must not respawn"))
    with pytest.raises(FileExistsError):
        launcher.run_cells(path, plan, plan_hash, [cell], workers=1)
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def wait_ready(root, cell, process, sleep):
    deadline = time.monotonic() + 5
    marker = Path(root) / "launches" / cell["id"] / "ready.json"
    while not marker.exists():
        assert process.poll() is None, "Fake sleeper exited before readiness"
        assert time.monotonic() < deadline, "Fake sleeper failed to announce readiness"
        sleep(.01)


def ensure_stopped(processes):
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize("active_first", [False, True])
def test_real_spawn_failure_records_terminal_state_and_cleans_active_worker(
        plan, fake_source, monkeypatch, active_first):
    _, path = fake_source
    selected = plan["cells"][:2 if active_first else 1]
    if active_first:
        plan["fake_modes"] = {selected[0]["id"]: "sleep"}
    plan_hash = seal(plan, path)
    original_popen, original_sleep = subprocess.Popen, time.sleep
    processes, logs = [], []

    def spawn(command, **kwargs):
        logs.append(kwargs["stdout"])
        if active_first and not processes:
            process = original_popen(command, **kwargs)
            processes.append(process)
            wait_ready(plan["output_root"], selected[0], process, original_sleep)
            return process
        # Exercise a real OS spawn failure, rather than fabricate a receipt.
        return original_popen([str(path.parent / "missing-executable")], **kwargs)

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    try:
        with pytest.raises(FileNotFoundError):
            launcher.run_cells(path, plan, plan_hash, selected, workers=2)
        failed = read(Path(plan["output_root"]) / "launches" / selected[-1]["id"] / "exit.json")
        assert failed["status"] == "launch_failed" and failed["returncode"] is None
        assert failed["error_type"] == "FileNotFoundError"
        if active_first:
            assert processes[0].poll() is not None
            interrupted = read(Path(plan["output_root"]) / "launches" / selected[0]["id"] / "exit.json")
            assert interrupted["status"] == "launcher_interrupted"
            assert interrupted["returncode"] == processes[0].returncode
        assert all(log.closed for log in logs)
    finally:
        ensure_stopped(processes)


@pytest.mark.parametrize("ignore_term", [False, True])
def test_interruption_reaps_real_fake_worker_including_kill_fallback(
        plan, fake_source, monkeypatch, ignore_term):
    _, path = fake_source
    cell = plan["cells"][0]
    plan["fake_modes"] = {cell["id"]: "ignore_term" if ignore_term else "sleep"}
    plan_hash = seal(plan, path)
    original_popen, original_sleep = subprocess.Popen, time.sleep
    processes, logs = [], []

    def spawn(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        logs.append(kwargs["stdout"])
        processes.append(process)
        if ignore_term:
            original_wait = process.wait
            # Exercise the real TimeoutExpired/kill path without waiting five seconds.
            process.wait = lambda timeout=None: original_wait(.1 if timeout == 5 else timeout)
        return process

    interruption_sent = False

    def interrupt_after_readiness(seconds):
        nonlocal interruption_sent
        if interruption_sent:
            return original_sleep(seconds)
        wait_ready(plan["output_root"], cell, processes[0], original_sleep)
        interruption_sent = True
        raise KeyboardInterrupt("injected launcher interruption")

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(launcher.time, "sleep", interrupt_after_readiness)
    try:
        with pytest.raises(KeyboardInterrupt):
            launcher.run_cells(path, plan, plan_hash, [cell], workers=1)
        assert processes[0].poll() == (-signal.SIGKILL if ignore_term else -signal.SIGTERM)
        launch = Path(plan["output_root"]) / "launches" / cell["id"]
        receipt = read(launch / "exit.json")
        assert receipt["status"] == "launcher_interrupted"
        assert receipt["returncode"] == processes[0].returncode
        assert receipt["log_sha256"] == digest(launch / "process.log")
        assert all(log.closed for log in logs)
        with pytest.raises(ValueError):
            launcher.accept_cell(plan, cell, plan_hash)
    finally:
        ensure_stopped(processes)


@pytest.mark.parametrize("fault", ["plan_hash", "duplicate", "unknown"])
def test_cli_rejects_bad_plan_or_selection_before_spawning(plan, fake_source, monkeypatch, fault):
    _, path = fake_source
    plan_hash = seal(plan, path)
    selected = [plan["cells"][0]["id"]]
    if fault == "duplicate":
        selected *= 2
    elif fault == "unknown":
        selected = ["not-declared"]
    argv = ["launch_fitting_sensitivity.py", "--plan", str(path), "--plan-sha256",
            "wrong" if fault == "plan_hash" else plan_hash, "--cells", *selected]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn"))
    with pytest.raises(ValueError):
        launcher.main()
    assert not Path(plan["output_root"]).exists()

"""Run declared CPU fitting cells and retain each process's terminal disposition."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_fitting_sensitivity import ROOT, require, sha, validate_cells, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def accept_cell(plan, cell, plan_sha256):
    """Check execution and file identity; numerical replay is a separate check."""
    root = Path(plan["output_root"])
    output = root / cell["id"]
    launch = root / "launches" / cell["id"]
    launch_started = json.loads((launch / "started.json").read_text())
    exit_record = json.loads((launch / "exit.json").read_text())
    require(exit_record["returncode"] == 0 and exit_record["status"] == "exited",
            f"Worker did not exit successfully: {cell['id']}")
    require(exit_record["plan_sha256"] == plan_sha256 and exit_record["cell"] == cell,
            "Process record belongs to another cell or plan")
    require(all(exit_record[k] == launch_started[k] for k in
                ("cell", "plan_sha256", "command", "started_utc", "cwd")),
            "Launch and exit records differ")
    require(sha(launch / "process.log") == exit_record["log_sha256"], "Process log changed")
    require(not (output / "failed.json").exists(), "Failed worker cannot supply an accepted result")
    result = json.loads((output / "result.json").read_text())
    started = json.loads((output / "started.json").read_text())
    json.dumps(result, allow_nan=False)
    require(result["status"] == "finite_fit", "Worker did not return a finite fit")
    for record in (started, result):
        require(record["cell"] == cell and record["plan_sha256"] == plan_sha256,
                "Result identity differs from the declared cell")
        require(record["producer_commit"] == plan["producer_commit"]
                and record["source_sha256"] == plan["source_sha256"], "Result producer differs")
        require(record["runtime"] == {k: plan[k] for k in
                ("python_version", "package_versions", "runtime_identity")}, "Result runtime differs")
    require(sha(output / "fitted_arrays.npz") == result["arrays_sha256"], "Fitted arrays changed")
    return {"cell": cell["id"], "result_sha256": sha(output / "result.json"),
            "exit_sha256": sha(root / "launches" / cell["id"] / "exit.json"),
            "arrays_sha256": result["arrays_sha256"]}


def accept_all(plan, plan_sha256):
    cells = plan["cells"]
    validate_cells(cells)
    root = Path(plan["output_root"])
    expected = {c["id"] for c in cells}
    require({p.name for p in root.iterdir() if p.is_dir() and p.name != "launches"} == expected,
            "Missing or unexpected cell directories")
    require({p.name for p in (root / "launches").iterdir() if p.is_dir()} == expected,
            "Missing or unexpected process records")
    return [accept_cell(plan, cell, plan_sha256) for cell in cells]


def run_cells(plan_path, plan, plan_sha256, cells, workers):
    require(1 <= workers <= 4, "Use one through four worker processes")
    for cell in cells:
        for path in (Path(plan["output_root"]) / cell["id"],
                     Path(plan["output_root"]) / "launches" / cell["id"]):
            if path.exists():
                raise FileExistsError(f"Prior experiment output already exists: {path}")
    pending = list(cells)
    active = []
    terminal = []

    def finish(job, status="exited"):
        process, log, folder, record = job
        process.wait()
        log.close()
        receipt = {**record, "status": status, "returncode": process.returncode,
                   "finished_utc": now(), "log_sha256": sha(folder / "process.log")}
        write_json(folder / "exit.json", receipt)
        terminal.append(receipt)

    try:
        while pending or active:
            while pending and len(active) < workers:
                cell = pending.pop(0)
                folder = Path(plan["output_root"]) / "launches" / cell["id"]
                folder.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, str(ROOT / "scripts/run_fitting_sensitivity.py"),
                           "--plan", str(plan_path.resolve()), "--plan-sha256", plan_sha256,
                           "--cell", cell["id"]]
                record = {"cell": cell, "plan_sha256": plan_sha256, "command": command,
                          "started_utc": now(), "cwd": str(ROOT)}
                write_json(folder / "started.json", record)
                log = (folder / "process.log").open("xb")
                try:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL)
                except BaseException as error:
                    log.close()
                    write_json(folder / "exit.json", {**record, "status": "launch_failed", "returncode": None,
                               "finished_utc": now(), "error_type": type(error).__name__, "error": str(error)})
                    raise
                active.append((process, log, folder, {**record, "pid": process.pid}))
            for job in list(active):
                if job[0].poll() is not None:
                    active.remove(job)
                    finish(job)
            if active:
                time.sleep(.2)
    finally:
        for job in active:
            if job[0].poll() is None:
                job[0].terminate()
        for job in active:
            try:
                job[0].wait(timeout=5)
            except subprocess.TimeoutExpired:
                job[0].kill()
            finish(job, "launcher_interrupted")
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--workers", type=int, default=4)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--cells", nargs="+")
    choice.add_argument("--check-all", action="store_true")
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Study specification changed")
    plan = json.loads(args.plan.read_text())
    validate_cells(plan["cells"])
    for protected in (ROOT, Path(plan["feature_root"])):
        require(not Path(plan["output_root"]).resolve().is_relative_to(protected.resolve()),
                "Output overlaps protected inputs")
    if args.check_all:
        accepted = accept_all(plan, args.plan_sha256)
    else:
        require(len(args.cells) == len(set(args.cells)), "Duplicate requested cells")
        cells = [c for c in plan["cells"] if c["id"] in args.cells]
        require(len(cells) == len(args.cells), "Unknown requested cell")
        run_cells(args.plan, plan, args.plan_sha256, cells, args.workers)
        accepted = [accept_cell(plan, cell, args.plan_sha256) for cell in cells]
    print(json.dumps({"status": "execution_checked", "cells": accepted,
                      "limits": "Numerical replay and scientific acceptance remain separate"}))


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    main()

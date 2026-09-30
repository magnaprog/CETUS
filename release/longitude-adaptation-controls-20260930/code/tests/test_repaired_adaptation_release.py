"""Synthetic full-cohort acceptance and public-only CPU reconstruction."""

import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest
from scripts import verify_repaired_adaptation_release as v
from tests.repaired_adaptation_release_fixtures import synthetic, build_with_claims


@pytest.fixture
def packet(tmp_path):
    return synthetic(tmp_path)


def test_full_cohort_counts_nested_rates_and_nonpooled_sd(packet):
    data, f, a = packet
    result, ad, bd = v.validate_payloads(data, f, a)
    assert (
        result["test_vectors"] == 675
        and result["validation_vectors"] == 75
        and result["manifests"] == 210
    )
    assert (
        result["shared_memberships"] == 25
        and result["test_prediction_rows"] == 225 * 135
    )
    assert (
        len(ad["macro_rows"]) == 3
        and len(ad["class_rows"]) == 18
        and len(ad["training_rows"]) == 75
    )
    assert bd["lr_diagnostic_counts"] == dict(
        fits=100, fits_with_warnings=1, convergence_warnings=1, iteration_caps=1
    )

    # Independent scalar arithmetic on confusion counts, without the exporter helper.
    def f1(row):
        c = row["confusion_counts"]
        return (
            sum(2 * c[k][k] / (sum(c[k]) + sum(r[k] for r in c)) for k in range(6)) / 6
        )

    rows = data["results/adaptation.json"]["models"]["dofa"]["folds"]
    values = [sum(f1(r["finetuned"]) for r in row["seeds"]) / 5 for row in rows]
    display = ad["models"]["dofa"]["summary"]["finetuned"]["macro"]["f1"]
    assert display["mean"] == pytest.approx(100 * sum(values) / 5)
    assert display["sample_std"] == pytest.approx(100 * np.std(values, ddof=1))
    assert display["sample_std"] != pytest.approx(100 * np.std(values, ddof=0))
    weights = [r["counts"]["test"]["total"] for r in rows]
    assert np.mean(values) != pytest.approx(np.average(values, weights=weights))
    for recipe in bd["recipes"].values():
        for model in recipe.values():
            assert any(abs(x) > 0 for x in model["f1"]["difference"]["fold_values"])


@pytest.mark.parametrize(
    "defect",
    [
        "old_replay_plan",
        "same_plan",
        "old_replay_source",
        "new_control_source",
        "one_thread",
        "missing_interop",
        "boolean_interop",
        "declared_threads",
        "gate_schema",
        "gate_status",
        "gate_counts",
        "gate_report",
    ],
)
def test_public_recovery_bindings_reject_detached_corruption(packet, defect):
    original, frozen, auxiliary = packet
    data = json.loads(v.base.encoded(original))
    acceptance = data["evidence/acceptance.json"]
    run = data["evidence/runs.json"]["checkpoint_replay"][0]
    commits = data["provenance.json"]["producer_commits"]
    if defect == "old_replay_plan":
        acceptance["reports"]["checkpoint_replay"]["plan_sha256"] = (
            acceptance["inputs"]["followup_plan"]
        )
    elif defect == "same_plan":
        for value in (acceptance["inputs"], data["provenance.json"]["input_sha256"]):
            value["followup_plan"] = value["replay_plan"]
    elif defect == "old_replay_source":
        commits["checkpoint_replay"] = v.CONTROL_COMMIT
    elif defect == "new_control_source":
        commits["boundary"] = commits["buffer_sgd"] = v.adaptation.REPLAY_COMMIT
    elif defect == "one_thread":
        run["environment"]["cpu_intraop_threads"] = 1
    elif defect == "missing_interop":
        del run["environment"]["cpu_interop_threads"]
    elif defect == "boolean_interop":
        run["environment"]["cpu_interop_threads"] = True
    elif defect == "declared_threads":
        run["environment"]["declared_environment"]["OMP_NUM_THREADS"] = "1"
    elif defect == "gate_schema":
        acceptance["full_gate"]["schema_version"] = "1.0.0"
    elif defect == "gate_status":
        acceptance["full_gate"]["status"] = "complete"
    elif defect == "gate_counts":
        acceptance["full_gate"]["counts"]["replay"] = 74
    else:
        acceptance["full_gate"]["reports"]["replay"] = "0" * 64
    with pytest.raises((ValueError, KeyError)):
        v.validate_payloads(data, frozen, auxiliary)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_test",
        "missing_validation",
        "duplicate",
        "unknown_model",
        "bool_fold",
        "truth",
        "membership",
        "draw_seed",
        "draw_order",
        "normalization",
        "source",
        "protocol",
        "manifest",
        "checkpoint",
        "epoch_tie",
        "updates",
        "history_nonfinite",
        "head_mean",
        "fold_mean",
        "population_sd",
        "delta_units",
        "lr_scaler",
        "lr_warning",
        "lr_cap",
        "process_exit",
        "private_path",
        "private_key",
    ],
)
def test_semantic_corruption_is_rejected(packet, defect):
    d, f, a = packet
    pred = d["results/predictions.json.gz"]
    ar = d["results/adaptation.json"]["models"]["dinov2"]
    if defect == "missing_test":
        pred["test"].pop()
    if defect == "missing_validation":
        pred["validation"].pop()
    if defect == "duplicate":
        pred["test"][-1] = copy.deepcopy(pred["test"][0])
    if defect == "unknown_model":
        pred["test"][0]["model"] = "unknown"
    if defect == "bool_fold":
        pred["test"][0]["fold"] = False
    if defect == "truth":
        d["membership.json.gz"]["folds"][0]["test_labels"][0] = 5
    if defect == "membership":
        d["membership.json.gz"]["folds"][0]["test_ids"].reverse()
    if defect == "draw_seed":
        d["membership.json.gz"]["folds"][0]["draws"][0]["draw_seed"] = 1
    if defect == "draw_order":
        d["membership.json.gz"]["folds"][0]["draws"][0]["training_ids"].reverse()
    if defect == "normalization":
        d["membership.json.gz"]["folds"][0]["normalization_provenance"]["lower"] = 9
    if defect == "source":
        d["provenance.json"]["producer_commits"]["buffer_lr"] = "0" * 40
    if defect == "protocol":
        d["protocols/buffer_lr.json"]["specification"]["catalog"]["sha256"] = "0" * 64
    if defect == "manifest":
        d["evidence/runs.json"]["adaptation"][0]["manifest_sha256"] = "0" * 64
    if defect == "checkpoint":
        d["evidence/acceptance.json"]["replays"][0]["comparisons"]["test"][
            "prediction_mismatches"
        ] = 1
    if defect == "epoch_tie":
        ar["folds"][0]["seeds"][0]["training"]["selected_epoch"] = 50
    if defect == "updates":
        d["results/adaptation_history.json.gz"]["records"][0]["history"][
            "optimizer_steps"
        ][0] += 1
    if defect == "history_nonfinite":
        d["results/adaptation_history.json.gz"]["records"][0]["history"]["train_loss"][
            0
        ] = float("nan")
    if defect == "head_mean":
        ar["folds"][0]["seed_summaries"]["finetuned"]["macro"]["f1"]["mean"] += 0.1
    if defect == "fold_mean":
        ar["summary"]["finetuned"]["macro"]["f1"]["mean"] += 0.01
    if defect == "population_sd":
        r = ar["summary"]["finetuned"]["macro"]["f1"]
        r["sample_std"] = float(np.std(r["fold_values"], ddof=0))
    if defect == "delta_units":
        ar["summary"]["ft_minus_lr"]["macro"]["f1"]["mean"] *= 100
    fit = d["results/buffer_lr.json"]["folds"][0]["draws"][0]["fit"]
    if defect == "lr_scaler":
        fit["scaler_parameters"]["with_mean"] = False
    if defect == "lr_warning":
        fit["fit_warnings"] = []
    if defect == "lr_cap":
        fit["iteration_limit_reached"] = False
    if defect == "process_exit":
        d["evidence/acceptance.json"]["processes"]["analysis"]["exit_code"] = 1
    if defect == "private_path":
        d["results/adaptation.json"]["limitations"].append(
            "/home/private/checkouts/source"
        )
    if defect == "private_key":
        d["results/adaptation.json"]["private"] = {"hostname": "internal"}
    with pytest.raises((ValueError, KeyError, TypeError)):
        v.validate_payloads(d, f, a)


def test_earliest_selection_tie_and_valid_zero_updates_rejected(packet):
    d, f, a = packet
    h = d["results/adaptation_history.json.gz"]["records"][0]["history"]
    assert h["val_balanced_accuracy"][0] == h["val_balanced_accuracy"][-1]
    v.validate_payloads(d, f, a)
    h["optimizer_steps"][0] = 0
    h["amp_skipped_steps"][0] = 6
    with pytest.raises(ValueError, match="History"):
        v.validate_payloads(d, f, a)


def test_two_exports_byte_identical_and_exact_table_inventory(packet, tmp_path):
    d, f, a = packet
    first = tmp_path / "first"
    second = tmp_path / "second"
    left = build_with_claims(d, f, a, first)
    right = build_with_claims(d, f, a, second)
    assert left["acceptance_sha256"] == right["acceptance_sha256"]
    assert v.base.inventory(first) == v.base.inventory(second)
    v.base.verify_indexes(first)
    assert not (first / ".INCOMPLETE").exists()
    assert (
        json.loads((tmp_path / "first.receipt.json").read_text())["status"]
        == "complete"
    )
    for name, count in [
        ("summary", 72),
        ("folds", 360),
        ("lr_fits", 100),
        ("support", 5),
    ]:
        with (first / f"display/substitution/{name}.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == count
    for name, count in [("macro", 3), ("classes", 18), ("training", 75)]:
        with (
            first / f"display/adaptation/repaired_adaptation_{name}_20260930.csv"
        ).open() as stream:
            assert len(list(csv.DictReader(stream))) == count
    assert len(v.base.read_json(first / "results/predictions.json.gz")["test"]) == 675
    assert v.CLAIMS_FILE in v.base.inventory(first)
    assert (first / "code/requirements.txt").read_text() == v.CPU_REQUIREMENTS
    assert len(v.CPU_REQUIREMENTS.splitlines()) == 11
    (second / ".git").mkdir()
    with pytest.raises(ValueError, match="Git metadata"):
        v.verify_package(second, f, a, right["acceptance_sha256"])


def test_render_dirty_rc_is_byte_identical_and_restores_caller(packet, tmp_path):
    import matplotlib as mpl

    mpl.use("Agg")
    data, frozen, auxiliary = packet
    _, adaptation, substitution = v.validate_payloads(data, frozen, auxiliary)
    clean = tmp_path / "clean-render"
    dirty = tmp_path / "dirty-render"
    with mpl.rc_context(mpl.rcParamsDefault):
        clean_before = dict(mpl.rcParams)
        v.render(adaptation, substitution, clean)
        clean_after = dict(mpl.rcParams)
    with mpl.rc_context(
        {
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 13,
            "figure.dpi": 173,
        }
    ):
        dirty_before = dict(mpl.rcParams)
        v.render(adaptation, substitution, dirty)
        dirty_after = dict(mpl.rcParams)
    assert len(list(clean.rglob("*.pdf"))) == 2
    assert len(list(clean.rglob("*.png"))) == 2
    assert v.base.inventory(clean) == v.base.inventory(dirty)
    assert clean_after == clean_before
    assert dirty_after == dirty_before


def test_render_restores_caller_rc_after_plot_failure(packet, tmp_path, monkeypatch):
    import matplotlib as mpl

    mpl.use("Agg")
    data, frozen, auxiliary = packet
    _, adaptation, substitution = v.validate_payloads(data, frozen, auxiliary)

    def fail_plot(report):
        mpl.rcParams.update({"axes.labelsize": 23, "axes.facecolor": "pink"})
        raise RuntimeError("synthetic plotting failure")

    monkeypatch.setattr(v.display_b, "make_figure", fail_plot)
    with mpl.rc_context({"axes.labelsize": 19, "figure.dpi": 173}):
        before = dict(mpl.rcParams)
        with pytest.raises(RuntimeError, match="synthetic plotting failure"):
            v.render(adaptation, substitution, tmp_path / "failed-render")
        assert dict(mpl.rcParams) == before


def test_public_only_process_blocks_torch_git_network_and_private_source(
    packet, tmp_path
):
    d, f, a = packet
    p = tmp_path / "package"
    r = build_with_claims(d, f, a, p)
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    for src, name in ((p, "package"), (f, "frozen"), (a, "auxiliary")):
        shutil.copytree(src, isolated / name)
    # Python -I removes cwd/user-site. Only the CPU dependencies and exported
    # source are admitted explicitly; block model imports and external actions.
    cpu_site = str(Path(np.__file__).resolve().parents[1])
    driver = """import sys, importlib.abc, socket, subprocess, runpy
sys.path.insert(0, CPU_SITE)
sys.path.insert(0, PUBLIC_CODE)
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','titansar','transformers','timm'}: raise RuntimeError('forbidden import '+fullname)
sys.meta_path.insert(0,Block())
def denied(*a,**k): raise RuntimeError('external operation forbidden')
socket.socket.connect=denied
socket.socket.connect_ex=denied
def no_process(command,*a,**k):
 if command[0]=='fc-list': raise FileNotFoundError('exercise Matplotlib bundled-font fallback')
 return denied(command,*a,**k)
subprocess.Popen=no_process
sys.argv=['verify','--package',PACKAGE,'--frozen-package',FROZEN,'--auxiliary-package',AUX,'--acceptance-sha256',PIN,'--output',OUTPUT]
import scripts.verify_repaired_adaptation_release as public
from pathlib import Path
assert public.main()==0
# Mutate actual imported bytes after startup, then restore the isolated copy.
source=Path(public.__file__); original=source.read_bytes()
try:
 source.write_bytes(original+b'\\n# changed after import\\n')
 try: public.source_snapshot()
 except ValueError as error: assert 'source changed' in str(error)
 else: raise AssertionError('Source mutation was accepted')
finally: source.write_bytes(original)
"""
    values = dict(
        CPU_SITE=cpu_site,
        PUBLIC_CODE=str(isolated / "package/code"),
        PACKAGE=str(isolated / "package"),
        FROZEN=str(isolated / "frozen"),
        AUX=str(isolated / "auxiliary"),
        PIN=r["acceptance_sha256"],
        OUTPUT=str(isolated / "rebuild"),
    )
    program = (
        "\n".join(k + "=" + repr(value) for k, value in values.items()) + "\n" + driver
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(tmp_path / "mpl"),
    }
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=isolated,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["test_vectors"] == 675
    assert v.base.inventory(isolated / "package") == v.base.inventory(
        isolated / "rebuild"
    )


def test_rehashed_corruption_still_fails_scientific_reconstruction(packet, tmp_path):
    d, f, a = packet
    p = tmp_path / "package"
    build_with_claims(d, f, a, p)
    target = p / "results/buffer_sgd.json"
    value = v.base.read_json(target)
    value["models"]["dinov2"]["difference"]["f1"]["sample_std"] = 0
    target.write_bytes(v.base.encoded(value))
    # Even an internally consistent replacement hash inventory cannot repair a
    # wrong scientific aggregate. This separately exercises arithmetic from pins.
    acceptance = v.base.read_json(p / "evidence/acceptance.json")
    acceptance["public_files_sha256"]["results/buffer_sgd.json"] = v.base.sha256(target)
    (p / "evidence/acceptance.json").write_bytes(v.base.encoded(acceptance))
    v.base.write_indexes(p, dict(schema_version="1.0.0"))
    with pytest.raises(ValueError, match="aggregate"):
        v.verify_package(p, f, a, v.base.sha256(p / "evidence/acceptance.json"))


def test_old_acceptance_pin_rejects_rehashed_projection(packet, tmp_path):
    d, f, a = packet
    p = tmp_path / "package"
    r = build_with_claims(d, f, a, p)
    target = p / "results/buffer_lr.json"
    value = v.base.read_json(target)
    value["parents"]["common_plan"]["sha256"] = "0" * 64
    target.write_bytes(v.base.encoded(value))
    v.base.write_indexes(p, dict(schema_version="1.0.0"))
    with pytest.raises(ValueError, match="Externally bound"):
        v.verify_package(p, f, a, r["acceptance_sha256"])


def test_output_protection_and_input_mutation(packet, tmp_path, monkeypatch):
    d, f, a = packet
    for output in (f, f / "bad", a / "bad", v.ROOT / "bad"):
        with pytest.raises(ValueError):
            build_with_claims(d, f, a, output)
    tracked_file = tmp_path / "bound.json"
    tracked_file.write_text("{}")
    # Put the input in its own directory so a sibling fresh output is valid.
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    tracked_file.rename(inputs / "bound.json")
    tracked_file = inputs / "bound.json"
    original = v.render

    def mutate(*args):
        original(*args)
        tracked_file.write_text('{"changed":true}')

    monkeypatch.setattr(v, "render", mutate)
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="changed"):
        build_with_claims(d, f, a, out, {tracked_file: v.base.sha256(tracked_file)})
    assert (out / ".INCOMPLETE").exists() and not (
        tmp_path / "failed.receipt.json"
    ).exists()


@pytest.mark.parametrize("mode", ["short", "close"])
def test_checked_publisher_short_write_and_close_failure(tmp_path, monkeypatch, mode):
    real = Path.open

    class Stream:
        def __init__(self, p, *a, **k):
            self.raw = real(p, *a, **k)

        def __enter__(self):
            return self

        def write(self, b):
            return self.raw.write(b[:-1]) if mode == "short" else self.raw.write(b)

        def __exit__(self, *args):
            self.raw.close()
            if mode == "close":
                raise OSError("synthetic close failure")

    monkeypatch.setattr(Path, "open", lambda p, *a, **k: Stream(p, *a, **k))
    destination = tmp_path / "receipt.json"
    with pytest.raises((ValueError, OSError)):
        v.write_new(destination, b'{"status":"complete"}\n')
    assert not destination.exists() and (tmp_path / ".receipt.json.pending").exists()


def test_changed_imported_source_fails_before_output(packet, tmp_path, monkeypatch):
    d, f, a = packet
    actual = v.base.sha256
    monkeypatch.setattr(
        v.base,
        "sha256",
        lambda p: "0" * 64 if Path(p) == Path(v.__file__) else actual(p),
    )
    with pytest.raises(ValueError, match="source changed"):
        build_with_claims(d, f, a, tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_rehashed_table_corruption_rejected(packet, tmp_path):
    d, f, a = packet
    p = tmp_path / "table"
    build_with_claims(d, f, a, p)
    target = p / "display/substitution/folds.csv"
    target.write_bytes(target.read_bytes().replace(b"dinov2", b"BROKEN", 1))
    acc = v.base.read_json(p / "evidence/acceptance.json")
    acc["public_files_sha256"][target.relative_to(p).as_posix()] = v.base.sha256(target)
    (p / "evidence/acceptance.json").write_bytes(v.base.encoded(acc))
    v.base.write_indexes(p, dict(schema_version="1.0.0"))
    with pytest.raises(ValueError, match="table differs"):
        v.verify_package(p, f, a, v.base.sha256(p / "evidence/acceptance.json"))


def test_helper_projection_changes_only_denylist_constants():
    name = "scripts/export_repaired_release.py"
    raw = (v.ROOT / name).read_bytes()
    projected = v.public_helper_bytes(name, raw)
    # Prove every source byte outside the denylist fragment is unchanged.
    import re

    assert re.sub(rb'"gpu-[a-z]+", "gpu-[a-z]+",', b'"gpu-",', raw) == projected
    assert projected.count(b'"gpu-"') == 1
    assert v.base.sha256(v.ROOT / name) == v.HELPERS[name]


@pytest.mark.parametrize(
    "defect",
    [
        "summary_empty",
        "summary_short",
        "summary_bool",
        "ft_protocol_empty",
        "sgd_protocol_empty",
        "sgd_draws_empty",
        "sgd_draws_short",
        "replay_test_metric",
        "replay_validation_metric",
        "expected_validation",
        "seed_replay_manifest",
        "seed_replay_result",
    ],
)
def test_v2_detached_grid_and_replay_regressions(packet, defect):
    original, f, a = packet
    # deepcopy retains aliases; roundtrip matches the independent JSON payloads.
    d = json.loads(v.base.encoded(original))
    mutate_v2_projection(d, defect)
    with pytest.raises(ValueError):
        v.validate_payloads(d, f, a)


def mutate_v2_projection(d, defect):
    folds = d["results/buffer_sgd.json"]["models"]["dinov2"]["folds"]
    if defect == "summary_empty":
        folds.clear()
    elif defect == "summary_short":
        folds.pop()
    elif defect == "summary_bool":
        folds[0]["fold"] = False
    elif defect in ("ft_protocol_empty", "sgd_protocol_empty"):
        family = "adaptation" if defect == "ft_protocol_empty" else "buffer_sgd"
        d[f"protocols/{family}.json"]["specification"]["folds"] = []
    elif defect in ("sgd_draws_empty", "sgd_draws_short"):
        draws = d["protocols/buffer_sgd.json"]["specification"]["folds"][0]["draws"]
        draws.clear() if defect == "sgd_draws_empty" else draws.pop()
    elif defect == "replay_test_metric":
        d["evidence/acceptance.json"]["replays"][0]["test_metrics"]["macro"]["f1"] = (
            0.0123456
        )
    elif defect == "replay_validation_metric":
        d["evidence/acceptance.json"]["replays"][0]["validation_metrics"]["macro"][
            "recall"
        ] = 0.0123456
    elif defect == "expected_validation":
        d["evidence/acceptance.json"]["replays"][0]["comparisons"]["validation"][
            "expected_macro_recall"
        ] = 0.0123456
    else:
        field = (
            "replay_manifest_sha256"
            if defect == "seed_replay_manifest"
            else "replay_sha256"
        )
        d["results/adaptation.json"]["models"]["dinov2"]["folds"][0]["seeds"][0][
            field
        ] = "0" * 64
    for family in ("adaptation", "buffer_sgd"):
        p = d[f"protocols/{family}.json"]
        p["extract_specification_sha256"] = v.digest(p["specification"])


@pytest.mark.parametrize("defect", ["summary_short", "replay_test_metric"])
def test_v2_rehashed_package_regressions(packet, tmp_path, defect):
    d, f, a = packet
    package = tmp_path / "package"
    accepted = build_with_claims(d, f, a, package)
    d = {name: v.base.read_json(package / name) for name in v.JSON_FILES}
    mutate_v2_projection(d, defect)
    for name in ("results/buffer_sgd.json", "evidence/acceptance.json"):
        (package / name).write_bytes(v.base.encoded(d[name]))
    acceptance = v.base.read_json(package / "evidence/acceptance.json")
    acceptance["public_files_sha256"] = {
        name: row["sha256"]
        for name, row in v.base.inventory(package).items()
        if name not in ("evidence/acceptance.json", "file_index.json", "SHA256SUMS")
    }
    (package / "evidence/acceptance.json").write_bytes(v.base.encoded(acceptance))
    v.base.write_indexes(package, dict(schema_version="1.0.0"))
    with pytest.raises(ValueError, match="External acceptance"):
        v.verify_package(package, f, a, accepted["acceptance_sha256"])
    # Deliberately supply a replacement pin to isolate semantic validation.
    with pytest.raises(ValueError):
        v.verify_package(
            package, f, a, v.base.sha256(package / "evidence/acceptance.json")
        )


@pytest.fixture(scope="module")
def claims_packet(tmp_path_factory):
    from tests.repaired_adaptation_release_fixtures import synthetic_claims

    root = tmp_path_factory.mktemp("synthetic-claims")
    data, frozen, auxiliary = synthetic(root)
    _, a, b = v.validate_payloads(data, frozen, auxiliary)
    display = root / "display-package"
    v.render(a, b, display)
    for name in v.CLAIM_SOURCES:
        if name not in v.CLAIM_DISPLAYS:
            v.write_json(display / name, data[name])
    claims = synthetic_claims(data, frozen, auxiliary)
    return display, a, b, claims


def rebind_synthetic_claims(claims):
    specification = claims["specification"]
    specification["required_entry_ids"] = [
        row["id"] for row in specification["entries"]
    ]
    claims["numeric_check"]["specification_sha256"] = v.digest(specification)


def test_claims_complete_grid_units_and_finite_calculations(claims_packet):
    root, a, b, claims = claims_packet
    assert v.validate_claims(claims, root, a, b) == claims
    fixed, _ = v.claim_catalog(a, b)
    assert len(fixed) == 386
    assert sum(name.startswith("macro.") for name in fixed) == 24
    assert sum(name.startswith("class.") for name in fixed) == 234
    assert sum(name.startswith("figure_a.") for name in fixed) == 72
    assert sum(name.startswith("figure_b.") for name in fixed) == 56
    rows = {r["id"]: r for r in claims["specification"]["entries"]}
    assert rows["full.prose"]["value"] == 75
    support = b["support_design"][0]
    assert rows["full.support_table"]["value"] == (
        100 * support["boundary_total"] / support["train_total"]
    )
    row = rows["macro.dinov2.ft_f1.mean"]
    assert row["value"] == a["macro_rows"][0]["ft_f1_mean"]
    assert row["source_unit"] == row["display_unit"] == "percent"
    assert row["scale"] == 1
    assert any(row["value"] < 0 for row in rows.values())


@pytest.mark.parametrize(
    "defect",
    [
        "missing_point",
        "missing_prose",
        "missing_support",
        "missing_occurrence",
        "duplicate_occurrence",
        "wrong_model",
        "scale_100",
        "wrong_units",
        "rounded_value",
        "wrong_text",
        "wrong_format",
        "cross_fold_percentage",
        "arbitrary_pointer",
        "negative_index",
        "boolean_index",
        "unknown_calculation",
        "pending",
        "bool_exit",
        "stale_specification",
        "changed_artifact",
        "changed_display",
        "wrong_document",
        "wrong_support_label",
        "private_document",
    ],
)
def test_claims_mutations_fail_after_detached_registration(claims_packet, defect):
    root, a, b, original = claims_packet
    claims = json.loads(v.base.encoded(original))
    spec = claims["specification"]
    entries = spec["entries"]
    fixed = next(r for r in entries if r["id"] == "macro.dinov2.ft_f1.mean")
    support = next(r for r in entries if r["id"] == "full.support_table")
    prose = next(r for r in entries if r["id"] == "full.prose")
    if defect == "missing_point":
        entries.remove(next(r for r in entries if r["id"].startswith("figure_a.")))
    elif defect in ("missing_prose", "missing_support"):
        entries.remove(prose if defect == "missing_prose" else support)
    elif defect == "missing_occurrence":
        fixed["target"]["occurrences"].pop()
    elif defect == "duplicate_occurrence":
        prose["target"]["occurrences"].append(
            copy.deepcopy(prose["target"]["occurrences"][0])
        )
    elif defect == "wrong_model":
        fixed["sources"][0]["pointer"][1] = 1
    elif defect == "scale_100":
        fixed["scale"] = 100
    elif defect == "wrong_units":
        fixed["source_unit"] = "fraction"
    elif defect == "rounded_value":
        fixed["value"] = round(fixed["value"], 2) + 0.01
    elif defect == "wrong_text":
        fixed["text"] = "999.00"
    elif defect == "wrong_format":
        fixed["format"] = "signed_two_decimals"
    elif defect == "cross_fold_percentage":
        support["sources"][1]["pointer"][1] = 1
    elif defect == "arbitrary_pointer":
        prose["sources"][0]["pointer"] = ["class_rows"]
    elif defect in ("negative_index", "boolean_index"):
        support["sources"][0]["pointer"][1] = (
            -1 if defect == "negative_index" else False
        )
    elif defect == "unknown_calculation":
        prose["calculation"] = "sum"
    elif defect == "pending":
        claims["numeric_check"]["status"] = "pending_final_tex_mapping"
    elif defect == "bool_exit":
        claims["numeric_check"]["returncode"] = False
    elif defect == "stale_specification":
        spec["documents"][0]["tex_sha256"] = "1" * 64
    elif defect == "changed_artifact":
        spec["artifact_files_sha256"][v.CLAIM_ARTIFACTS[2]] = "0" * 64
    elif defect == "changed_display":
        spec["source_files_sha256"][v.CLAIM_DISPLAYS[0]] = "0" * 64
    elif defect == "wrong_document":
        prose["target"]["file"] = spec["documents"][1]["tex_file"]
    elif defect == "wrong_support_label":
        support["target"]["label"] = "tab:other"
    else:
        spec["documents"][0]["tex_file"] = "/private/manuscript.tex"
    if defect != "stale_specification":
        rebind_synthetic_claims(claims)
    with pytest.raises((ValueError, KeyError, TypeError)):
        v.validate_claims(claims, root, a, b)


@pytest.mark.parametrize("value", [-0.0001, -0.0, 0.0, 0.0001])
def test_claim_formats_preserve_coordinate_and_normalize_printed_zero(value):
    assert v.claim_number(value, "coordinate") is None
    assert v.claim_number(value, "unsigned_two_decimals") == "0.00"
    assert v.claim_number(value, "signed_two_decimals") == "0.00"
    assert value == float(value)


def test_claim_formats_keep_signs_and_reject_unsupported_values():
    assert v.claim_number(-1.234, "unsigned_two_decimals") == "-1.23"
    assert v.claim_number(-1.234, "signed_two_decimals") == "-1.23"
    assert v.claim_number(1.234, "signed_two_decimals") == "+1.23"
    assert v.claim_number(75, "integer") == "75"
    for value, mode in [
        (True, "integer"),
        (1.5, "integer"),
        (float("nan"), "coordinate"),
        (float("inf"), "coordinate"),
        (1.0, "scientific"),
    ]:
        with pytest.raises(ValueError):
            v.claim_number(value, mode)


def test_missing_final_claims_fails_before_publication(packet, tmp_path):
    data, frozen, auxiliary = packet
    output = tmp_path / "unmapped"
    with pytest.raises(ValueError, match="Accepted final claims map"):
        v.build_package(data, frozen, auxiliary, output)
    assert not output.exists()
    assert not output.with_name(output.name + ".receipt.json").exists()


def test_rehashed_claim_scalar_cannot_approve_complete_package(packet, tmp_path):
    data, frozen, auxiliary = packet
    output = tmp_path / "claims-corruption"
    build_with_claims(data, frozen, auxiliary, output)
    claims = v.base.read_json(output / v.CLAIMS_FILE)
    claims["specification"]["entries"][0]["value"] += 1
    rebind_synthetic_claims(claims)
    (output / v.CLAIMS_FILE).write_bytes(v.base.encoded(claims))
    acceptance = v.base.read_json(output / "evidence/acceptance.json")
    acceptance["public_files_sha256"][v.CLAIMS_FILE] = v.base.sha256(
        output / v.CLAIMS_FILE
    )
    (output / "evidence/acceptance.json").write_bytes(v.base.encoded(acceptance))
    v.base.write_indexes(output, dict(schema_version="1.0.0"))
    with pytest.raises(ValueError, match="Claim scalar differs"):
        v.verify_package(
            output,
            frozen,
            auxiliary,
            v.base.sha256(output / "evidence/acceptance.json"),
        )




@pytest.mark.parametrize(
    "source,pointer,calculation,unit,expected",
    [
        (
            v.CLAIM_PROTOCOL,
            ["specification", "recipe", "epochs"],
            "scalar",
            "count",
            50,
        ),
        (
            v.CLAIM_PROTOCOL,
            ["specification", "recipe", "unfreeze_blocks"],
            "scalar",
            "count",
            2,
        ),
        (
            v.CLAIM_PROTOCOL,
            ["specification", "optimization_seeds"],
            "count",
            "count",
            5,
        ),
        (v.CLAIM_DISPLAYS[0], ["training_rows"], "count", "count", 75),
        (
            v.CLAIM_DISPLAYS[0],
            ["training_summary", 0, "selected_epoch_min"],
            "scalar",
            "count",
            None,
        ),
        (
            v.CLAIM_DISPLAYS[0],
            ["training_summary", 0, "selected_epoch_max"],
            "scalar",
            "count",
            None,
        ),
        (
            v.CLAIM_DISPLAYS[0],
            [
                "models",
                "croma",
                "summary",
                "ft_minus_lr",
                "macro",
                "recall",
                "sample_std",
            ],
            "scalar",
            "percentage points",
            None,
        ),
        (
            v.CLAIM_DISPLAYS[1],
            ["recipes", "lr", "croma", "f1", "difference", "sample_std"],
            "scalar",
            "percentage points",
            None,
        ),
    ],
)
def test_explicit_design_and_contrast_sources(
    claims_packet, source, pointer, calculation, unit, expected
):
    root, a, b, original = claims_packet
    claims = copy.deepcopy(original)
    data = dict(zip(v.CLAIM_DISPLAYS, (a, b)))
    data[v.CLAIM_PROTOCOL] = v.base.read_json(root / v.CLAIM_PROTOCOL)
    row = dict(
        id="full.explicit_design",
        target=dict(
            kind="prose",
            file="full/main.tex",
            label="sec:adaptation",
            registration="explicit_design",
            occurrences=[
                dict(document="full", registration="explicit_design", ordinal=0)
            ],
        ),
        sources=[dict(file=source, pointer=pointer)],
        calculation=calculation,
        source_unit=unit,
        display_unit=unit,
        scale=1,
        format="integer" if unit == "count" else "unsigned_two_decimals",
    )
    row["value"] = v.claim_value(row, data)
    row["text"] = v.claim_number(row["value"], row["format"])
    if expected is not None:
        assert row["value"] == expected and row["text"] == str(expected)
    claims["specification"]["entries"].append(row)
    claims["specification"]["entries"].sort(key=lambda entry: entry["id"])
    rebind_synthetic_claims(claims)
    v.validate_claims(claims, root, a, b)
    row["sources"][0]["pointer"] = ["specification", "recipe", "T_max"]
    rebind_synthetic_claims(claims)
    with pytest.raises(ValueError, match="Unregistered scalar"):
        v.validate_claims(claims, root, a, b)


@pytest.mark.parametrize(
    "defect",
    ["epochs", "unfreeze_blocks", "seed_order", "boolean_seed", "missing_identity"],
)
def test_explicit_design_sources_fail_closed_under_rehash(
    claims_packet, tmp_path, defect
):
    source, a, b, original = claims_packet
    root = tmp_path / "rehashed-design"
    shutil.copytree(source, root)
    claims = copy.deepcopy(original)
    protocol = v.base.read_json(root / v.CLAIM_PROTOCOL)
    if defect in ("epochs", "unfreeze_blocks"):
        protocol["specification"]["recipe"][defect] += 1
    elif defect == "seed_order":
        protocol["specification"]["optimization_seeds"].reverse()
    elif defect == "boolean_seed":
        protocol["specification"]["optimization_seeds"][0] = False
    else:
        del claims["specification"]["source_files_sha256"][v.CLAIM_PROTOCOL]
    protocol["extract_specification_sha256"] = v.digest(protocol["specification"])
    (root / v.CLAIM_PROTOCOL).write_bytes(v.base.encoded(protocol))
    if defect != "missing_identity":
        claims["specification"]["source_files_sha256"][v.CLAIM_PROTOCOL] = (
            v.base.sha256(root / v.CLAIM_PROTOCOL)
        )
    rebind_synthetic_claims(claims)
    with pytest.raises(ValueError, match="design sources|source/artifact inventory"):
        v.validate_claims(claims, root, a, b)


def test_epoch_range_is_two_integer_entries():
    assert [v.claim_number(value, "integer") for value in (1, 14, 50, 2, 5, 75)] == [
        "1",
        "14",
        "50",
        "2",
        "5",
        "75",
    ]
    with pytest.raises(ValueError, match="finite"):
        v.claim_number("1--14", "integer")


def append_printed_claim(claims, sources, calculation, unit, mode, value):
    row = dict(
        id="full.final_draft_scalar",
        target=dict(
            kind="prose",
            file="full/main.tex",
            label="sec:adaptation",
            registration="final_draft_scalar",
            occurrences=[
                dict(document="full", registration="final_draft_scalar", ordinal=0)
            ],
        ),
        sources=sources,
        calculation=calculation,
        source_unit=unit,
        display_unit=unit,
        scale=1,
        format=mode,
        value=value,
        text=v.claim_number(value, mode),
    )
    claims["specification"]["entries"].append(row)
    claims["specification"]["entries"].sort(key=lambda entry: entry["id"])
    rebind_synthetic_claims(claims)
    return row


@pytest.mark.parametrize(
    "file,pointer,calculation,unit,mode,expected",
    [
        (
            v.CLAIM_PROTOCOL,
            ["specification", "recipe", "backbone_lr"],
            "scalar",
            "dimensionless",
            "tex_power_of_ten",
            0.0001,
        ),
        (
            v.CLAIM_PROTOCOL,
            ["specification", "recipe", "head_lr"],
            "scalar",
            "dimensionless",
            "tex_power_of_ten",
            0.001,
        ),
        (
            v.CLAIM_PROTOCOL,
            ["specification", "recipe", "weight_decay"],
            "scalar",
            "dimensionless",
            "unsigned_two_decimals",
            0.01,
        ),
        (
            v.CLAIM_SGD_PROTOCOL,
            ["specification", "recipe", "probe_epochs"],
            "scalar",
            "count",
            "integer",
            100,
        ),
        (
            v.CLAIM_SGD_PROTOCOL,
            ["specification", "recipe", "probe_batch_size"],
            "scalar",
            "count",
            "integer",
            256,
        ),
        (
            v.CLAIM_SGD_PROTOCOL,
            ["specification", "recipe", "probe_lr"],
            "scalar",
            "dimensionless",
            "unsigned_two_decimals",
            0.01,
        ),
        (
            v.CLAIM_SGD_PROTOCOL,
            ["specification", "recipe", "momentum"],
            "scalar",
            "dimensionless",
            "unsigned_one_decimal",
            0.9,
        ),
        (v.CLAIM_ANALYSIS, ["validation", "checkpoint_replays"], "scalar", "count", "integer", 75),
        (
            v.CLAIM_MEMBERSHIP,
            ["folds", 0, "normalization_provenance", "sampling", "selected_tile_ids"],
            "count",
            "count",
            "integer",
            6,
        ),
        (
            v.CLAIM_DISPLAYS[1],
            ["support_design", 0, "fold"],
            "scalar",
            "index",
            "integer",
            0,
        ),
    ],
)
def test_final_draft_sources_are_explicit(
    claims_packet, file, pointer, calculation, unit, mode, expected
):
    root, a, b, original = claims_packet
    claims = copy.deepcopy(original)
    append_printed_claim(
        claims, [dict(file=file, pointer=pointer)], calculation, unit, mode, expected
    )
    v.validate_claims(claims, root, a, b)
    assert (
        "evidence/acceptance.json" not in claims["specification"]["source_files_sha256"]
    )


@pytest.mark.parametrize(
    "defect", [None, "omit_list", "repeat_list", "substitute_replays", "wrong_total"]
)
def test_sgd_head_count_uses_all_disjoint_record_lists(claims_packet, defect):
    root, a, b, original = claims_packet
    claims = copy.deepcopy(original)
    sources = [
        dict(file=v.CLAIM_SGD_RESULTS, pointer=["folds", i, "runs"]) for i in range(20)
    ]
    value = 500
    if defect == "omit_list":
        sources.pop()
    elif defect == "repeat_list":
        sources[-1] = copy.deepcopy(sources[0])
    elif defect == "substitute_replays":
        sources = [dict(file=v.CLAIM_ANALYSIS, pointer=["validation", "checkpoint_replays"])]
    elif defect == "wrong_total":
        value = 499
    append_printed_claim(claims, sources, "count", "count", "integer", value)
    if defect is None:
        v.validate_claims(claims, root, a, b)
    else:
        with pytest.raises(ValueError):
            v.validate_claims(claims, root, a, b)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_cell",
        "duplicate_cell",
        "wrong_format",
        "wrong_label",
        "unrelated_counter",
        "fractional_count",
    ],
)
def test_inline_training_table_has_exact_counter_coverage(claims_packet, defect):
    root, a, b, original = claims_packet
    claims = copy.deepcopy(original)
    rows = claims["specification"]["entries"]
    row = next(r for r in rows if r["id"] == "full.training.0.optimizer_steps_total")
    assert sum(r["target"]["kind"] == "training_table" for r in rows) == 36
    if defect == "missing_cell":
        rows.remove(row)
    elif defect == "duplicate_cell":
        duplicate = copy.deepcopy(row)
        duplicate["id"] += ".copy"
        duplicate["target"]["occurrences"][0]["ordinal"] = 1
        rows.append(duplicate)
        rows.sort(key=lambda r: r["id"])
    elif defect == "wrong_format":
        row["format"] = "integer"
        row["text"] = v.claim_number(row["value"], row["format"])
    elif defect == "wrong_label":
        row["target"]["label"] = "tab:repairedcontrolsupport"
    elif defect == "unrelated_counter":
        row["sources"][0]["pointer"][-1] = "jobs"
    else:
        row["value"] += 0.5
    rebind_synthetic_claims(claims)
    with pytest.raises(ValueError):
        v.validate_claims(claims, root, a, b)


@pytest.mark.parametrize(
    "value,mode,printed",
    [
        (527750, "tex_grouped_integer", "527{,}750"),
        (16871750, "tex_grouped_integer", "16{,}871{,}750"),
        (275, "tex_grouped_integer", "275"),
        (0.0001, "tex_power_of_ten", "10^{-4}"),
        (0.001, "tex_power_of_ten", "10^{-3}"),
        (0.9, "unsigned_one_decimal", "0.9"),
        (-0.001, "unsigned_one_decimal", "0.0"),
    ],
)
def test_final_draft_printing_is_exact(value, mode, printed):
    assert v.claim_number(value, mode) == printed


def test_final_formats_do_not_infer_words_ranges_or_arbitrary_powers():
    for value, mode in [
        (1.5, "tex_grouped_integer"),
        (0.01, "tex_power_of_ten"),
        (5, "english_word"),
        ("1--14", "integer"),
    ]:
        with pytest.raises(ValueError):
            v.claim_number(value, mode)
    assert "Narrative English quantities" in v.CLAIMS_SCOPE
    assert "external audit bookkeeping" in v.CLAIMS_SCOPE
    assert "separately checks English-word quantities" in v.README
    assert "evidence/acceptance.json" not in v.CLAIM_SOURCES


def test_both_paper_table_grids_have_622_scalars(claims_packet):
    root, a, b, claims = claims_packet
    v.validate_claims(claims, root, a, b)
    entries = claims["specification"]["entries"]
    table_kinds = {"table", "training_table", "support_table"}
    assert (
        sum(
            len(row["target"]["occurrences"])
            for row in entries
            if row["target"]["kind"] in table_kinds
        )
        == 622
    )
    assert sum(row["target"]["kind"] == "support_table" for row in entries) == 70


@pytest.mark.parametrize(
    "defect", ["missing_column", "duplicate_column", "wrong_class", "wrong_format"]
)
def test_support_table_requires_all_five_rows_and_seven_columns(claims_packet, defect):
    root, a, b, original = claims_packet
    claims = copy.deepcopy(original)
    entries = claims["specification"]["entries"]
    row = next(row for row in entries if row["id"] == "full.support.1.training")
    if defect == "missing_column":
        entries.remove(row)
    elif defect == "duplicate_column":
        extra = copy.deepcopy(row)
        extra["id"] += ".copy"
        extra["target"]["occurrences"][0]["ordinal"] = 1
        entries.append(extra)
        entries.sort(key=lambda row: row["id"])
    elif defect == "wrong_class":
        row["sources"][0]["pointer"] = ["support_design", 1, "test_class_counts", 4]
    else:
        row["format"] = "integer"
    rebind_synthetic_claims(claims)
    with pytest.raises(ValueError):
        v.validate_claims(claims, root, a, b)




def test_claims_specification_validates_before_any_checker_receipt(claims_packet):
    root, a, b, original = claims_packet
    specification = copy.deepcopy(original["specification"])
    assert "numeric_check" not in specification
    assert v.validate_claims_specification(specification, root, a, b) is specification


@pytest.mark.parametrize("defect", ["source", "value", "support_grid", "extra_key"])
def test_pure_specification_validator_retains_bound_finite_checks(claims_packet, defect):
    root, a, b, original = claims_packet
    specification = copy.deepcopy(original["specification"])
    if defect == "source":
        specification["source_files_sha256"][v.CLAIM_ANALYSIS] = "0" * 64
    elif defect == "value":
        specification["entries"][0]["value"] += 1
    elif defect == "support_grid":
        rows = specification["entries"]
        rows.remove(next(row for row in rows if row["target"]["kind"] == "support_table"))
        specification["required_entry_ids"] = [row["id"] for row in rows]
    else:
        specification["unregistered"] = True
    with pytest.raises(ValueError):
        v.validate_claims_specification(specification, root, a, b)


def test_receipt_validator_delegates_only_after_receipt_binding(claims_packet, monkeypatch):
    root, a, b, original = claims_packet
    calls = []
    validate = v.validate_claims_specification

    def record(specification, root_arg, a_arg, b_arg):
        calls.append(specification)
        return validate(specification, root_arg, a_arg, b_arg)

    monkeypatch.setattr(v, "validate_claims_specification", record)
    assert v.validate_claims(original, root, a, b) is original
    assert calls == [original["specification"]]
    changed = copy.deepcopy(original)
    changed["numeric_check"]["specification_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Numeric check registration binding"):
        v.validate_claims(changed, root, a, b)
    assert len(calls) == 1

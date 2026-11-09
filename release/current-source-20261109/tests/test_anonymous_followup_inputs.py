"""Exercise the real input gates using shipped metadata; never construct a model."""
import copy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest
import torch

from scripts import model_followup_support as support
from tests.test_model_followup import fixture_spec


ROOT = Path(__file__).resolve().parents[1]
ANONYMOUS_SELECTOR = "cvpr2027-anonymous-v1"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def forbid_model_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Input-contract regression attempted model/state/GPU execution")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    for name in ("_lazy_init", "set_device", "get_device_properties", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture(scope="module")
def anonymous_metadata():
    # Read actual distributed bytes, not the constants being tested. No extraction
    # or image/model access is needed for the five accepted metadata identities.
    with zipfile.ZipFile(ROOT / "paper/cvpr2027/scientific-supplement.zip") as archive:
        return {
            "catalog": archive.read("data/catalog.json"),
            "splits": [archive.read(f"data/splits/titan_contiguous_fold_{fold}.json") for fold in range(5)],
        }


def _anonymous_inputs(metadata, fold):
    return {
        "binding_contract": ANONYMOUS_SELECTOR,
        "catalog": {"sha256": _sha(metadata["catalog"])},
        "split": {"sha256": _sha(metadata["splits"][fold])},
    }


def _historical_inputs(fold):
    path = ROOT / f"audit/gpu_execution/repaired_splits/titan_contiguous_fold_{fold}.json"
    data = path.read_bytes()
    return {
        "catalog": {"sha256": json.loads(data)["catalog_sha256"]},
        "split": {"sha256": _sha(data)},
    }


@pytest.mark.parametrize("fold", range(5))
def test_all_shipped_anonymous_folds_require_explicit_selector(anonymous_metadata, fold):
    inputs = _anonymous_inputs(anonymous_metadata, fold)
    before = copy.deepcopy(inputs)
    support.require_followup_inputs(inputs, fold)
    assert inputs == before
    # The original gate must still reject the anonymous catalog, even when a
    # selector is supplied to a caller that has not opted into the new gate.
    with pytest.raises(ValueError, match="current Titan catalog"):
        support.require_current_inputs(inputs, fold)
    del inputs["binding_contract"]
    with pytest.raises(ValueError, match="current Titan catalog"):
        support.require_followup_inputs(inputs, fold)
    inputs["binding_contract"] = None
    with pytest.raises(ValueError, match="current Titan catalog"):
        support.require_followup_inputs(inputs, fold)


@pytest.mark.parametrize("fold", range(5))
def test_default_retains_checked_in_historical_contract(fold):
    inputs = _historical_inputs(fold)
    support.require_current_inputs(inputs, fold)
    support.require_followup_inputs(inputs, fold)
    support.require_followup_inputs({**inputs, "binding_contract": None}, fold)
    with pytest.raises(ValueError, match="anonymous Titan catalog"):
        support.require_followup_inputs({**inputs, "binding_contract": ANONYMOUS_SELECTOR}, fold)


@pytest.mark.parametrize("selector", ["", "cvpr2027-anonymous-v2", "historical"])
def test_unknown_selector_cannot_fall_back_to_historical_inputs(selector):
    inputs = {**_historical_inputs(0), "binding_contract": selector}
    with pytest.raises(ValueError, match="Unknown follow-up input contract"):
        support.require_followup_inputs(inputs, 0)


@pytest.mark.parametrize("fold", [-1, 5, True, 0.0, "0"])
def test_anonymous_fold_requires_in_range_integer(anonymous_metadata, fold):
    with pytest.raises(ValueError, match="Invalid anonymous fold"):
        support.require_followup_inputs(_anonymous_inputs(anonymous_metadata, 0), fold)


@pytest.mark.parametrize("fold", range(5))
def test_valid_split_from_another_anonymous_fold_is_rejected(anonymous_metadata, fold):
    inputs = _anonymous_inputs(anonymous_metadata, fold)
    inputs["split"]["sha256"] = _sha(anonymous_metadata["splits"][(fold + 1) % 5])
    with pytest.raises(ValueError, match="anonymous split for this fold"):
        support.require_followup_inputs(inputs, fold)


@pytest.mark.parametrize("anonymous_catalog", [False, True])
def test_catalog_and_split_cannot_mix_contracts(anonymous_metadata, anonymous_catalog):
    anonymous = _anonymous_inputs(anonymous_metadata, 0)
    historical = _historical_inputs(0)
    inputs = anonymous if anonymous_catalog else historical
    inputs["split"] = historical["split"] if anonymous_catalog else anonymous["split"]
    with pytest.raises(ValueError, match="split for this fold"):
        support.require_followup_inputs(inputs, 0)


@pytest.mark.parametrize("member", ["catalog", "split"])
@pytest.mark.parametrize("refresh_hash", [False, True], ids=["stale-declared-hash", "changed-declared-hash"])
def test_changed_metadata_bytes_rejected_before_image_access(anonymous_metadata, tmp_path, member, refresh_hash):
    inputs = _anonymous_inputs(anonymous_metadata, 0)
    for name, data in (("catalog", anonymous_metadata["catalog"]), ("split", anonymous_metadata["splits"][0])):
        path = tmp_path / f"{name}.json"
        path.write_bytes(data)
        inputs[name]["path"] = str(path)
    # Even a whitespace-only byte change must not silently replace pinned input.
    changed = Path(inputs[member]["path"])
    changed.write_bytes(changed.read_bytes() + b"\n")
    if refresh_hash:
        inputs[member]["sha256"] = _sha(changed.read_bytes())
        with pytest.raises(ValueError, match=f"accepted anonymous .*{member}"):
            support.require_followup_inputs(inputs, 0)
    else:
        support.require_followup_inputs(inputs, 0)  # The gate checks declared identities.
        with pytest.raises(ValueError, match="File hash differs"):
            support.validate_inputs({"purpose": "titan_followup", "fold": 0, "inputs": inputs})
    assert not (tmp_path / "tiles").exists()


def test_optional_selector_survives_validated_input_identity(tmp_path):
    # Reuse the existing 18-image synthetic input fixture, without preparing or
    # running a model. This checks propagation, not scientific input admission;
    # the real anonymous contract is exercised above using distributed metadata.
    spec = fixture_spec(tmp_path)
    original = support.validate_inputs(spec)
    assert "binding_contract" not in original
    spec["inputs"]["binding_contract"] = ANONYMOUS_SELECTOR
    actual = support.validate_inputs(spec)
    assert actual.pop("binding_contract") == ANONYMOUS_SELECTOR
    assert actual == original

"""Synthetic public-payload and artist checks; no source catalogs or preparation."""
from copy import deepcopy
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts import dataset_geography_payload as prep
from scripts import plot_dataset_geography as plot


def raw(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    titan = []
    for fold, counts in enumerate(prep.TITAN_COUNTS):
        for label, count in enumerate(counts):
            for n in range(count):
                titan.append([f"titan_{len(titan):06}", fold*72+.25+n % 70,
                              float(-85+n % 170), label, fold])
    footprints = []
    for si, (sid, (_, _, train, val)) in enumerate(prep.EARTH_SITES.items()):
        for n in range(train+val):
            lon, lat = -150.+si*23+n*.001, -48.+si*6+n*.001
            ring = [[lon-.02, lat-.02], [lon+.02, lat-.02], [lon+.02, lat+.02],
                    [lon-.02, lat+.02], [lon-.02, lat-.02]]
            geometry = dict(type="Polygon", coordinates=[ring])
            if si == n == 0:
                hole = [[lon-.01, lat-.01], [lon+.01, lat-.01], [lon+.01, lat+.01], [lon-.01, lat-.01]]
                second = [[x+.1, y+.1] for x, y in ring]
                geometry = dict(type="MultiPolygon", coordinates=[[ring, hole], [second]])
            footprints.append([f"earth_{sid}_{n:06}", sid, "train" if n < train else "val", [lon, lat], geometry])
    sites = [dict(site_id=s, name=n, class_index=c, train_count=t, val_count=v)
             for s, (c, n, t, v) in sorted(prep.EARTH_SITES.items(), key=lambda item: (item[1][0], item[0]))]
    venus = [[f"venus_{i:06}", .125+n % 359, float(-70+n % 140), .8 if n < 419 else 1.]
             for n, i in enumerate([*range(2943), 7094, 8759])]
    payload = dict(schema_version=1, class_names=prep.CLASSES,
                   input_sha256={name: "0"*64 for name in prep.INPUT_NAMES | {"titan_catalog"}},
                   limitations=prep.LIMITATIONS,
                   titan=dict(columns=prep.TITAN_COLUMNS, points=titan,
                              sector_boundaries_east_deg=prep.BOUNDARIES,
                              counts_by_fold_and_class=prep.TITAN_COUNTS,
                              test_union_sorted_ids_sha256=prep.digest(json.dumps([r[0] for r in titan]).encode()),
                              catalog_count=23380, excluded_catalog_count=134),
                   earth=dict(columns=prep.EARTH_COLUMNS, footprints=sorted(footprints), sites=sites,
                              absent_site=dict(site_id="haughton", count=0), absent_class="hummocky",
                              geometry_note="Retained nominal source-window outlines and geometric centers; no new geometry or SAR validity masking."),
                   venus=dict(columns=prep.VENUS_COLUMNS, points=venus,
                              examples=[dict(code=chr(65+i), tile_id=tid) for i, tid in enumerate(prep.VENUS_EXAMPLES)],
                              complete_count=2526, partial_count=419))
    root = tmp_path_factory.mktemp("synthetic-geography")
    path = root / "dataset-geography.json"
    path.write_bytes(raw(payload))
    return dict(root=root, payload=payload, path=path)


def test_public_payload_keeps_complete_synthetic_populations(prepared):
    payload = prep.validate_payload(prepared["payload"])
    assert (len(payload["titan"]["points"]), len(payload["earth"]["footprints"]),
            len(payload["venus"]["points"])) == (23246, 398, 2945)
    assert not {"prepare", "prepare_titan", "prepare_earth", "prepare_venus", "decode_carrier", "main"} & set(vars(prep))


def test_bad_hash_precedes_all_decoding(tmp_path, monkeypatch):
    path = tmp_path / "bad"; path.write_bytes(b"not JSON")
    monkeypatch.setattr(prep, "decode", lambda raw: pytest.fail("decoded unauthenticated bytes"))
    with pytest.raises(ValueError, match="SHA256"):
        plot.render(path, "0"*64, tmp_path / "output", tmp_path / "receipt", "cvpr")
    assert not (tmp_path / "output").exists()


def test_payload_byte_bound_precedes_decode(tmp_path, monkeypatch):
    path = tmp_path / "large"; path.write_bytes(b"x"*33)
    monkeypatch.setattr(prep, "MAX_INPUT_BYTES", 32)
    monkeypatch.setattr(prep, "decode", lambda raw: pytest.fail("oversized payload decoded"))
    with pytest.raises(ValueError, match="byte limit"):
        plot.render(path, prep.digest(path.read_bytes()), tmp_path / "output", tmp_path / "receipt", "cvpr")


@pytest.mark.parametrize("fault", ["extra", "missing", "duplicate", "boolean_label", "region", "union", "site", "role", "ring", "fraction", "example"])
def test_public_payload_corruption_rejected(prepared, fault):
    payload = deepcopy(prepared["payload"])
    if fault == "extra": payload["execution"] = {}
    elif fault == "missing": payload["titan"]["points"].pop()
    elif fault == "duplicate": payload["titan"]["points"][1][0] = payload["titan"]["points"][0][0]
    elif fault == "boolean_label": payload["titan"]["points"][0][3] = False
    elif fault == "region": payload["titan"]["points"][0][4] = 1
    elif fault == "union": payload["titan"]["test_union_sorted_ids_sha256"] = "0"*64
    elif fault == "site": payload["earth"]["footprints"][0][1] = "unknown"
    elif fault == "role": payload["earth"]["footprints"][0][2] = "test"
    elif fault == "ring": payload["earth"]["footprints"][0][4] = dict(type="Polygon", coordinates=[[[0, 0], [1, 1], [2, 2]]])
    elif fault == "fraction": payload["venus"]["points"][0][3] = True
    else: payload["venus"]["examples"][0]["code"] = "terrain"
    with pytest.raises(ValueError):
        prep.validate_payload(payload)


@pytest.mark.parametrize("style", ["cvpr", "full"])
def test_frozen_earth_card_height(prepared, style):
    figures = plot.make_figures(prepared["payload"], style)
    try:
        np.testing.assert_array_equal(figures["earth-site-footprints"].get_size_inches(), [6.5, 8.1])
    finally:
        for figure in figures.values(): plt.close(figure)


def test_bound_read_is_single_buffer(monkeypatch):
    calls = []
    def changed(path, *args):
        calls.append(str(path))
        return io.BytesIO(b'{"accepted":true}' if len(calls) == 1 else b'{"replacement":true}')
    monkeypatch.setattr(Path, "open", changed)
    value = prep.decode(prep.read_bound(dict(path="synthetic", sha256=prep.digest(b'{"accepted":true}'))))
    assert value == {"accepted": True} and calls == ["synthetic"]


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e400}'])
def test_invalid_json(text):
    with pytest.raises(ValueError):
        prep.decode(text)


@pytest.mark.parametrize("fault", ["coordinates", "id", "missing", "ring", "role", "panel", "table", "example_label", "completeness"])
def test_actual_artist_mutations_rejected(prepared, fault):
    figures = plot.make_figures(prepared["payload"], "cvpr")
    try:
        report = plot.audit_artists(figures, prepared["payload"])
        assert report["counts"] == dict(titan=23246, earth_centers=398, earth_ring=400, venus=2945, venus_example=6)
        scatter = figures["titan-class-locations"].axes[0].collections[0]
        if fault == "coordinates":
            values = scatter.get_offsets().copy(); values[0, 0] += .001; scatter.set_offsets(values)
        elif fault == "id": scatter._geography["ids"][0] = "titan_unknown"
        elif fault == "missing": scatter.remove()
        elif fault in ("ring", "role"):
            line = figures["earth-site-footprints"].axes[0].lines[0]
            if fault == "ring":
                x, y = line.get_data(); x = x.copy(); x[1] += .0001; line.set_data(x, y)
            else: line.set_linestyle(":" )
        elif fault == "panel": figures["titan-class-locations"].axes[0].set_title("Wrong class")
        elif fault == "table": figures["earth-site-locator"].axes[1].tables[0].get_celld()[1, 3].get_text().set_text("999")
        elif fault == "example_label": figures["venus-sample-centers"].axes[0].texts[0].set_text("Wrong")
        else: figures["venus-sample-centers"].axes[0].collections[0].set_facecolor("red")
        with pytest.raises((ValueError, KeyError)): plot.audit_artists(figures, prepared["payload"])
    finally:
        for figure in figures.values(): plt.close(figure)


@pytest.mark.parametrize("fault", ["hash", "oversized"])
def test_context_authentication_precedes_decoding(tmp_path, monkeypatch, fault):
    (tmp_path / "natural-earth-110m-coastline.json").write_bytes(
        b"not JSON" if fault == "hash" else b"x" * 512_001)
    monkeypatch.setattr(plot, "CONTEXT_DIR", tmp_path)
    monkeypatch.setattr(prep, "decode", lambda raw: pytest.fail("decoded unbound context"))
    with pytest.raises(ValueError, match="SHA256|byte limit"):
        plot.coastline_context()


def test_context_retains_source_parts_without_dateline_chords():
    geometry, provenance = plot.coastline_context()
    parts = geometry["parts"]
    assert [(p["source_record"], p["source_part"]) for p in parts] == [(i, 0) for i in range(1, 135)]
    assert sum(len(p["coordinates"]) for p in parts) == 5128
    assert max(c[0] for p in parts for c in p["coordinates"]) == 180.00000044181039
    assert max(abs(a[0]-b[0]) for p in parts for a, b in zip(p["coordinates"], p["coordinates"][1:])) < 10
    assert provenance["version"] == "4.1.0" and provenance["license"]["name"] == "Public domain"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "vertices", "order", "linewidth", "color",
                                   "fill", "limits", "transform", "binding", "wrong_axes",
                                   "offsets", "offset_transform", "clip_box", "clip_path", "clip_box_none"])
def test_coastline_context_mutations_rejected(prepared, fault):
    figure = plot.earth_locator(prepared["payload"])
    figures = {"earth-site-locator": figure}
    try:
        assert plot.audit_context(figures)["vertices"] == 5128
        ax = figure.axes[0]; coastline = ax.collections[0]
        if fault == "missing": coastline.remove()
        elif fault == "duplicate":
            from copy import copy
            duplicate = copy(coastline); ax.add_collection(duplicate, autolim=False)
        elif fault in ("vertices", "order"):
            parts = [p.copy() for p in coastline.get_segments()]
            if fault == "vertices": parts[0][0, 0] += .001
            else: parts[0], parts[1] = parts[1], parts[0]
            coastline.set_segments(parts)
        elif fault == "linewidth": coastline.set_linewidth(.8)
        elif fault == "color": coastline.set_color("black")
        elif fault == "fill": coastline.set_facecolor("gray")
        elif fault == "limits": ax.set_xlim(-181, 181)
        elif fault == "transform": coastline.set_transform(ax.transAxes)
        elif fault == "binding": coastline._geography_context["data_sha256"] = "0" * 64
        elif fault == "offsets": coastline.set_offsets([[10, 0]])
        elif fault == "offset_transform":
            from matplotlib.transforms import Affine2D
            coastline.set_offset_transform(Affine2D().translate(10, 0))
        elif fault == "clip_box":
            from matplotlib.transforms import Bbox
            coastline.set_clip_box(Bbox.from_bounds(0, 0, 10, 10))
        elif fault == "clip_path":
            from matplotlib.patches import Circle
            coastline.set_clip_path(Circle((.5, .5), .1, transform=ax.transAxes))
        elif fault == "clip_box_none": coastline.set_clip_box(None)
        else:
            coastline.remove(); figure.axes[1].add_collection(coastline, autolim=False)
        with pytest.raises(ValueError, match="[Cc]oastline"):
            plot.audit_context(figures)
    finally:
        plt.close(figure)


def test_context_does_not_change_scientific_artist_digest_or_admit_untracked_lines(prepared, monkeypatch):
    figures = plot.make_figures(prepared["payload"], "cvpr")
    try:
        with_context = plot.audit_artists(figures, prepared["payload"])
        ax = figures["earth-site-locator"].axes[0]
        coastline = ax.collections[0]; coastline.remove()
        with monkeypatch.context() as patch:
            patch.setattr(plot, "audit_context", lambda figures: None)
            assert plot.audit_artists(figures, prepared["payload"]) == with_context
        ax.add_collection(coastline, autolim=False)
        ax.plot([-170, 170], [0, 0], color="gray")
        with pytest.raises(ValueError, match="Untracked data artist"):
            plot.audit_artists(figures, prepared["payload"])
    finally:
        for figure in figures.values(): plt.close(figure)


@pytest.mark.parametrize("style", ["cvpr", "full"])
def test_earth_locator_nearby_site_callouts_are_separate(style):
    from matplotlib import rc_context
    from matplotlib.text import Text

    sites = [dict(site_id=sid, name=name, class_index=label, train_count=train, val_count=val)
             for sid, (label, name, train, val) in sorted(prep.EARTH_SITES.items(), key=lambda s: (s[1][0], s[0]))]
    # Invented nearly coincident centers reproduce the callout collision.
    rows = [[f"synthetic_{i}", site["site_id"], "train",
             [-70+i*.0001, -20] if i in (7, 11) else [-150+i*20, -60], None]
            for i, site in enumerate(sites, 1)]
    with rc_context():
        (plot.cvpr if style == "cvpr" else plot.full).configure()
        figure = plot.earth_locator(dict(earth=dict(sites=sites, footprints=rows)))
        try:
            figure.set_dpi(120)
            if style == "cvpr": plot.cvpr.prepare_figure(figure, "earth-site-locator")
            else: figure.canvas.draw()
            annotations = {a.get_text(): a for a in figure.axes[0].texts}
            for label in ("7", "11"):
                assert list(annotations[label].xy) == rows[int(label)-1][3]
                assert annotations[label].arrow_patch is not None
            a, b = [Text.get_window_extent(annotations[label], figure.canvas.get_renderer()) for label in ("7", "11")]
            gap = max(a.x0-b.x1, b.x0-a.x1, a.y0-b.y1, b.y0-a.y1) * 72/figure.dpi
            assert gap >= 4, "Nearby site callouts need at least four points of separation"
        finally:
            plt.close(figure)


@pytest.mark.parametrize("style", ["cvpr", "full"])
def test_synthetic_complete_exports_and_freshness(prepared, tmp_path, style):
    output = tmp_path / style; receipt = tmp_path / "private.json"
    result = plot.render(prepared["path"], prep.digest(prepared["path"].read_bytes()), output, receipt, style)
    assert result["status"] == "complete" and len(result["outputs"]) == 8
    assert result["context"]["data_sha256"] == plot.COASTLINE_SHA256
    assert result["context"]["parts"] == 134 and result["context"]["vertices"] == 5128
    assert json.loads(receipt.read_text())["context"] == result["context"]
    assert set(p.name for p in output.iterdir()) == set(result["outputs"]) | {"manifest.json"}
    assert all(p.stat().st_size > 0 for p in output.iterdir())
    with pytest.raises(ValueError, match="fresh"):
        plot.render(prepared["path"], "0"*64, output, receipt, style)


def test_failed_export_has_no_completion(prepared, tmp_path, monkeypatch):
    original = prep.write_bytes_new
    def fail(path, content):
        if Path(path).name.endswith(".png"): raise OSError("synthetic publication failure")
        original(path, content)
    monkeypatch.setattr(prep, "write_bytes_new", fail)
    output = tmp_path / "failed"; receipt = tmp_path / "receipt.json"
    with pytest.raises(OSError, match="publication failure"):
        plot.render(prepared["path"], prep.digest(prepared["path"].read_bytes()), output, receipt, "cvpr")
    assert not receipt.exists() and not (output / "manifest.json").exists()


def test_failed_final_manifest_does_not_leave_complete_receipt(prepared, tmp_path, monkeypatch):
    original = prep.write_new_atomic
    def fail(path, record):
        if Path(path).name == "manifest.json": raise OSError("synthetic manifest failure")
        original(path, record)
    monkeypatch.setattr(prep, "write_new_atomic", fail)
    output = tmp_path / "failed"; receipt = tmp_path / "receipt.json"
    with pytest.raises(OSError, match="manifest failure"):
        plot.render(prepared["path"], prep.digest(prepared["path"].read_bytes()), output, receipt, "cvpr")
    assert json.loads(receipt.read_text())["status"] == "verified_before_public_manifest"
    assert not (output / "manifest.json").exists()

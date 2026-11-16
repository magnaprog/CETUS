"""Plot the complete sanitized geography payload without reading source science.

Artist checks inspect actual offsets and ring vertices before export. Local
Earth axes show degrees with varying extents, not equal-area or physical scale.
"""

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
import textwrap

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

from scripts import cvpr_figure_style as cvpr
from scripts import figure_style as full
from scripts import dataset_geography_payload as data

STEMS = ("titan-class-locations", "earth-site-locator", "earth-site-footprints", "venus-sample-centers")
ROLE_STYLE = {"train": ("#0077BB", "-"), "val": ("#EE7733", "--")}


def bind(artist, kind, ids, **details):
    artist._geography = dict(kind=kind, ids=list(ids), **details)
    artist.set_gid(kind)
    return artist


def global_axes(ax, *, earth=False):
    ax.set(xlim=(-180, 180) if earth else (0, 360), ylim=(-90, 90),
           xlabel="Longitude [degrees]" if earth else "East longitude [degrees]",
           ylabel="Latitude [degrees]", yticks=[-90, -45, 0, 45, 90])
    ax.set_xticks([-180, -90, 0, 90, 180] if earth else data.BOUNDARIES)
    ax.grid(color=".87", linewidth=.4)


def titan_figure(payload):
    figure, axes = plt.subplots(3, 2, figsize=(6.5, 6.5))
    figure.subplots_adjust(left=.115, right=.97, bottom=.13, top=.94, wspace=.38, hspace=.62)
    for c, ax in enumerate(axes.flat):
        ax._geography_panel = ("titan", c)
        rows = [r for r in payload["titan"]["points"] if r[3] == c]
        global_axes(ax)
        ax.grid(False)
        for boundary in data.BOUNDARIES[1:-1]:
            ax.axvline(boundary, color=".7", linewidth=.5, zorder=0)
        bind(ax.scatter([r[1] for r in rows], [r[2] for r in rows], s=2,
                        color=full.CLASS_COLORS[c], linewidths=0, rasterized=True),
             "titan", [r[0] for r in rows], class_index=c)
        ax.set_title(data.CLASSES[c].capitalize())
    figure.text(.5, .035, "Test tile centers; regions 1–5 run west to east.\n"
                 "Neutral lines mark 72-degree sector boundaries; points are not surface coverage.", ha="center")
    return figure


def earth_locator(payload):
    figure = plt.figure(figsize=(6.5, 7.6))
    ax = figure.add_axes([.11, .62, .85, .32])
    global_axes(ax, earth=True)
    rows = payload["earth"]["footprints"]
    bind(ax.scatter([r[3][0] for r in rows], [r[3][1] for r in rows], s=5, color=".35", linewidths=0),
         "earth_centers", [r[0] for r in rows])
    for i, site in enumerate(payload["earth"]["sites"], 1):
        # A retained actual center locates the site; no invented centroid.
        center = next(r[3] for r in rows if r[1] == site["site_id"])
        # Separate the nearby Atacama and Salar de Uyuni callouts.
        offset = {"atacama_channels": (-12, -10), "salar_de_uyuni": (8, 6)}.get(
            site["site_id"], ((8 if i % 2 else -12), (6 if i % 2 else -10)))
        ax.annotate(str(i), center, xytext=offset,
                    textcoords="offset points", arrowprops=dict(arrowstyle="-", color=".45", lw=.5))
    ax.set_title("Earth sample locations")
    table_axes = figure.add_axes([.04, .10, .92, .44]); table_axes.set_axis_off()
    cells = [[str(i), s["name"], data.CLASSES[s["class_index"]].capitalize(),
              str(s["train_count"]), str(s["val_count"])]
             for i, s in enumerate(payload["earth"]["sites"], 1)]
    table = table_axes.table(cellText=cells, colLabels=["Site", "Catalog site assignment", "Class", "Train", "Val"],
                             colWidths=[.07, .48, .19, .13, .13], cellLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(.3); cell.set_edgecolor(".8")
        if row == 0:
            cell.set_facecolor(".94")
    figure.text(.5, .035, "398 images: 352 train / 46 validation. Haughton: 0 images; Hummocky absent.\n"
                 "Intended site labels are unverified within footprints; Egyptian images remain in this population.", ha="center")
    return figure


def earth_footprints(payload):
    # Leave room for a manuscript caption without scaling the nine-point callouts.
    figure, axes = plt.subplots(5, 3, figsize=(6.5, 8.1))
    figure.subplots_adjust(left=.10, right=.98, bottom=.12, top=.94, wspace=.62, hspace=1.02)
    for i, (ax, site) in enumerate(zip(axes.flat, payload["earth"]["sites"]), 1):
        ax._geography_panel = ("earth", site["site_id"])
        rows = [r for r in payload["earth"]["footprints"] if r[1] == site["site_id"]]
        coordinates = []
        for tid, sid, role, center, geometry in sorted(rows, key=lambda r: (r[2] == "val", r[0])):
            color, linestyle = ROLE_STYLE[role]
            for pi, ri, ring in data.geometry_rings(geometry):
                artist, = ax.plot([p[0] for p in ring], [p[1] for p in ring], color=color,
                                   linestyle=linestyle, linewidth=.65)
                bind(artist, "earth_ring", [tid], site_id=sid, role=role, polygon=pi, ring=ri)
                coordinates.extend(ring)
        values = np.asarray(coordinates)
        low, high = values.min(axis=0), values.max(axis=0)
        padding = np.maximum((high-low)*.08, .001)
        ax.set(xlim=(low[0]-padding[0], high[0]+padding[0]), ylim=(low[1]-padding[1], high[1]+padding[1]))
        # Two interior labels keep long signed coordinates legible in narrow cards.
        ax.set_xticks([low[0] + (high[0]-low[0])*fraction for fraction in (.25, .75)])
        ax.set_yticks([low[1] + (high[1]-low[1])*fraction for fraction in (.25, .75)])
        # Tick text is a rounded coordinate guide; stored artist vertices stay exact.
        formatter = FuncFormatter(lambda value, _: f"{value:.3f}".rstrip("0").rstrip("."))
        ax.xaxis.set_major_formatter(formatter); ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(pad=1, length=2)
        ax.grid(color=".9", linewidth=.4)
        ax.set_title(f"{i}. " + textwrap.fill(site["name"], width=20), pad=4)
    for ax in list(axes.flat)[13:]:
        ax.set_axis_off()
    handles = [Line2D([], [], color=color, linestyle=style, label=role.capitalize())
               for role, (color, style) in ROLE_STYLE.items()]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .055), ncol=2, frameon=False)
    figure.text(.5, .018, "Axes: longitude/latitude in degrees. Local extents vary; areas are not comparable.\n"
                 "All 398 nominal outlines are shown; terrain classes within these footprints remain unverified.", ha="center")
    return figure


def venus_figure(payload):
    figure, ax = plt.subplots(figsize=(6.5, 4.2))
    figure.subplots_adjust(left=.10, right=.97, bottom=.27, top=.90)
    global_axes(ax)
    for complete in (True, False):
        rows = [r for r in payload["venus"]["points"] if (r[3] == 1) == complete]
        bind(ax.scatter([r[1] for r in rows], [r[2] for r in rows], s=7 if complete else 13,
                        facecolors=".55" if complete else "none", edgecolors="none" if complete else ".20",
                        linewidths=0 if complete else .5,
                        label="Complete pixels" if complete else "Contains missing pixels", rasterized=True),
             "venus", [r[0] for r in rows], complete=complete)
    by_id = {r[0]: r for r in payload["venus"]["points"]}
    for ex in payload["venus"]["examples"]:
        row = by_id[ex["tile_id"]]
        bind(ax.scatter([row[1]], [row[2]], s=33, marker="s", facecolors="none", edgecolors="black", linewidths=.8),
             "venus_example", [row[0]], code=ex["code"])
        ax.annotate(ex["code"], (row[1], row[2]), xytext=(4, 5), textcoords="offset points")
    ax.set_title("All 2,945 retained Venus sample centers")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
    figure.text(.5, .035, "A–F: the six displayed unlabeled examples. Symbols describe pixel completeness.\n"
                 "These centers do not measure surface-area or acquisition coverage.", ha="center")
    return figure


def audit_artists(figures, payload):
    """Read every plotted data artist, including all overlapping footprint rings."""
    data.require(set(figures) == set(STEMS), "Figure inventory differs")
    titan = {r[0]: r for r in payload["titan"]["points"]}
    earth = {r[0]: r for r in payload["earth"]["footprints"]}
    venus = {r[0]: r for r in payload["venus"]["points"]}
    expected_rings = {(tid, pi, ri): ring for tid, row in earth.items()
                      for pi, ri, ring in data.geometry_rings(row[4])}
    seen = {kind: [] for kind in ("titan", "earth_centers", "earth_ring", "venus", "venus_example")}
    records = []
    for stem, figure in figures.items():
        for ax in figure.axes:
            for artist in [*ax.collections, *ax.lines]:
                binding = getattr(artist, "_geography", None)
                if binding is None:
                    # Only the four explicit Titan sector lines are non-data lines.
                    data.require(stem == "titan-class-locations" and isinstance(artist, Line2D)
                                 and list(artist.get_xdata()) in [[x, x] for x in data.BOUNDARIES[1:-1]],
                                 "Untracked data artist")
                    continue
                kind, ids = binding["kind"], binding["ids"]
                data.require(kind in seen and ids and len(ids) == len(set(ids)), "Invalid artist ID inventory")
                if kind == "earth_ring":
                    data.require(stem == "earth-site-footprints" and len(ids) == 1, "Wrong footprint artist")
                    key = (ids[0], binding["polygon"], binding["ring"])
                    expected = expected_rings[key]; row = earth[ids[0]]
                    data.require(binding["site_id"] == row[1] and binding["role"] == row[2]
                                 and ax._geography_panel == ("earth", row[1])
                                 and artist.get_linestyle() == ROLE_STYLE[row[2]][1]
                                 and artist.get_color() == ROLE_STYLE[row[2]][0], "Footprint role/style differs")
                    actual = np.column_stack(artist.get_data()).tolist(); seen[kind].append(key)
                else:
                    source = titan if kind == "titan" else earth if kind == "earth_centers" else venus
                    data.require(all(tid in source for tid in ids), "Unknown artist ID")
                    expected = [source[tid][3] if kind == "earth_centers" else source[tid][1:3] for tid in ids]
                    actual = np.asarray(artist.get_offsets()).tolist(); seen[kind].extend(ids)
                    if kind == "titan":
                        data.require(stem == "titan-class-locations" and all(titan[tid][3] == binding["class_index"] for tid in ids),
                                     "Titan class artist differs")
                        data.require(ax._geography_panel == ("titan", binding["class_index"])
                                     and ax.get_title() == data.CLASSES[binding["class_index"]].capitalize()
                                     and tuple(ax.get_xlim()) == (0, 360) and tuple(ax.get_ylim()) == (-90, 90),
                                     "Titan panel differs")
                    elif kind == "earth_centers":
                        data.require(stem == "earth-site-locator", "Wrong Earth locator")
                    elif kind == "venus":
                        data.require(stem == "venus-sample-centers" and all((venus[tid][3] == 1) == binding["complete"] for tid in ids),
                                     "Venus completeness artist differs")
                        faces, edges = artist.get_facecolors().tolist(), artist.get_edgecolors().tolist()
                        data.require((faces == [list(to_rgba(".55"))] and not edges) if binding["complete"]
                                     else (not faces and edges == [list(to_rgba(".20"))]), "Venus completeness encoding differs")
                    else:
                        data.require(stem == "venus-sample-centers" and len(ids) == 1
                                     and {"code": binding["code"], "tile_id": ids[0]} in payload["venus"]["examples"],
                                     "Venus example artist differs")
                data.require(actual == expected, "Actual artist coordinates differ")
                xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
                data.require(all(xmin <= x <= xmax and ymin <= y <= ymax for x, y in actual), "Data artist clipped by axes")
                records.append(dict(figure=stem, binding=binding, coordinates=actual))
    expected_ids = dict(titan=set(titan), earth_centers=set(earth), earth_ring=set(expected_rings),
                        venus=set(venus), venus_example=set(data.VENUS_EXAMPLES))
    for kind, ids in seen.items():
        data.require(len(ids) == len(expected_ids[kind]) and set(ids) == expected_ids[kind], "Incomplete artist population")
    sites = payload["earth"]["sites"]
    locator = figures["earth-site-locator"]
    expected_cells = [["Site", "Catalog site assignment", "Class", "Train", "Val"]] + [
        [str(i), s["name"], data.CLASSES[s["class_index"]].capitalize(), str(s["train_count"]), str(s["val_count"])]
        for i, s in enumerate(sites, 1)]
    data.require(len(locator.axes[1].tables) == 1, "Site table inventory differs")
    cells = locator.axes[1].tables[0].get_celld()
    data.require(set(cells) == {(r, c) for r in range(14) for c in range(5)}
                 and all(cells[r, c].get_text().get_text() == text for r, row in enumerate(expected_cells)
                         for c, text in enumerate(row)), "Actual site table differs")
    annotations = locator.axes[0].texts
    data.require(len(annotations) == 13, "Earth locator labels differ")
    for i, (site, annotation) in enumerate(zip(sites, annotations), 1):
        center = next(r[3] for r in payload["earth"]["footprints"] if r[1] == site["site_id"])
        data.require(annotation.get_text() == str(i) and list(annotation.xy) == center, "Earth locator annotation differs")
        ax = figures["earth-site-footprints"].axes[i-1]
        data.require(ax.get_title() == f"{i}. " + textwrap.fill(site["name"], width=20), "Earth card title differs")
    annotations = figures["venus-sample-centers"].axes[0].texts
    data.require(len(annotations) == 6 and all(a.get_text() == ex["code"]
                 and list(a.xy) == venus[ex["tile_id"]][1:3] for a, ex in zip(annotations, payload["venus"]["examples"])),
                 "Venus A-F annotation differs")
    return dict(counts={kind: len(ids) for kind, ids in seen.items()},
                actual_artist_sha256=data.digest(json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()))


def make_figures(payload, style):
    data.validate_payload(payload)
    data.require(style in ("cvpr", "full"), "Unknown figure style")
    (cvpr if style == "cvpr" else full).configure()
    return dict(zip(STEMS, [titan_figure(payload), earth_locator(payload), earth_footprints(payload), venus_figure(payload)]))


def render(payload_path, payload_sha256, output, receipt, style):
    output, receipt = Path(output), Path(receipt)
    data.require(not output.exists() and not receipt.exists(), "Output/receipt must be fresh")
    data.require(output.resolve() not in receipt.resolve().parents, "Receipt must be outside public output")
    raw = data.read_bound(dict(path=str(payload_path), sha256=payload_sha256))
    payload = data.validate_payload(data.decode(raw))
    expected = {stem + suffix for stem in STEMS for suffix in (".pdf", ".png")}
    data.require(expected and len(expected) == 8, "Empty/incomplete export inventory")
    figures = make_figures(payload, style)
    try:
        metrics, fonts = {}, set()
        for name, figure in figures.items():
            if style == "cvpr":
                metrics[name] = cvpr.prepare_figure(figure, name)
            else:
                figure.canvas.draw()
                metrics[name] = cvpr.text_metrics(figure)
            for text in figure.findobj(Text):
                if text.get_visible() and text.get_text():
                    fonts.add(font_manager.findfont(text.get_fontproperties()))
            data.require(not metrics[name]["clipped_text"], f"Clipped text in {name}: {metrics[name]['clipped_text']}")
            if name == "earth-site-footprints":
                renderer = figure.canvas.get_renderer()
                for ax in figure.axes[:13]:
                    boxes = sorted((t.get_window_extent(renderer) for t in ax.get_xticklabels()
                                    if t.get_visible() and t.get_text()), key=lambda b: b.x0)
                    data.require(all(left.x1 < right.x0 for left, right in zip(boxes, boxes[1:])),
                                 "Overlapping Earth longitude tick labels")
        font_root = (cvpr if style == "cvpr" else full).FONT_DIR.resolve()
        data.require(fonts and all(Path(p).resolve().parent == font_root for p in fonts), "Unexpected/unbound font")
        artists = audit_artists(figures, payload)
        output.mkdir()
        outputs = {}
        for name, figure in figures.items():
            for suffix in (".pdf", ".png"):
                buffer = io.BytesIO()
                kwargs = dict(metadata={"CreationDate": None, "ModDate": None}) if suffix == ".pdf" else dict(dpi=120)
                figure.savefig(buffer, format=suffix[1:], **kwargs)
                content = buffer.getvalue()
                data.require(0 < len(content) <= 20_000_000, "Invalid/big figure export")
                data.require(sum(r["bytes"] for r in outputs.values()) + len(content) <= 100_000_000,
                             "Figure inventory exceeds 100 MB")
                data.write_bytes_new(output / (name + suffix), content)
                outputs[name + suffix] = dict(sha256=data.digest(content), bytes=len(content))
        data.require(set(outputs) == expected and {p.name for p in output.iterdir()} == expected
                     and all((output / n).stat().st_size == r["bytes"] for n, r in outputs.items()), "Export inventory differs")
        record = dict(status="complete", style=style, payload_sha256=payload_sha256, outputs=outputs, artists=artists)
        data.write_new_atomic(receipt, {**record, "status": "verified_before_public_manifest", "font_metrics": metrics,
                                        "fonts_sha256": {p: data.digest(Path(p).read_bytes()) for p in sorted(fonts)},
                                        "source_sha256": data.digest(Path(__file__).read_bytes()),
                                        "sources_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): data.digest(p.read_bytes())
                                                           for p in (Path(__file__).resolve(), Path(data.__file__).resolve(),
                                                                     Path(__file__).resolve().with_name("display_io.py"),
                                                                     Path(cvpr.__file__).resolve(), Path(full.__file__).resolve())}})
        data.write_new_atomic(output / "manifest.json", record)
        return record
    finally:
        for figure in figures.values():
            plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--payload-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--style", choices=("cvpr", "full"), required=True)
    args = parser.parse_args()
    render(args.payload, args.payload_sha256, args.output, args.receipt, args.style)


if __name__ == "__main__":
    main()

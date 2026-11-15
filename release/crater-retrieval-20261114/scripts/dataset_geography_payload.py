"""Read and validate the frozen display payload; no preparation entrypoint."""
from collections import Counter
import json
import math
from pathlib import Path
import re

from scripts.display_io import digest, write_new_atomic, write_bytes_new


CLASSES = ["plains", "dunes", "hummocky", "labyrinths", "lakes", "craters"]


BOUNDARIES = [0, 72, 144, 216, 288, 360]


TITAN_COUNTS = [[2486, 492, 372, 51, 251, 8], [2842, 1065, 380, 109, 100, 1],
                [3956, 1164, 403, 175, 9, 14], [2992, 349, 1142, 102, 102, 61],
                [3130, 888, 460, 61, 33, 48]]


EARTH_SITES = {
    "arctic_tundra": (0, "Arctic tundra", 77, 0),
    "icelandic_sandur": (0, "Icelandic sandur", 15, 0),
    "desert_pavement": (0, "Desert pavement", 17, 16),
    "namib_sand_sea": (1, "Namib Sand Sea", 47, 0),
    "grand_erg_oriental": (1, "Grand Erg Oriental", 14, 10),
    "rub_al_khali": (1, "Rub' al Khali", 113, 0),
    "atacama_channels": (3, "Atacama channels", 19, 0),
    "wadi_rum": (3, "Egyptian site (Wadi Rum label)", 16, 0),
    "madagascar_tsingy": (3, "Madagascar tsingy", 10, 6),
    "arctic_thermokarst": (4, "Arctic thermokarst", 16, 0),
    "salar_de_uyuni": (4, "Salar de Uyuni", 0, 12),
    "manicouagan": (5, "Manicouagan", 4, 0),
    "mistastin": (5, "Mistastin", 4, 2),
}


VENUS_EXAMPLES = ["venus_000717", "venus_000636", "venus_007094",
                  "venus_002192", "venus_002727", "venus_008759"]


TITAN_COLUMNS = ["tile_id", "east_longitude_deg", "latitude_deg", "class_index", "saved_test_fold"]


EARTH_COLUMNS = ["tile_id", "site_id", "role", "geometric_center_lonlat", "geometry"]


VENUS_COLUMNS = ["tile_id", "east_longitude_deg", "latitude_deg", "valid_fraction"]


INPUT_NAMES = {"titan_catalog_bundle", "titan_protocol", "earth_audit", "venus_catalog", "venus_examples"} | {
    f"titan_split_{f}" for f in range(5)}


MAX_INPUT_BYTES = 32 * 1024**2


LIMITATIONS = [
    "Titan and Venus marks are sample centers, not surface coverage or independent geological occurrences.",
    "Titan classes are modal map labels within the existing geographic tests.",
    "Earth outlines are retained nominal source windows, not SAR-valid masks or terrain annotations.",
    "Earth roles are the original 352 training / 46 validation images; the Egyptian site remains included.",
    "Earth Hummocky is absent; historical Haughton supplies no image.",
    "Venus is unlabeled. A-F identify existing illustrations without terrain correspondence.",
    "Longitude/latitude panels have no physical scale bar; local Earth extents differ and are not comparable areas.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, low, high):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            "Invalid numeric coordinate/count/fraction")
    return value


def integer(value, low, high):
    require(type(value) is int and low <= value <= high, "Invalid integer")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def decode(raw):
    def finite(text):
        value = float(text)
        require(math.isfinite(value), "Nonfinite JSON number")
        return value
    return json.loads(raw, object_pairs_hook=_object, parse_float=finite, parse_constant=finite)


def read_bound(binding):
    require(type(binding) is dict and set(binding) == {"path", "sha256"}, "Invalid input binding")
    require(type(binding["path"]) is str and type(binding["sha256"]) is str
            and re.fullmatch(r"[a-f0-9]{64}", binding["sha256"]), "Invalid binding path/hash")
    with Path(binding["path"]).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    require(len(raw) <= MAX_INPUT_BYTES, "Input exceeds byte limit")
    require(digest(raw) == binding["sha256"], "Input SHA256 differs")
    return raw


def lonlat(point, *, east=False):
    require(type(point) is list and len(point) == 2, "Invalid coordinate pair")
    number(point[0], 0 if east else -180, 360 if east else 180)
    number(point[1], -90, 90)
    require(not east or point[0] < 360, "East longitude must be below 360")
    return point


def geometry_rings(geometry):
    require(type(geometry) is dict and set(geometry) == {"type", "coordinates"}
            and geometry["type"] in ("Polygon", "MultiPolygon"), "Invalid stored geometry")
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    require(type(polygons) is list and polygons, "Empty polygon set")
    rings = []
    for pi, polygon in enumerate(polygons):
        require(type(polygon) is list and polygon, "Empty polygon")
        for ri, ring in enumerate(polygon):
            require(type(ring) is list and len(ring) >= 4 and ring[0] == ring[-1], "Unclosed/empty ring")
            for point in ring:
                lonlat(point)
            rings.append((pi, ri, ring))
    return rings


def validate_payload(p):
    require(type(p) is dict and set(p) == {"schema_version", "class_names", "input_sha256", "titan", "earth", "venus", "limitations"}
            and type(p["schema_version"]) is int and p["schema_version"] == 1
            and p["class_names"] == CLASSES and p["limitations"] == LIMITATIONS, "Payload schema differs")
    require(set(p["input_sha256"]) == INPUT_NAMES | {"titan_catalog"}
            and all(type(h) is str and re.fullmatch(r"[a-f0-9]{64}", h) for h in p["input_sha256"].values()),
            "Payload input bindings differ")
    expected_keys = {
        "titan": {"columns", "points", "sector_boundaries_east_deg", "counts_by_fold_and_class", "test_union_sorted_ids_sha256", "catalog_count", "excluded_catalog_count"},
        "earth": {"columns", "footprints", "sites", "absent_site", "absent_class", "geometry_note"},
        "venus": {"columns", "points", "examples", "complete_count", "partial_count"},
    }
    for domain, keys in expected_keys.items():
        require(type(p[domain]) is dict and set(p[domain]) == keys, "Unexpected public field")
    for domain, columns, count in (("titan", TITAN_COLUMNS, 23246), ("earth", EARTH_COLUMNS, 398), ("venus", VENUS_COLUMNS, 2945)):
        d = p[domain]; rows = d["footprints" if domain == "earth" else "points"]
        require(d["columns"] == columns and type(rows) is list and len(rows) == count, "Payload population differs")
        require(all(type(r) is list and len(r) == len(columns) and type(r[0]) is str
                    and re.fullmatch(domain + r"_[a-z0-9_]+", r[0]) for r in rows), "Invalid payload rows")
        require([r[0] for r in rows] == sorted({r[0] for r in rows}), "Unsorted/duplicate payload IDs")
    titan = p["titan"]; counts = [[0]*6 for _ in range(5)]
    require(type(titan["counts_by_fold_and_class"]) is list
            and len(titan["counts_by_fold_and_class"]) == 5
            and all(type(row) is list and len(row) == 6 and all(type(n) is int for n in row)
                    for row in titan["counts_by_fold_and_class"]), "Untyped Titan counts")
    for tid, lon, lat, label, fold in titan["points"]:
        lonlat([lon, lat], east=True); integer(label, 0, 5); integer(fold, 0, 4)
        require(BOUNDARIES[fold] <= lon < BOUNDARIES[fold+1], "Titan point/region differs")
        counts[fold][label] += 1
    require(counts == TITAN_COUNTS == titan["counts_by_fold_and_class"] and titan["sector_boundaries_east_deg"] == BOUNDARIES
            and titan["catalog_count"] == 23380 and titan["excluded_catalog_count"] == 134
            and digest(json.dumps([r[0] for r in titan["points"]]).encode()) == titan["test_union_sorted_ids_sha256"],
            "Titan payload support/identity differs")
    earth = p["earth"]; support = Counter()
    for tid, site, role, center, geometry in earth["footprints"]:
        require(type(site) is str and site in EARTH_SITES and role in ("train", "val"), "Invalid Earth role/site")
        lonlat(center); geometry_rings(geometry); support[site, role] += 1
    expected_sites = [dict(site_id=s, name=n, class_index=c, train_count=t, val_count=v)
                      for s, (c, n, t, v) in sorted(EARTH_SITES.items(), key=lambda x: (x[1][0], x[0]))]
    require(type(earth["sites"]) is list and all(type(s) is dict
            and all(type(s.get(k)) is int for k in ("class_index", "train_count", "val_count"))
            for s in earth["sites"]) and type(earth["absent_site"].get("count")) is int,
            "Untyped Earth counts")
    require(earth["sites"] == expected_sites and earth["absent_site"] == {"site_id": "haughton", "count": 0}
            and earth["absent_class"] == "hummocky"
            and earth["geometry_note"] == "Retained nominal source-window outlines and geometric centers; no new geometry or SAR validity masking."
            and all((support[s, "train"], support[s, "val"]) == (t, v) for s, (_, _, t, v) in EARTH_SITES.items()),
            "Earth payload support differs")
    venus = p["venus"]; complete = 0
    for tid, lon, lat, fraction in venus["points"]:
        lonlat([lon, lat], east=True); number(fraction, 0, 1); complete += fraction == 1
    require(complete == venus["complete_count"] == 2526 and 2945-complete == venus["partial_count"] == 419,
            "Venus completeness differs")
    require(venus["examples"] == [dict(code=chr(65+i), tile_id=tid) for i, tid in enumerate(VENUS_EXAMPLES)]
            and set(VENUS_EXAMPLES) <= {r[0] for r in venus["points"]}, "Venus example join differs")
    return p

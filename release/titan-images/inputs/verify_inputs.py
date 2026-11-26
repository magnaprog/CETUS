"""Verify the released Titan images, catalog and five geographic splits."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct

CATALOG_SHA256 = '8152226f876e0cb5fad105e0b00a37ef5b62d156d7f6eb3a85b77c549a402967'
PROTOCOL_SHA256 = '1ce5fd9e8b135a172ce3e671b12a9a06e160e089debb9cac22bb14213f784060'
SPLIT_SHA256 = ('b69e2fa3551dde125edf31a5601c52ccae755cb6169ed57d145b16f6983f85ab', '58ad14b783c0d6ffd2851cb622f1d0d4fd1438978aeb146e32e5dcb8d55a8adc', '2eee98044e7a636a6726800328b0361991432e9ea272eaee7873b3faa2ee28d2', '2dbd6117de1c299655116af827ee95998839c67da443bf2a03729b7e4e2fab75', '1df555f0f95c16e3fb44243ce7ecca41225e4e588353148f25725c68f1cb4271')

def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_npy(path):
    with path.open("rb") as stream:
        require(stream.read(8) == b"\x93NUMPY\x01\x00", "Unexpected NPY format: " + str(path))
        size = struct.unpack("<H", stream.read(2))[0]
        header = ast.literal_eval(stream.read(size).decode("latin1"))
        require(header == {"descr": "<f4", "fortran_order": False, "shape": (128, 128)},
                "Unexpected tile dtype/shape/order: " + str(path))
        require(path.stat().st_size == 10 + size + 128 * 128 * 4, "Unexpected NPY length")


def verify(root):
    root = Path(root).resolve(strict=True)
    paths = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "Symlink in input package")
    names = {str(p.relative_to(root)) for p in paths if p.is_file()}
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["schema"] == "titan-images-v1", "Unknown image package schema")
    catalog_path = root / "data/catalog.json"
    require(sha(catalog_path) == CATALOG_SHA256, "Titan catalog identity differs")
    require(sha(root / "data/splits/protocol.json") == PROTOCOL_SHA256, "Protocol identity differs")
    protocol = json.loads((root / "data/splits/protocol.json").read_text())
    catalog = json.loads(catalog_path.read_text())
    tiles = catalog["tiles"]
    ids = [t["tile_id"] for t in tiles]
    require(len(ids) == len(set(ids)) == 23380, "Catalog cardinality differs")
    payload_names = {"README.md", "LICENSE", "DATA_LICENSES.md", "THIRD_PARTY_NOTICES.md",
                     "NOTICE.md", "verify_inputs.py", "data/catalog.json", "data/splits/protocol.json"}
    payload_names.update(f"data/splits/titan_contiguous_fold_{f}.json" for f in range(5))
    payload_names.update(f"data/tiles/{i}.npy" for i in ids)
    require(names == payload_names | {"manifest.json", "SHA256SUMS"}, "Missing or extra package file")
    require(set(manifest["files"]) == payload_names, "Manifest member set differs")
    for name, item in manifest["files"].items():
        path = root / name
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                "Package member changed: " + name)
    sums = "".join(f"{sha(root / n)}  {n}\n" for n in sorted(payload_names | {"manifest.json"}))
    require((root / "SHA256SUMS").read_text() == sums, "Checksum inventory differs")
    for tile in tiles:
        name = f"data/tiles/{tile['tile_id']}.npy"
        require(manifest["files"][name]["sha256"] == tile["tile_sha256"], "Accepted tile hash differs: " + name)
        check_npy(root / name)
    require(protocol["catalog_sha256"] == CATALOG_SHA256 and len(protocol["splits"]) == 5,
            "Catalog/protocol binding differs")
    counts = []
    test_ids = set()
    for fold, expected in enumerate(SPLIT_SHA256):
        path = root / f"data/splits/titan_contiguous_fold_{fold}.json"
        require(sha(path) == expected, "Titan split identity differs")
        require(protocol["splits"][fold] == {"fold": fold, "path": path.name, "sha256": expected},
                "Protocol/split binding differs")
        split = json.loads(path.read_text())
        require(split["catalog_sha256"] == CATALOG_SHA256 and split["policy"]["test_fold"] == fold
                and set(split["assignments"]) == set(ids), "Catalog/split join differs")
        region = {tile for tile, role in split["assignments"].items() if role == "test"}
        require(test_ids.isdisjoint(region), "Overlapping geographic test regions")
        test_ids.update(region)
        counts.append({role: sum(split["assignments"][i] == role for i in ids)
                       for role in ("train", "val", "test")})
    union_sha = hashlib.sha256(json.dumps(sorted(test_ids)).encode()).hexdigest()
    require(len(test_ids) == protocol["test_union_count"] == 23246
            and union_sha == protocol["test_union_sorted_ids_sha256"], "Test union differs")
    return {"status": "inputs_verified", "distinct_test_tiles": len(test_ids), "tiles": len(ids), "fold_role_counts": counts,
            "catalog_sha256": CATALOG_SHA256, "model_executed": False,
            "normalization_recomputed": False, "inference_reproduced": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, allow_nan=False))

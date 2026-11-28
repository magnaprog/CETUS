# Verifying a CETUS release

To verify the downloaded Titan images or rebuild their archive, use the [image collection guide](titan-images/README.md). The checks below apply to the Git checkout. Run them from the repository root.

```sh
python3 release/check_inventory.py
```

The [file inventory](inventory.json) records repository-relative paths, file sizes and SHA-256 digests. The checker uses Git’s tracked file list and compares it with the inventory and the root [SHA256SUMS](../SHA256SUMS). It also checks that the catalog, result, software and release records identify the files present in the checkout. The [software source record](software-sources.json) identifies the library, checkout scripts and their original source files. The [provenance record](provenance.json) identifies the scientific packages and the separately published image archive.

## Which source should I use?

The [checkout runner](../scripts/run_probing.py) imports the [root Python package](../cetus_sar/). Each [training implementation](../training/README.md) has its own import root and source manifest. For example, the [encoder comparison manifest](../training/encoder-comparisons/SOURCE_EXPORT.json) records the files used by that runner, including modules with the same names as those in the root package. Run the examples from the directory named in the corresponding guide.

## Updating the inventory

After editing a package, verify its local results and checksums using that package’s README. If a file referenced by a curated record changes, update that specific record using the verified file. Stage additions, moves and deletions so Git lists the paths intended for the release. Then refresh the calculated sizes and digests and check them again.

```sh
python3 release/check_inventory.py --update
python3 release/check_inventory.py
git diff HEAD -- release/inventory.json SHA256SUMS
```

The updater checks curated source and scientific identities before it refreshes the calculated file entries. An earlier producer’s recorded input digest belongs to that run.

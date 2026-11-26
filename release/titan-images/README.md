# Titan image collection

The [v0.2.0 release](https://github.com/magnaprog/CETUS/releases/tag/v0.2.0) includes 23,380 Cassini radar images, their terrain map labels and five geographic splits. The five test sets together contain 23,246 distinct images. See the [release notice](RELEASE_NOTICE.md) for source credits and terms.

Download `titan-images.tar.zst`, `RELEASE_NOTICE.md` and `SHA256SUMS.release` to the same directory. From that directory, check both files and extract the archive into a new directory. The verifier checks every image against the catalog and reports the train, validation and test counts for each fold.

```sh
sha256sum -c SHA256SUMS.release &&
  mkdir titan-images &&
  tar --zstd -xf titan-images.tar.zst -C titan-images &&
  python3 -I -B titan-images/verify_inputs.py
```

Keep verification output outside the extracted directory. The verifier checks its file inventory. The [classifier guide](../../docs/classifiers.md) and [encoder training guide](../../training/encoder-comparisons/CONTRACTS.md) describe how to use the catalog and splits.

## Rebuild the release files offline

From the root of the v0.2.0 source checkout, supply the extracted collection and a new output directory. Use Python 3.12 and zstd 1.5.5 for byte-for-byte reproduction. Allow about 1.4 GB for the output.

```sh
python3 release/titan-images/build.py /path/to/titan-images rebuilt-assets
(cd rebuilt-assets && sha256sum -c SHA256SUMS.release)
```

The builder verifies the extracted files, packs them in filename order and checks the archive and notice against the [release checksums](SHA256SUMS.release) before saving all three files. The command repackages the processed images. The [image guide](inputs/README.md) describes their origin and preprocessing.

# CETUS

Cross-domain Evaluation of Earth-to-Titan Transfer Using SAR (CETUS). This
review package contains catalog metadata, split definitions, and result
summaries for the Representations for the Physical Sciences workshop at
NeurIPS 2026. Tiles, model checkpoint files, extracted embeddings, and
analysis code are not included.

CETUS asks whether representations with encoder weights unchanged support
terrain map classification on Titan, whether a classifier trained on Earth
remains useful on Titan, and whether smaller global differences between Earth
and Titan representations improve that transfer. The catalogs cover 11,371
labeled Titan tiles, 398 labeled Earth tiles, and 2,945 unlabeled Venus tiles,
each 45 km by 45 km at 128 by 128 pixels. Join catalogs, splits, and Selk
records by tile ID. Titan tiles contain uncalibrated 8-bit display digital
numbers after logarithmic stretching.

## Contents

- `catalogs/`: Earth, Titan, and Venus catalog metadata. Local paths were
  removed. Earth tile records retain source product IDs. Titan and Venus name
  source products in catalog metadata, and product hashes appear where supplied.
  Tile records retain tile IDs, tile hashes, labels, and summary statistics.
- `splits/`: 19 split files joined by tile ID. Five Titan spatial folds,
  `earth_transfer_grouped_validation.json`, and 13 Earth files named after the
  morphological analog sites. Each file records the SHA-256 of the matching
  catalog.
- `results/`: fold summaries, Venus and Selk summaries, and records for
  constrained training, preprocessing, and recall after omitting each
  Earth source site. The curve of macro recall versus the number of labeled
  Titan training tiles
  is split by model so each file stays under 8 MB. `final_results.json`
  records hashes of the result files.
- `DATA_LICENSES.md` and `THIRD_PARTY_NOTICES.md`: source terms, including the
  required VIMS+ISS BSD 3-Clause notice.
- `FILES.md`: file guide.
- `provenance.json`: catalog hashes and which result keys were dropped.
- `file_index.json` and `SHA256SUMS`: file list and hashes.

Check hashes with `sha256sum -c SHA256SUMS`. `file_index.json` lists
hashes for every file except itself and `SHA256SUMS`. The checksum file
also covers `file_index.json`.

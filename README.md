# CETUS

How well do image representations learned on Earth transfer to Cassini synthetic aperture radar (SAR) images of Titan? CETUS compares features from pretrained and untrained encoders with measurements of radar intensity and texture. The terrain labels come from the geomorphological map of Lopes et al., which draws partly on the Cassini radar observations. A correct prediction means agreement with the mapped class.

The [CETUS paper](https://arxiv.org/abs/2610.07576) reports results on five longitude sectors. Those test regions contain 23,246 distinct tiles from the 23,380-tile catalog. A separate catalog of 11,371 Titan tiles supports classification within Titan and transfer from Earth. After reserving 65 tiles at Selk, the split assigns whole geographic areas roughly 250 km across to five folds. The [data guide](data/README.md) identifies the catalogs and splits for both evaluations. The [correction record](docs/corrections.md#western-titan-coverage) explains the difference in map coverage.

| Task | Start here |
| --- | --- |
| Check a result from saved predictions | [Count Craters predictions with Python's standard library](experiments/crater-training/README.md#compare-logistic-classifiers-on-craters), or [browse experiments by question](experiments/README.md). The [reproduction guide](docs/reproduce.md) explains which scores you can compare. |
| Get the images | [Download the 23,380 processed Titan tiles](https://github.com/magnaprog/CETUS/releases/download/v0.2.0/titan-images.tar.zst) and follow the [verification and extraction instructions](https://github.com/magnaprog/CETUS/releases/download/v0.2.0/RELEASE_NOTICE.md). |
| Fit models on the images | [Run the native classifiers](docs/classifiers.md), or [choose a training implementation](training/README.md) for initialization, loss normalization or DOFA fine-tuning. |
| Work on the classifier code | The [Python library](cetus_sar/) supplies the checkout runner. Run its [tests](tests/) from the repository root with the [installation and test commands](docs/classifiers.md#install-and-check). |
| Verify files in this checkout | [Release inventory and checksums](release/README.md). |

## Citation

Cite Kevin Lee’s [CETUS paper](https://doi.org/10.48550/arXiv.2610.07576). Download the [BibTeX entry from arXiv](https://arxiv.org/bibtex/2610.07576) or use the [citation record in this checkout](CITATION.cff).

Project code is released under the [MIT license](LICENSE). See the [data terms](licenses/DATA_LICENSES.md) and [third-party notices](licenses/THIRD_PARTY_NOTICES.md) for source credits and redistribution terms.

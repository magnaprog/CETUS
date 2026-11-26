# CETUS

How well do image representations learned on Earth transfer to Cassini synthetic aperture radar (SAR) images of Titan? CETUS compares features from pretrained and untrained encoders with measurements of radar intensity and texture. The terrain labels come from the geomorphological map of Lopes et al., which draws partly on the Cassini radar observations. A correct prediction means agreement with the mapped class.

The longitude-sector benchmark contains 23,380 Titan tiles. Its five test regions cover 23,246 distinct tiles. The spatial-block experiments use a smaller Titan catalog for within-Titan classification and transfer from Earth. See the [data guide](data/README.md) for the populations and the [correction record](docs/corrections.md) for the change in map coverage.

| Task | Start here |
| --- | --- |
| Check a result from saved predictions | [Count Craters predictions with Python's standard library](docs/reproduce.md#craters-training-and-map-composition), or [browse experiments by question](experiments/README.md). |
| Get the images | [Download the 23,380 processed Titan tiles](https://github.com/magnaprog/CETUS/releases/download/v0.2.0/titan-images.tar.zst) and follow the [verification and extraction instructions](https://github.com/magnaprog/CETUS/releases/download/v0.2.0/RELEASE_NOTICE.md). |
| Fit models on the images | [Run the native classifiers](docs/classifiers.md), or [choose a training implementation](training/README.md) for initialization, loss normalization or DOFA fine-tuning. |
| Verify files in this checkout | [Release inventory and checksums](release/README.md). |

## Citation

```bibtex
@misc{lee2026cetus,
  author        = {Lee, Kevin},
  title         = {{CETUS}: How Far Do Representations Trained on {Earth} Transfer to {Cassini} {SAR} of {Titan}?},
  year          = {2026},
  eprint        = {2610.07576},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.07576}
}
```

Project code is released under the [MIT license](LICENSE). See the [data terms](licenses/DATA_LICENSES.md) and [third-party notices](licenses/THIRD_PARTY_NOTICES.md) for source credits and redistribution terms.

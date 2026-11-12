# DOFA partial-training source

This directory contains the training and replay runner, its supporting helper,
the original training modules, and focused tests for the paired DOFA comparison:
numeric inputs 13.78 and 3.75 across five geographic folds and five seeds.

The files support source inspection and preparation of new runs. Training and
checkpoint replay require the original source checkout, data, model assets,
environment and executable contract described in [CONTRACTS.md](CONTRACTS.md).
Those prerequisites are not bundled here; the supplied contract view is
documentation only.

- [Study recipe](docs/DOFA_PARTIAL_BAND_PLAN.md) and [band interpretation](docs/DOFA_INPUT_REVIEW.md)
- [Source origins and hashes](SOURCE_EXPORT.json) and [historical file inventory](contracts/historical-source-inventory.json)
- [Documentation contract view](contracts/candidate02.public.json)

Check file integrity from this directory:

```sh
sha256sum -c SHA256SUMS
```

Kevin Lee's code uses the [MIT license](LICENSE). Retain the
[CROMA license](licenses/CROMA-LICENSE), [data credits](DATA_LICENSES.md) and
[third-party notices](THIRD_PARTY_NOTICES.md). External software and model assets
have the separate terms described in [CONTRACTS.md](CONTRACTS.md#runtime-and-external-assets).

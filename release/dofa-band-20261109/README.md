# DOFA numeric input comparison

This comparison changes DOFA's numeric band input from 13.78 to 3.75 while
retaining its encoder weights, images and processing. All five geographic
regions use fresh logistic regression classifiers with the same settings.
The value 3.75 is specified for Sentinel-1 in
[Appendix D of the DOFA preprint, version 2](https://arxiv.org/html/2403.15356v2#A4);
the [pinned upstream README](https://github.com/zhu-xlab/DOFA/blob/73ff5c0721da322689ae890bed5b4efd78935f47/README.md)
uses 5.405. The checkpoint's pretraining identifier convention and the appropriate
input for Cassini remain unresolved.

| Numeric input | Mean macro F1 | Sample SD among regions |
| --- | ---: | ---: |
| 13.78 | 35.91% | 3.52% |
| 3.75 | 38.42% | 3.85% |

All five differences are positive, with mean and sample SD 2.51 ± 0.96
percentage points. Craters F1 changes from 1.61% to 1.35%. These are
exploratory comparisons on previously examined test regions; the scores
measure agreement with an expert map.

The ZIP contains ten records of test IDs, labels, predictions and probabilities
and a NumPy reconstruction script. It checks class and macro metrics,
regional means, sample SDs and paired differences. It also supplies small
corruption tests and a ZIP builder. The 22 members total 6,922,999 bytes as
packaged; [members.json](members.json) lists their individual hashes.

Run from this companion directory. Set `CETUS_DOFA_WORK` to a new directory outside
the CETUS checkout; its parent must exist.

```sh
CETUS_DOFA_WORK=/absolute/path/to/external-dofa-reconstruction
mkdir "$CETUS_DOFA_WORK"
unzip dofa-numerical-companion.zip -d "$CETUS_DOFA_WORK/inputs"
PYTHONDONTWRITEBYTECODE=1 sh "$CETUS_DOFA_WORK/inputs/reproduce.sh" "$CETUS_DOFA_WORK/results"
cmp dofa-numerical-companion.zip "$CETUS_DOFA_WORK/results/dofa-numerical-companion.zip"
```

Python 3.12.3 and NumPy 2.4.4 were used for the
saved arithmetic check. The script installs nothing and makes no network
requests. The embedded README explains class order, averaging and tolerances.

The underlying study completed twenty GPU feature matrices and ten CPU fits.
The ten matrices at 13.78 reproduced the original features exactly. Separate
replay of the saved classifiers reproduced every prediction and probability;
all ten fits completed below the iteration limit without warnings. The ZIP
reconstructs saved arithmetic only. It omits images, features and estimators
and performs no inference or fitting.

The original scientific source is development commit
`a45526c44d935c89ca8ecd94812be49ba8f659a7`. The companion's source and data
are preserved inside the ZIP. Earlier Earth and partial-training experiments
retain their original inputs and results. [NOTICE.md](NOTICE.md) provides
public attribution while preserving the anonymous package and source credits.

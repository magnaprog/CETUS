# Titan saved results, 9 November 2026

This package publishes four preserved numerical archives for the current Titan
evaluation. It supplies saved predictions, numerical records, small SAR/map
inspection excerpts and CPU reconstruction code. The catalog contains 23,380
tiles, with 23,246 distinct test tiles across five geographic folds.

The initialization study's 90 cases and the thirty saved epoch-100 evaluations
were complete by 8 November. This release adds no model run or scientific
result. The five earlier CETUS packages retain their original bytes and metrics.
See the [access inventory](DATA_ACCESS.md) for supplied and missing inputs and
the [notice](NOTICE.md) for attribution, assistance and historical review wording.

The separate [processed-image release](https://github.com/magnaprog/CETUS/releases/tag/titan-processed-images-20261109)
provides all 23,380 current Titan arrays with the matching anonymous catalog and
five divisions. Its [companion notice](https://github.com/magnaprog/CETUS/releases/download/titan-processed-images-20261109/RELEASE_NOTICE.md)
gives source terms and input-verification commands. Public download and input
verification passed on 9 November 2026; the four numerical ZIPs below are unchanged.

| Archive | Entries | Compressed bytes | Reconstruction scope |
| --- | ---: | ---: | --- |
| [Scientific](scientific-supplement.zip) | 203 | 49,621,594 | 315 saved test-prediction vectors, including 80 fitting cells; sixteen settings, twenty contrasts and eleven table fragments. |
| [Follow-up](followup-supplement.zip) | 84 | 4,385,713 | Coverage for 115 saved vectors; votes from 2,685 sampled map grids and sixty SAR inspection packets. Coverage requires the unchanged scientific ZIP. |
| [Initialization](initialization-supplement.zip) | 19 | 35,858,028 | 90 cells: thirty DINOv2 readouts, thirty ResNet18 tuning fits and thirty final fits. Sixty native test records, 6,000 validation-history rows from sixty CNN fits, ten learning-rate choices and 180 coverage confusions. |
| [Reassessment](reassessment-supplement.zip) | 54 | 32,559,371 | Thirty epoch-100 validation/test records, validation-choice checks and thirty-seven SAR/map excerpts. Requires the unchanged initialization ZIP. |

The archives total 122,424,706 bytes and 360 entries. Entries overlap between
archives; this is not a count of unique scientific files. Exact hashes, source
paths, original producers and receipt identities are in
[provenance.json](provenance.json). The source revision used to publish the
archives is distinct from the revisions that produced their results.

## Reconstruct saved results

Use Python 3.12.3 and NumPy 2.4.4, the recorded numerical versions. Run the
commands below in one shell from the CETUS repository root. They create a new
external work directory, verify the four ZIP identities, extract each into its
own directory and write reports outside the extracted packages. Installation
uses PyPI; reconstruction needs no network, model library, images beyond the
supplied excerpts, or weights. Inspect the extracted READMEs and scripts before
running their consumers. Keep failed outputs and use a fresh directory for a retry.

The initialization verifier rejects extra files, including caches and reports.
Use only its `reconstruct.py` entrypoint: the copied historical helper CLIs
require inputs that are not supplied. Reassessment reads the initialization
ZIP directly and uses `MANIFEST.json` in place of a checksum list.

```sh
set -eu
export CETUS_PACKAGE="$PWD/release/titan-followup-20261109"
CETUS_REPLAY_WORK="$(mktemp -d "${TMPDIR:-/tmp}/cetus-saved-results.XXXXXXXX")"
export CETUS_REPLAY_WORK
mkdir "$CETUS_REPLAY_WORK/reports"
python3.12 -m venv "$CETUS_REPLAY_WORK/venv"
export CETUS_PYTHON="$CETUS_REPLAY_WORK/venv/bin/python"
"$CETUS_PYTHON" -c 'import sys; assert sys.version_info[:3] == (3, 12, 3)'
"$CETUS_PYTHON" -m pip install --only-binary=:all: numpy==2.4.4
"$CETUS_PYTHON" -m pip check
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$CETUS_PYTHON" -B - <<'PY'
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

package = Path(os.environ['CETUS_PACKAGE']).resolve()
work = Path(os.environ['CETUS_REPLAY_WORK']).resolve()
pins = {
    'scientific': '67b6aa65be5bab9b7cbe83df90aafcf7b479117291562471652f59ed7f182a1d',
    'followup': '6a6179b76278d7fac55924643a24133d9304deb8a77d1706316e1d171d775eaf',
    'initialization': 'ada744e9278f4fbac5f5f953b5688132ad0b080b35cff32ed23b7f02c3192486',
    'reassessment': '7c4a2f80e53b0c7834e526d10a4f334da40d376ba9a5c74f218505f6ec7c8dff',
}
counts = {'scientific': 203, 'followup': 84, 'initialization': 19, 'reassessment': 54}
for name, expected in pins.items():
    archive = package / (name + '-supplement.zip')
    with archive.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
    destination = work / name
    destination.mkdir()
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        assert len(members) == counts[name]
        assert len({m.filename for m in members}) == len(members)
        for member in members:
            rel = PurePosixPath(member.filename)
            assert not rel.is_absolute() and '..' not in rel.parts
            assert str(rel) == member.filename and '\\' not in member.filename
            assert not stat.S_ISLNK(member.external_attr >> 16)
        z.extractall(destination)
PY

(cd "$CETUS_REPLAY_WORK/scientific" && sha256sum -c SHA256SUMS)
(cd "$CETUS_REPLAY_WORK/followup" && sha256sum -c SHA256SUMS)
(cd "$CETUS_REPLAY_WORK/initialization" && sha256sum -c SHA256SUMS)
# Reassessment has MANIFEST.json, not SHA256SUMS; check.py verifies its members.

(
  cd "$CETUS_REPLAY_WORK/scientific"
  "$CETUS_PYTHON" -B scripts/replay_metrics.py \
    --output "$CETUS_REPLAY_WORK/reports/scientific.json"
  "$CETUS_PYTHON" -B scripts/prepare_cvpr_tables.py \
    --output paper/cvpr2027/tables --check
)
(
  cd "$CETUS_REPLAY_WORK/followup/coverage"
  "$CETUS_PYTHON" -B analyze_sar_coverage.py \
    --package "$CETUS_PACKAGE/scientific-supplement.zip" \
    --sha256 67b6aa65be5bab9b7cbe83df90aafcf7b479117291562471652f59ed7f182a1d \
    --output "$CETUS_REPLAY_WORK/reports/coverage.json"
  "$CETUS_PYTHON" -B - <<'PY'
import json
import os
from pathlib import Path
original = json.loads(Path('results.json').read_text())
replayed = json.loads((Path(os.environ['CETUS_REPLAY_WORK']) /
                       'reports/coverage.json').read_text())
original.pop('execution')
replayed.pop('execution')
assert original == replayed
PY
)
(
  cd "$CETUS_REPLAY_WORK/followup/label-support"
  "$CETUS_PYTHON" -B replay_vote_inputs.py data \
    --report "$CETUS_REPLAY_WORK/reports/votes.json" \
    --expect-manifest-sha256 be98e9950e19641889e8e8234a43b3b2913648848407df38796fc58b9e556e69
)
(
  cd "$CETUS_REPLAY_WORK/initialization"
  "$CETUS_PYTHON" -B reconstruct.py \
    --output "$CETUS_REPLAY_WORK/reports/initialization.json"
)
(
  cd "$CETUS_REPLAY_WORK/reassessment"
  "$CETUS_PYTHON" -B check.py \
    --initialization-zip "$CETUS_PACKAGE/initialization-supplement.zip" \
    > "$CETUS_REPLAY_WORK/reports/reassessment.json"
)
```

Successful exits must be accompanied by the expected report scope. The
scientific report says `pass`; its vector counts sum to 315, and the separate
table command checks eleven fragments. Coverage must equal the packaged result
after removing only `execution` metadata. The vote checker checks all saved
votes, summaries and sixty inspection packets. Initialization reports
`numeric_reconstruction_pass` with 90 cells, sixty native test vectors, 6,000
history rows, ten learning-rate choices and 180 cohort confusions. Reassessment
reports `supplied_numerical_records_reconstructed` and includes the epoch-100
table check. Inspect warnings and retain command output and runtime versions
with the reports. Recheck input checksums after reconstruction.

The supplied receipts describe earlier successful checks; this wrapper's
documentation does not itself constitute a new numerical execution receipt.
These consumers calculate saved-result arithmetic. They do not repeat model
training, generate GPU probabilities, reproduce omitted per-epoch predictions,
or verify every manuscript claim. Validation-removal counts use supplied
outcomes and cannot independently regenerate every removal score.

## Interpretation and execution limits

The 80-fit study retains 23 iteration caps and 54 gradient concerns. The
initialization study compares five random DINOv2 states with one pretrained
state under matched architecture and inputs. ResNet18 uses three seeds per
initialization and separately selected learning rates. Thirty DINO feature
jobs supplied the readout inputs; they are dependencies of the 90-case analysis.
The epoch-100 comparison evaluated saved states without retraining or
reselection. Original validation-selected results remain primary.

Scores measure agreement with a SAR-informed terrain map. Seeds are averaged
within each region before averaging the five equally weighted regions; the
regional SD is descriptive. The studies reuse examined test regions and do
not establish convergence, equal optimization adequacy, a causal pretraining
effect or independent geological accuracy. Rare Craters support, differences
in class/geographic composition and unsupported-class null summaries remain
part of the interpretation. Coverage differences do not isolate the effect
of missing pixels. DOFA's checkpoint/input convention remains unresolved.

The sixty follow-up inspection packets and thirty-seven reassessment excerpts
are not asserted to be disjoint and do not supply the full pixel collection.
Embeddings, model states, full label rasters and a complete current training
environment are absent. The separate [current source package](../current-source-20261109/README.md)
provides the execution code and documents these prerequisites. For the earlier
classifier runner, use
[SOFTWARE.md](../../SOFTWARE.md) with the images from the separate release and
upstream weights. See [DATA_ACCESS.md](DATA_ACCESS.md) for the distinction between
recorded author-side checks and independent-recipient access.

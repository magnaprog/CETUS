# DOFA band-input comparisons

Two experiments compare the DOFA band inputs 13.78 and 3.75 on the same five
geographic test regions. Choose the records for the calculation you want to run.

| Package | What you can check |
| --- | --- |
| [Frozen encoder](frozen-encoder/README.md) | Recalculate class and macro scores from ten logistic regression fits on features from the fixed encoder, one fit per band input and region. |
| [Partial training](partial-training/README.md) | Recalculate test metrics from the saved class predictions for fifty fits that trained the final two encoder blocks. Each band input has five seeds per region. |
| [Probability scores](probability-scores/README.md) | For those same fifty fits, check each prediction against its six class probabilities and the selected epoch against its validation macro recall history. |

Run each guide’s commands from its package directory. All three use saved results on CPU.

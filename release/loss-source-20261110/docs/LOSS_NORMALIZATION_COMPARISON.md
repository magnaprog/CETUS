# Class-weight normalization comparison

This prospective experiment asks whether changing the denominator of the
training loss changes the Craters predictions of ImageNet ResNet18 on the
existing five Titan regions. It adds fifteen fits, one for each region and
seed 0, 1 and 2. The fifteen accepted ImageNet fits from producer
`1f3fc8fe0e831777e3188d70caf2c5450b656333` provide the comparison. The original
results remain intact. This is an additional sensitivity study, not a repair
of the standard weighted cross-entropy implementation.

## Loss calculation

Let `u[c]` be the original positive float32 class weight, `K[c]` the number
of training images in class `c`, and `b` the actual number of images in the
batch. Both computations use the same float32 weighted cross-entropy sum
`T`. Original training divides `T` by the float32 sum of the batch's label
weights. New training divides `T` by a population normalization:

```text
m64 = math.fsum(float(K[c]) * float(u[c]) for c in range(6)) / sum(K)
m32 = float32(m64)
denominator = float32(float32(b) * m32)
loss = T / denominator
```

The six original weights retain their bits. Compute `m64` on CPU using
Python binary64 values, round once to float32, then reconstruct that exact
float32 value on the training device.
Both multiplication and final division use float32. Retain the weight bits,
counts, `m64` and `m32` bits in each result. The last partial batch uses its
actual size. The existing epoch training-loss diagnostic remains total `T`
divided by total native label weight. Separate per-batch records describe
the loss used for optimization, the two denominators and class counts.

The intervention changes the scale of the whole batch gradient. It does not
change relative class weights inside a batch, resample images, or increase
the number of batches containing Craters. Floating-point rounding prevents
a claim of exact equality to ideal rational class weights. Live BatchNorm
and evolving parameters also prevent interpreting the actual gradient as an
unbiased estimate of a separable image risk.

## Training and evaluation

Reuse the original catalog, all train/validation/test images and roles,
training normalization, ImageNet weights, head initialization, transforms,
and runtime. Each comparison uses its original physical GPU UUID. There are
five jobs on each of the three RTX 3080 Ti devices. Reuse is conditional on
the recorded input, source, runtime and initial-state checks.

Keep batch size 32 with the final partial batch, 100 complete epochs, learning
rate 0.0001, AdamW betas 0.9 and 0.999, epsilon 1e-8, weight decay 0.01, and
the original cosine schedule. Train all layers and BatchNorm. Preserve seed
`s`, head seed `20000+s`, loader seed `10000+s`, float32 arithmetic, single
CPU threads, and the original GPU precision settings. Add no augmentation,
gradient accumulation, sampling changes or learning-rate search. Logging
must not consume random numbers or create another loader iteration.

Select the earliest maximum unrounded six-class validation macro F1 among
the 100 completed epochs, independently in each run. Save initial, selected
and final states, all validation histories, actual training-order digests,
and per-batch loss records. The primary test evaluation uses the selected
state and every original test image. This study does not add a separate
epoch-100 test evaluation. Preserve native argmax and metric conventions.

The primary comparison is the difference in Craters F1. Report Craters
TP, FP, FN, number predicted, precision and recall, every class's confusion
counts, and six-class macro F1. Compute differences for each region and
seed, average the three seed differences within region, then give the five
regional differences and their equal-region mean. Seeds do not add
independent geographic observations. Do not select a replacement checkpoint,
threshold, loss, metric or experiment from these test outcomes.

All original selected ImageNet runs have zero Craters true positives. Their
Craters F1 is zero, so the new F1 differences cannot be negative. This alone
provides no sign-test or robustness evidence. False positives and effects on
other classes remain essential. A singleton Craters test image has recall
zero or one. Repeated image predictions do not count independent craters.

## Execution and interpretation

Bind each new specification to the accepted original result, its full output
inventory and source identities. Check the original and new initial tensor
states before training. Preserve fresh runtime and input checks, atomic
publication, failure records, process exit codes and complete outputs.
After training, reproduce the selected validation and test predictions in a
fresh process from each saved state. This verifies the recorded evaluation;
it adds no checkpoint choice or new test endpoint.
Keep new model states on the GPU server, following the author's storage
choice; copy smaller analysis records to persistent local disk.

Run the fifteen specified fits without interpreting partial test results.
Retain failures and diagnose them before any retry with the same scientific
settings. Completion means all fifteen declared jobs have valid outputs;
a favorable result is not a stopping rule. Accept an all-zero Craters result.

Earlier test inspection motivated this experiment. It therefore measures
sensitivity on reused geographic regions. The inherited learning rate was
chosen under the original loss, so this is not a comparison of equally tuned
losses. Different validation-selected epochs are part of the measured effect.
Neither a positive nor a negative result establishes a unique cause of the
Craters errors, general class-balancing performance, or geological accuracy.
The current map-agreement qualifications remain in force.

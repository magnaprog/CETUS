# Experiments

Each experiment includes saved results and a way to check them. For a first check on CPU, [count Craters predictions](crater-training/README.md#compare-logistic-classifiers-on-craters). To repeat a fit on images, see the [training implementations](../training/README.md).

## Which classifiers recover the mapped terrain?

| Question | Evidence |
| --- | --- |
| How well do classifiers on encoder features generalize across Titan? | The [longitude-sector benchmark](longitude-benchmark/README.md) supplies five geographic splits and predictions from linear classifiers fitted by SGD and from cosine nearest neighbors. |
| How do encoder features compare with intensity and texture measurements? | The [regional prediction records](regional-predictions/README.md) give the test IDs, class labels and logistic regression decisions for each feature set. |
| Which classes benefit from pretraining? How does the answer depend on the fit? | The [encoder comparison archives](encoder-comparisons/README.md) cover classifier fitting, SAR coverage, DINOv2 readouts with matched input transforms, ResNet18 training and checkpoint selection. |
| How highly does each model rank tiles of a given class? | Reconstruct [average precision](class-ranking/README.md) for each fit and region. The [plots](terrain-plots/README.md) show the number of tiles retrieved, regional variation across classifier seeds and sampling locations on Titan, Earth and Venus. |

## Which training choices affect the result?

| Question | Evidence |
| --- | --- |
| What changes when part of an encoder is trained or training images are replaced? | Reconstruct the [adaptation and replacement scores](adaptation-controls/README.md). The [interpretation study](terrain-interpretation/README.md) examines differences by class, the effect of pooling regions, test-to-training distances and seven example tiles. |
| Does the denominator of the weighted training loss affect ResNet18? | The [loss normalization experiment](loss-normalization/README.md) compares the two denominators across the same regions and seeds. |
| How sensitive is DOFA to the numeric band input? | The [DOFA studies](dofa/README.md) compare two band inputs with the encoder held fixed and with two blocks trained. The latter fits also have saved probabilities and validation histories. |

## Where do the labels and images come from?

| Question | Evidence |
| --- | --- |
| How much terrain of each class lies within a tile? | The [map census](map-census/README.md) counts known and unknown cells and applies four SAR coverage rules to the 23,380 Titan tiles. |
| How often can a batch contain Craters, and which Craters tiles are recovered? | The [Craters analysis](crater-training/README.md) estimates batch availability from training counts and compares saved predictions with map composition on test tiles. |
| Which records describe transfer from Earth, the Selk holdout and the unlabeled Venus comparison? | The [spatial-block catalogs, splits and results](../data/README.md) cover those experiments. The [reproduction guide](../docs/reproduce.md#choosing-an-experiment) explains which Titan population and scaling rule each comparison uses. |

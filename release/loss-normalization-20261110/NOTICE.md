# Data and software credits

The numerical records contain model predictions, terrain labels and training
measurements. They accompany the CETUS comparison of loss normalization in
ResNet18. Model weights and original images are not included.

Terrain labels derive from the map of Lopes et al. 2020 and the dataset
“Titan Global Map Shapefiles,” contributed by Ashley Schoenfeld to Mendeley
Data, version 1: https://data.mendeley.com/datasets/f6jrtyfp66/1.
The dataset uses the Creative Commons Attribution 4.0 license:
https://creativecommons.org/licenses/by/4.0/.
The vector map was rasterized to assign classes to SAR image tiles.

The imagery used in the experiments is the Cassini SAR HiSAR Global Mosaic
351 m. Credit the Cassini SAR Team, Jet Propulsion Laboratory and USGS
Astrogeology Science Center.

The separate `initialization-supplement.zip` supplies the original comparison,
catalog and geographic divisions. Its data terms and third-party notices remain
inside that ZIP. This package reads its data without importing its Python code.

The accompanying `LICENSE` applies to this package's Python code. It does not
replace the source data terms above.

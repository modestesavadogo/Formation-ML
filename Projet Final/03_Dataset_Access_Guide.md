# Dataset access guide — Final ML Project

Choose **one track**. Download data only from the original source and cite the dataset version actually used. The project kit does not redistribute raw datasets; students should obtain them from the source pages below.

## Track A — Finance

- **Dataset:** Daily OHLCV Data for 20 S&P 500 Equities and the CBOE Volatility Index (1982–2025).
- **Source:** https://zenodo.org/records/20192822
- **DOI:** https://doi.org/10.5281/zenodo.20192822
- **Format:** ZIP archive containing data and README/manifest.
- **Licence stated by Zenodo:** CC BY 4.0.
- **Task:** choose one equity and predict the sign of its next-session Adjusted Close return.
- **Critical rule:** use a chronological holdout and walk-forward/time-series validation. Do not shuffle dates.

## Track B — Medicine / Biology

- **Dataset:** Breast Cancer Wisconsin (Diagnostic).
- **Source:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **DOI:** https://doi.org/10.24432/C5DW2B
- **Quick access:** `sklearn.datasets.load_breast_cancer(as_frame=True)`.
- **Size:** 569 observations, 30 continuous features, binary diagnosis. The UCI record states that features are computed from digitized fine-needle aspirate images.
- **Target convention:** if using scikit-learn's built-in loader, verify `target_names`; recode malignant as the positive class (`1`).
- **Feature rule:** exclude the sample identifier from the predictors if using the raw UCI table.
- **Caution:** educational classification exercise only; not a clinical diagnostic tool.
- **Licence:** UCI record states CC BY 4.0.

## Track C — Veterinary health

- **Dataset:** Transboundary cattle diseases (ILRI / MELDATA).
- **Source:** https://data.mel.cgiar.org/dataset.xhtml?persistentId=hdl:20.500.11766.1/FK2/8YFO3D
- **Persistent ID:** `hdl:20.500.11766.1/FK2/8YFO3D`
- **Licence stated by MELDATA:** CC BY 4.0.
- **Archive:** `data_published.zip`, containing `cbpp.xlsx`, `fmd.xlsx` and `lsd.xlsx`; each workbook includes a `variable` sheet with definitions.
- **Coverage:** the catalogue reports 3,533 data points collected in Ethiopia during 2012–2019. This is the total stated for the dataset; inspect individual workbook shapes and class distributions yourself.
- **Task:** choose one disease workbook only after inspecting the exact target column and the positive/negative counts.
- **Feature rules:** `Year_region` is categorical by default. `Weight` is described as an observation weighting factor; do not use it as an ordinary predictor by default. Explain whether coordinates are used and discuss spatial generalization.
- **Validation:** stratified split for the core exercise, plus discussion of dependence across regions/years. A group-based holdout is an extension when supported by the data structure.

## Shared technical requirements

- Keep an untouched copy of the raw data and document all transformations.
- Never fit imputers, scalers or encoders on the final test set. Use scikit-learn `Pipeline` and `ColumnTransformer`.
- Required models: `DummyClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, and `KNeighborsClassifier`.
- Finance: chronological test set + walk-forward validation.
- Medicine and Veterinary: stratified holdout + stratified cross-validation; discuss any grouped or spatial/temporal dependence.
- Submit the completed notebook and a 12-minute presentation, followed by 3 minutes of questions.

## Dataset citations

1. Al Ridhawi, M., Haj Ali, M., & Al Osman, H. (2026). *Daily OHLCV Data for 20 S&P 500 Equities and the CBOE Volatility Index (1982–2025).* Zenodo. https://doi.org/10.5281/zenodo.20192822
2. Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic) [Dataset].* UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
3. Gizaw, S., Dione, M., Wieland, B., & Desta, H. (2026). *Transboundary cattle diseases.* MELDATA, Version 1. https://hdl.handle.net/20.500.11766.1/FK2/8YFO3D

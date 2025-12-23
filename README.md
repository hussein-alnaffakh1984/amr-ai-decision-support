# AMR AI Decision Support (Research Demo)

An AI-assisted decision support system for **early antibiotic selection under incomplete microbiological evidence**.
This project demonstrates confidence-aware recommendations with abstention when uncertainty is high.

> ⚠️ **Research/Educational Use Only** — Not a clinical device.

## Features
- Handles **incomplete laboratory data** (missing MIC/typing).
- **Confidence-aware** predictions with an abstain policy.
- **Interpretable** baseline (Logistic Regression) with calibration.
- Simple **Streamlit web app** for interactive testing.

## Methodology (Brief)
1. Load AMR phenotype data (public).
2. Normalize heterogeneous MIC representations.
3. Train a baseline model with preprocessing (imputation + one-hot).
4. Apply **probability calibration** and **confidence thresholding**.
5. Recommend / abstain based on confidence.
6. Explain decisions via feature importance (research setting).

## App
- Built with **Streamlit**.
- Data can be loaded remotely (public AMR portal) or from a CSV.
- Scenarios simulate missing evidence:
  - S0_full
  - S1_no_MIC
  - S2_no_typing
  - S3_basic_only

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

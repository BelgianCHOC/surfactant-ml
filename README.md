# Surfactant SFT–log(c) Predictor

A web app that predicts the surface tension–concentration profile of hydrocarbon surfactants from a SMILES string, based on the paper:

> Seddon, D., Müller, E. A., & Cabral, J. T. (2022). **Machine learning hybrid approach for the prediction of surface tension profiles of hydrocarbon surfactants in aqueous solution.** *Journal of Colloid and Interface Science*, 625, 328–339.

## How it works

The app uses a **hybrid ML + physics** approach:

1. **Molecular descriptors** are computed from the input SMILES using RDKit and mordred
2. **Three XGBoost models** predict the Szyszkowski equation parameters:
   - Γ_max — maximum surface excess concentration (mol/m²)
   - K_L — Langmuir adsorption constant (m³/mol)
   - CMC — critical micelle concentration (M)
3. The **Szyszkowski equation** reconstructs the full SFT–log(c) curve:

```
γ = γ₀ − R·T·Γ_max·ln(1 + K_L·c)
```

Above the CMC, the surface is saturated and surface tension stays flat.

## Project structure

```
app.py              # Streamlit web app
szyszkowski.py      # Szyszkowski equation and curve generation
features.py         # SMILES → mordred molecular descriptors
train_models.py     # Download dataset, train and save XGBoost models
requirements.txt    # Python dependencies
models/             # Saved XGBoost model files (.pkl)
```

## Getting started

**1. Clone the repo and create a virtual environment:**
```bash
git clone https://github.com/BelgianCHOC/surfactant-ml.git
cd surfactant-ml
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Train the models** (downloads the 154-molecule dataset and takes ~3 minutes):
```bash
python3 train_models.py
```

**4. Run the app:**
```bash
streamlit run app.py
```

## Example

Enter the SMILES for sodium dodecyl sulphate (SDS):
```
CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]
```

The app will display the predicted Γ_max, K_L, and CMC, and plot the SFT–log(c) isotherm with a dashed line marking the CMC.

## Notes

- The models were retrained using open-source mordred descriptors rather than the commercial AlvaDesc/SIRMS/CDK tools used in the original paper. Predictions are approximate.
- Training data: 154 hydrocarbon surfactants at 20–30°C from the original paper's dataset.
- Temperature assumed: 25°C (298.15 K).

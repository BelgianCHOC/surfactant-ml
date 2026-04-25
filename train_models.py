import os
import requests
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle
from features import smiles_to_descriptors

DATA_URL = "https://github.com/DaleSeddon-GitHub/SFT-logc-Prediction/raw/main/MasterSMILES-SFT-logc.xlsx"
MODELS_DIR = "models"
CACHE_FILE = "descriptor_cache.csv"

# download the dataset

def download_data():
    print("Downloading training data...")
    r = requests.get(DATA_URL)
    with open("master.xlsx", "wb") as f:
        f.write(r.content)
    df = pd.read_excel("master.xlsx")
    print(f"  {len(df)} molecules loaded")
    return df

# compute mordred descriptors for every molecule

def compute_all_descriptors(smiles_list):
    if os.path.exists(CACHE_FILE):
        print("Loading cached descriptors...")
        return pd.read_csv(CACHE_FILE)
        
    print(f"Computing descriptors for {len(smiles_list)} molecules (this takes ~3 min)...")
    rows = []
    for i, smi in enumerate(smiles_list):
        try:
            row = smiles_to_descriptors(smi)
            rows.append(row)
        except Exception as e:
            print(f"  Skipping {smi}: {e}")
            rows.append(None)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(smiles_list)} done")

    df = pd.concat([r for r in rows if r is not None], ignore_index=True)
    df.to_csv(CACHE_FILE, index=False)
    return df

# clean the descriptor matrix

def clean_features(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1)
    constant_cols = X.columns[X.std() == 0]
    X = X.drop(columns=constant_cols)
    return X

# train one model and save it

def train_and_save(X, y, name):
    mask = y.notna()
    X_fit = X[mask]
    y_fit = y[mask]
    print(f"\nTraining {name} on {len(y_fit)} samples")

    model = XGBRegressor(n_estimators=300, learning_rate=0.05,
                         max_depth=4, subsample=0.8,
                         colsample_bytree=0.8, random_state=42)
    model.fit(X_fit, y_fit)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(f"{MODELS_DIR}/{name}.pkl", "wb") as f:
        pickle.dump((model, list(X_fit.columns)), f)
    print(f"  Saved to models/{name}.pkl")

# main function that wires it all together

def main():
    df = download_data()

    desc_df = compute_all_descriptors(list(df["SMILES"]))
    desc_df = clean_features(desc_df)

    targets = {
        "gamma_max": df["Maximum Surface Excess Concentration (x10^6)"].reset_index(drop=True),
        "log_kl": df["Log(Langmuir Constant)"].reset_index(drop=True),
        "log_cmc": df["LogCMC"].reset_index(drop=True),
    }

    valid_idx = desc_df.index
    for name, y in targets.items():
        train_and_save(desc_df, y.loc[valid_idx], name)

    print("\nALL models trained and saved.")

if __name__ == "__main__":
    main()
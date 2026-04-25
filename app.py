import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pickle
import pandas as pd
from features import smiles_to_descriptors
from szyszkowski import compute_sft_profile

# load all three models at startup

@st.cache_resource
def load_models():
    models = {}
    for name in ["gamma_max", "log_kl", "log_cmc"]:
        with open(f"models/{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    return models

# predict the three parameters from a SMILES string

def predict_parameters(smiles, models):
    desc = smiles_to_descriptors(smiles)

    results = {}
    for name, (model, columns) in models.items():
        X = desc.reindex(columns=columns, fill_value=0)
        X = X.replace([float("inf"), float("-inf")], 0).fillna(0)
        results[name] = float(model.predict(X)[0])

    gamma_max = results["gamma_max"] * 1e-6
    kl        = 10 ** results["log_kl"]
    cmc       = 10 ** results["log_cmc"]
    return gamma_max, kl, cmc

# the UI

st.title("Surfactant SFT-log(c) Predictor")
st.markdown("Enter a hydrocarbon surfactant SMILES to predict its surface tension profile.")

smiles_input = st.text_input("SMILES", placeholder="e.g. CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]")

models = load_models()

if smiles_input:
    with st.spinner("Computing descriptors and predicting..."):
        try:
            gamma_max, kl, cmc = predict_parameters(smiles_input, models)

            st.subheader("Predicted Parameters")
            col1, col2, col3 = st.columns(3)
            col1.metric("Γ_max (mol/m²)", f"{gamma_max:.2e}")
            col2.metric("K_L (m³/mol)",   f"{kl:.2e}")
            col3.metric("CMC (M)",        f"{cmc:.2e}")

            log_c, sft = compute_sft_profile(gamma_max, kl, cmc)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=log_c, y=sft, mode="lines",
                                     line=dict(color="royalblue", width=2)))
            fig.add_vline(x=np.log10(cmc), line_dash="dash",
                          line_color="red", annotation_text="CMC")
            fig.update_layout(
                xaxis_title="log₁₀(c) [c in mol/L]",
                yaxis_title="Surface Tension (mN/m)",
                title="SFT - log(c) Profile",
                yaxis=dict(range=[20, 75]),
                template="simple_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
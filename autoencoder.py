"""
models/autoencoder.py
─────────────────────
Build, train, and evaluate a Keras Autoencoder for anomaly detection.
Trains on LEGITIMATE transactions only — fraud = high reconstruction error.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    classification_report
)
from utils.helpers import BLUE, RED, AMBER, plot_confusion

SEED = 42
tf.random.set_seed(SEED)


def build_autoencoder(input_dim: int, bottleneck_dim: int = 14):
    """Return a compiled Keras autoencoder model."""
    inp = keras.Input(shape=(input_dim,))
    x   = layers.Dense(32, activation="relu")(inp)
    x   = layers.Dropout(0.1)(x)
    x   = layers.Dense(bottleneck_dim, activation="relu", name="bottleneck")(x)
    x   = layers.Dense(32, activation="relu")(x)
    x   = layers.Dropout(0.1)(x)
    out = layers.Dense(input_dim, activation="linear")(x)
    model = keras.Model(inp, out, name="Autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(X_train_legit, X_test, y_test,
                      epochs=20, bottleneck_dim=14):
    """
    Train autoencoder on legit-only data.
    Stores results in session_state and renders metrics.
    """
    INPUT_DIM = X_train_legit.shape[1]

    bar = st.progress(0, text="Autoencoder: building model...")
    ae  = build_autoencoder(INPUT_DIM, bottleneck_dim)

    bar.progress(15, text="Autoencoder: training on legit transactions...")
    history = ae.fit(
        X_train_legit, X_train_legit,
        epochs=epochs,
        batch_size=256,
        validation_split=0.1,
        shuffle=True,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
        ]
    )

    bar.progress(80, text="Autoencoder: computing reconstruction errors...")
    X_rec   = ae.predict(X_test, verbose=0)
    ae_sc   = np.mean(np.power(X_test - X_rec, 2), axis=1)

    # Threshold: 95th percentile of legit reconstruction error
    leg_rec = ae.predict(X_test[y_test == 0], verbose=0)
    leg_err = np.mean(np.power(X_test[y_test == 0] - leg_rec, 2), axis=1)
    thresh  = np.percentile(leg_err, 95)
    ae_pred = (ae_sc > thresh).astype(int)

    # Persist
    st.session_state.ae_model     = ae
    st.session_state.ae_threshold = thresh
    st.session_state.ae_pred      = ae_pred
    st.session_state.ae_scores    = ae_sc

    bar.progress(100, text="Autoencoder: done ✓")

    # Metrics
    f1  = f1_score(y_test, ae_pred)
    auc = roc_auc_score(y_test, ae_sc)
    ap  = average_precision_score(y_test, ae_sc)
    st.success(f"✅ Autoencoder — F1: `{f1:.4f}` | ROC-AUC: `{auc:.4f}` | AUC-PR: `{ap:.4f}`")

    # Training loss plot
    col_l, col_r = st.columns(2)
    with col_l:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(history.history["loss"],     color=BLUE,  linewidth=2, label="Train Loss")
        ax.plot(history.history["val_loss"], color=AMBER, linewidth=2,
                linestyle="--", label="Val Loss")
        ax.set_title("Autoencoder Training Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.legend(); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col_r:
        fig = plot_confusion(y_test, ae_pred,
                             "Autoencoder — Confusion Matrix", RED)
        st.pyplot(fig); plt.close()

    # Reconstruction error distribution
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(ae_sc[y_test == 0], bins=80, alpha=0.60, color=BLUE,
            label="Legitimate", density=True)
    ax.hist(ae_sc[y_test == 1], bins=30, alpha=0.75, color=RED,
            label="Fraud", density=True)
    ax.axvline(thresh, color=AMBER, linestyle="--", linewidth=1.5,
               label=f"Threshold = {thresh:.5f}")
    ax.set_xlim(0, np.percentile(ae_sc, 99.5))
    ax.set_title("Autoencoder — Reconstruction Error Distribution")
    ax.set_xlabel("MSE Reconstruction Error"); ax.set_ylabel("Density")
    ax.legend(); fig.tight_layout()
    st.pyplot(fig); plt.close()

    return {"Model": "Autoencoder",
            "F1": round(f1, 4), "ROC-AUC": round(auc, 4), "AUC-PR": round(ap, 4)}

"""
models/isolation_forest.py
──────────────────────────
Train and evaluate an Isolation Forest for anomaly detection.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    classification_report, precision_recall_curve
)
from utils.helpers import BLUE, RED, AMBER, plot_confusion, plot_score_dist

SEED = 42


def train_isolation_forest(X_train, X_test, y_test,
                            n_estimators=100, contamination=0.0017):
    """
    Train Isolation Forest and store results in session_state.
    Renders progress + result cards directly in the Streamlit page.
    """
    bar = st.progress(0, text="Isolation Forest: fitting...")
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=SEED,
        n_jobs=-1
    )
    iso.fit(X_train)

    bar.progress(70, text="Isolation Forest: predicting...")
    iso_pred_raw = iso.predict(X_test)
    iso_pred     = (iso_pred_raw == -1).astype(int)
    iso_scores   = -iso.score_samples(X_test)

    # Persist
    st.session_state.iso_model  = iso
    st.session_state.iso_pred   = iso_pred
    st.session_state.iso_scores = iso_scores

    bar.progress(100, text="Isolation Forest: done ✓")

    # Metrics
    f1  = f1_score(y_test, iso_pred)
    auc = roc_auc_score(y_test, iso_scores)
    ap  = average_precision_score(y_test, iso_scores)
    st.success(f"✅ Isolation Forest — F1: `{f1:.4f}` | ROC-AUC: `{auc:.4f}` | AUC-PR: `{ap:.4f}`")

    c1, c2 = st.columns(2)
    with c1:
        fig = plot_confusion(y_test, iso_pred,
                             "Isolation Forest — Confusion Matrix", BLUE)
        st.pyplot(fig); plt.close()
    with c2:
        fig = plot_score_dist(iso_scores, y_test,
                              "Isolation Forest — Anomaly Scores")
        st.pyplot(fig); plt.close()

    return {"Model": "Isolation Forest",
            "F1": round(f1, 4), "ROC-AUC": round(auc, 4), "AUC-PR": round(ap, 4)}

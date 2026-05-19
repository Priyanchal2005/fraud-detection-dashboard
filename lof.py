"""
models/lof.py
─────────────
Local Outlier Factor — density-based anomaly detection.
NOTE: LOF is transductive (no separate fit/predict); runs on test set directly.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score
)
from utils.helpers import PURPLE, plot_confusion, plot_score_dist

SEED = 42
LOF_SAMPLE = 10_000   # LOF is O(n²); sample for speed


def train_lof(X_test, y_test, n_neighbors=20, contamination=0.0017):
    """
    Run LOF on a sample of the test set.
    Stores results in session_state and renders metrics.
    """
    # Sample for speed
    np.random.seed(SEED)
    sample_idx = np.random.choice(len(X_test), min(LOF_SAMPLE, len(X_test)),
                                  replace=False)
    X_s = X_test[sample_idx]
    y_s = y_test[sample_idx]

    bar = st.progress(0, text=f"LOF: fitting on {len(X_s):,} samples...")
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1
    )
    lof_raw    = lof.fit_predict(X_s)
    lof_pred   = (lof_raw == -1).astype(int)
    lof_scores = -lof.negative_outlier_factor_

    # Persist
    st.session_state.lof_pred   = lof_pred
    st.session_state.lof_scores = lof_scores
    st.session_state.y_test_lof = y_s

    bar.progress(100, text="LOF: done ✓")

    # Metrics
    f1  = f1_score(y_s, lof_pred)
    auc = roc_auc_score(y_s, lof_scores)
    ap  = average_precision_score(y_s, lof_scores)
    st.success(f"✅ LOF — F1: `{f1:.4f}` | ROC-AUC: `{auc:.4f}` | AUC-PR: `{ap:.4f}`")

    c1, c2 = st.columns(2)
    with c1:
        fig = plot_confusion(y_s, lof_pred, "LOF — Confusion Matrix", PURPLE)
        st.pyplot(fig); plt.close()
    with c2:
        fig = plot_score_dist(lof_scores, y_s, "LOF — Anomaly Scores")
        st.pyplot(fig); plt.close()

    return {"Model": "LOF",
            "F1": round(f1, 4), "ROC-AUC": round(auc, 4), "AUC-PR": round(ap, 4)}

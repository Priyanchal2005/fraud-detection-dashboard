"""
app.py  ←  Entry point
──────────────────────
Run:  streamlit run app.py

Project structure
─────────────────
fraud_detection_project/
├── app.py                  ← you are here (entry point)
├── preprocessing.py        ← data loading & scaling
├── requirements.txt
├── README.md
├── models/
│   ├── isolation_forest.py ← Isolation Forest logic
│   ├── autoencoder.py      ← Keras Autoencoder logic
│   └── lof.py              ← Local Outlier Factor logic
├── pages/
│   ├── eda.py              ← EDA tab
│   ├── compare.py          ← Model comparison tab
│   └── tsne.py             ← t-SNE visualization tab
└── utils/
    └── helpers.py          ← colours, theme, shared plots
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from preprocessing           import load_data, preprocess
from models.isolation_forest import train_isolation_forest
from models.autoencoder      import train_autoencoder
from models.lof              import train_lof
from views                   import eda, compare, tsne
from utils.helpers           import apply_dark_theme, BLUE, RED

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_dark_theme()

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]         { font-family:'IBM Plex Sans',sans-serif; }
.stApp                           { background-color:#0a0e1a; color:#e2e8f0; }
[data-testid="stSidebar"]        { background-color:#0f1525 !important; border-right:1px solid #1e2d4a; }
[data-testid="stSidebar"] *      { color:#94a3b8 !important; }
[data-testid="metric-container"] { background:#111827; border:1px solid #1e2d4a; border-radius:8px; padding:16px !important; }
[data-testid="stMetricValue"]    { font-family:'IBM Plex Mono',monospace !important; font-size:28px !important; color:#38bdf8 !important; }
[data-testid="stMetricLabel"]    { color:#64748b !important; font-size:11px !important; text-transform:uppercase; letter-spacing:.08em; }
.stTabs [data-baseweb="tab-list"]{ background-color:#0f1525; border-bottom:1px solid #1e2d4a; gap:0; }
.stTabs [data-baseweb="tab"]     { background:transparent; color:#64748b !important; font-size:13px; padding:10px 20px; border-radius:0 !important; font-family:'IBM Plex Mono',monospace; }
.stTabs [aria-selected="true"]   { background-color:#0a0e1a !important; color:#38bdf8 !important; border-bottom:2px solid #38bdf8 !important; }
.stButton button { background:linear-gradient(135deg,#1d4ed8,#0369a1) !important; color:white !important; border:none !important; border-radius:6px !important; font-family:'IBM Plex Mono',monospace !important; font-size:13px !important; padding:10px 24px !important; }
.stButton button:hover { transform:translateY(-1px); box-shadow:0 4px 15px rgba(56,189,248,.3) !important; }
[data-testid="stDataFrame"]      { border:1px solid #1e2d4a !important; border-radius:8px; }
.stProgress > div > div          { background-color:#38bdf8 !important; }
h1,h2,h3  { font-family:'IBM Plex Mono',monospace; color:#e2e8f0 !important; }
hr        { border-color:#1e2d4a !important; }
code      { font-family:'IBM Plex Mono',monospace !important; background:#111827 !important; color:#38bdf8 !important; padding:2px 6px; border-radius:3px; }
.stSelectbox>div>div { background-color:#111827 !important; border-color:#1e2d4a !important; color:#e2e8f0 !important; }
.streamlit-expanderHeader { background-color:#111827 !important; color:#94a3b8 !important; border:1px solid #1e2d4a !important; border-radius:6px !important; font-size:13px !important; }
[data-testid="stFileUploader"] { background:#111827 !important; border:1px dashed #1e2d4a !important; border-radius:8px !important; }
::-webkit-scrollbar       { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#0a0e1a; }
::-webkit-scrollbar-thumb { background:#1e2d4a; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────
for key in ["df","X_train","X_test","y_train","y_test","X_train_legit","scaler",
            "iso_model","ae_model","ae_threshold",
            "iso_pred","iso_scores","ae_pred","ae_scores",
            "lof_pred","lof_scores","y_test_lof","results_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detector")
    st.markdown("---")
    uploaded = st.file_uploader("Upload creditcard.csv", type=["csv"])
    if uploaded:
        with st.spinner("Loading & preprocessing..."):
            df = load_data(uploaded)
            st.session_state.df = df
            (X_train, X_test, y_train, y_test,
             X_train_legit, scaler) = preprocess(df)
            st.session_state.X_train       = X_train
            st.session_state.X_test        = X_test
            st.session_state.y_train       = y_train
            st.session_state.y_test        = y_test
            st.session_state.X_train_legit = X_train_legit
            st.session_state.scaler        = scaler
        st.success(f"Loaded {len(df):,} rows")

    st.markdown("---")
    st.markdown("**Model Settings**")
    contamination = st.slider("Contamination",    0.001, 0.020, 0.0017, 0.0001, format="%.4f")
    n_estimators  = st.slider("IF: n_estimators",    50,   300,   100,   50)
    ae_epochs     = st.slider("AE: epochs",            5,    50,    20,    5)
    ae_bottleneck = st.slider("AE: bottleneck dim",    4,    28,    14,    2)
    lof_neighbors = st.slider("LOF: n_neighbors",     10,    50,    20,    5)
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#334155;'>INT 396 · Unsupervised ML</div>",
                unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(
        "<h1 style='font-size:26px;margin-bottom:4px;'>🛡️ Fraudulent Transaction Detection</h1>"
        "<p style='color:#475569;font-size:13px;font-family:IBM Plex Mono,monospace;'>"
        "Unsupervised ML · Isolation Forest · Autoencoder · LOF</p>",
        unsafe_allow_html=True
    )
with col_h2:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.metric("Dataset",    f"{len(df):,} rows")
        st.metric("Fraud rate", f"{df['Class'].mean()*100:.3f}%")
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  EDA", "🧠  Train Models", "🎯  Predictions", "⚖️  Compare", "🔮  t-SNE View"
])

# ── TAB 1 · EDA ───────────────────────────────────────────────
with tab1:
    if st.session_state.df is None:
        st.info("👈  Upload creditcard.csv in the sidebar to begin.")
    else:
        eda.show(st.session_state.df)

# ── TAB 2 · TRAIN ─────────────────────────────────────────────
with tab2:
    if st.session_state.df is None:
        st.info("👈  Upload creditcard.csv in the sidebar first.")
    else:
        st.markdown("### Train Unsupervised Anomaly Detection Models")
        c1, c2, c3 = st.columns(3)
        run_iso = c1.checkbox("Isolation Forest",    value=True)
        run_ae  = c2.checkbox("Autoencoder",         value=True)
        run_lof = c3.checkbox("Local Outlier Factor",value=True)

        if st.button("🚀  Run Training"):
            results = []
            if run_iso:
                r = train_isolation_forest(
                    st.session_state.X_train, st.session_state.X_test,
                    st.session_state.y_test, n_estimators, contamination)
                results.append(r)
            if run_ae:
                st.markdown("---")
                r = train_autoencoder(
                    st.session_state.X_train_legit, st.session_state.X_test,
                    st.session_state.y_test, ae_epochs, ae_bottleneck)
                results.append(r)
            if run_lof:
                st.markdown("---")
                r = train_lof(
                    st.session_state.X_test, st.session_state.y_test,
                    lof_neighbors, contamination)
                results.append(r)
            if results:
                import pandas as pd
                st.session_state.results_df = pd.DataFrame(results)
                st.success("✅  Done! Go to the Compare tab to see results.")

# ── TAB 3 · PREDICTIONS ───────────────────────────────────────
with tab3:
    st.markdown("### Live Predictions")
    if st.session_state.iso_model is None:
        st.warning("Train at least Isolation Forest first (Tab 2).")
    else:
        mode = st.radio("Data source",
                        ["Use existing test set", "Upload new CSV"], horizontal=True)
        if mode == "Use existing test set":
            X_pred = st.session_state.X_test
            y_true = st.session_state.y_test
        else:
            up2 = st.file_uploader("Upload CSV", type=["csv"], key="pred_upload")
            if up2 is None: st.stop()
            import pandas as pd
            df2 = pd.read_csv(up2)
            sc  = st.session_state.scaler
            df2["Amount_scaled"] = sc.transform(df2[["Amount"]])
            df2["Time_scaled"]   = sc.transform(df2[["Time"]])
            fcols  = [c for c in df2.columns if c not in ["Class","Amount","Time"]]
            X_pred = df2[fcols].values
            y_true = df2["Class"].values if "Class" in df2.columns else None

        model_choice = st.selectbox("Model", ["Isolation Forest","Autoencoder","LOF"])
        if st.button("🔍  Run Predictions"):
            if model_choice == "Isolation Forest":
                scores = -st.session_state.iso_model.score_samples(X_pred)
                preds  = (st.session_state.iso_model.predict(X_pred)==-1).astype(int)
            elif model_choice == "Autoencoder" and st.session_state.ae_model:
                X_rec  = st.session_state.ae_model.predict(X_pred, verbose=0)
                scores = np.mean(np.power(X_pred - X_rec, 2), axis=1)
                preds  = (scores > st.session_state.ae_threshold).astype(int)
            else:
                st.error("Model not trained yet."); st.stop()

            c1,c2,c3 = st.columns(3)
            c1.metric("Total Evaluated",  f"{len(preds):,}")
            c2.metric("Flagged as Fraud", f"{preds.sum():,}", delta=f"{preds.mean()*100:.2f}%")
            c3.metric("Legitimate",       f"{(preds==0).sum():,}")

            st.markdown("#### Top 50 Flagged Transactions")
            import pandas as pd
            rdf = pd.DataFrame({
                "Index":         np.arange(len(preds)),
                "Anomaly Score": scores.round(6),
                "Prediction":    ["🔴 FRAUD" if p else "🟢 Legit" for p in preds],
            })
            if y_true is not None:
                rdf["True Label"] = ["Fraud" if y else "Legit" for y in y_true]
            st.dataframe(
                rdf[rdf["Prediction"]=="🔴 FRAUD"]
                .sort_values("Anomaly Score", ascending=False).head(50),
                use_container_width=True
            )
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.hist(scores[preds==0], bins=60, alpha=0.65, color=BLUE,
                    label="Predicted Legit", density=True)
            ax.hist(scores[preds==1], bins=30, alpha=0.75, color=RED,
                    label="Predicted Fraud", density=True)
            ax.set_title(f"{model_choice} — Score Distribution")
            ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Density"); ax.legend()
            fig.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 4 · COMPARE ───────────────────────────────────────────
with tab4:
    compare.show()

# ── TAB 5 · t-SNE ─────────────────────────────────────────────
with tab5:
    tsne.show()

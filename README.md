# 🛡️ Fraud Detection Dashboard
### INT 396 — Unsupervised Machine Learning Project

A Streamlit-based dashboard for detecting fraudulent credit card transactions using multiple Unsupervised Machine Learning techniques.

---

# 🚀 Features

- 📊 Exploratory Data Analysis (EDA)
- 🧠 Train Multiple Fraud Detection Models
- 🎯 Predict Fraudulent Transactions
- ⚖️ Compare Model Performance
- 🔮 Interactive t-SNE Visualization
- 📈 Precision-Recall Curves & Metrics

---

# 🧠 Machine Learning Models Used

## 1. Isolation Forest
Detects anomalies by isolating rare observations using random trees.

## 2. Autoencoder (TensorFlow/Keras)
Learns patterns of normal transactions and identifies fraud using reconstruction error.

## 3. Local Outlier Factor (LOF)
Density-based anomaly detection algorithm for identifying local deviations.

---

# 📂 Dataset

Dataset Source:  
:contentReference[oaicite:0]{index=0}

### Dataset Details
- Total Transactions: **284,807**
- Fraud Cases: **492**
- Fraud Percentage: **0.17%**
- Features `V1–V28` are PCA transformed for confidentiality.

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

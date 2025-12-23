# Credit Card Fraud Detection - Machine Learning Project

Proyek machine learning untuk mendeteksi transaksi penipuan kartu kredit menggunakan XGBoost dengan tracking eksperimen melalui MLflow dan DagsHub.

## 📋 Deskripsi

Proyek ini mengimplementasikan dua pendekatan modeling:
1. **Baseline Model** - Model XGBoost standar untuk benchmark performa
2. **Optimized Model** - Model XGBoost dengan hyperparameter tuning menggunakan GridSearchCV dan validasi silang berulang (Repeated Stratified K-Fold)

## 🎯 Fitur Utama

- ✅ Preprocessing data dengan StandardScaler dalam pipeline
- ✅ Pencegahan data leakage dengan proper train-test split
- ✅ Hyperparameter tuning dengan GridSearchCV
- ✅ Evaluasi model dengan Repeated Stratified K-Fold CV (5 folds × 3 repeats)
- ✅ Tracking eksperimen dengan MLflow dan DagsHub
- ✅ Visualisasi performa: Confusion Matrix, Precision-Recall Curve, Feature Importance
- ✅ Logging metrics: F1, Precision, Recall, AUPRC (beserta mean & std dari CV)

## 📦 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/Dekhsa/SMSML_Muhamad-Dekhsa-Afnan.git
cd SMSML_Muhamad-Dekhsa-Afnan
```

### 2. Setup Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# atau
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r Membangun_model/requirements.txt
```

## 🚀 Cara Menjalankan

### Baseline Model
```bash
python Membangun_model/modelling.py
```

### Optimized Model dengan Tuning
```bash
python Membangun_model/modelling_tuning.py
```

## 📊 Struktur Project

```
SMSML_Muhamad-Dekhsa-Afnan/
├── .gitignore
├── README.md
├── Membangun_model/
│   ├── creditcardfraud_preprocessing.csv
│   ├── modelling.py              # Baseline model
│   ├── modelling_tuning.py       # Optimized model dengan tuning
│   ├── requirements.txt          # Dependencies
│   └── artifacts/                # Visualisasi (confusion matrix, PR curve, feature importance)
└── .venv/                        # Virtual environment (ignored)
```

## 📈 Hasil Eksperimen

### Baseline Model
- F1 Score: ~1.0000
- Precision: ~1.0000
- Recall: ~1.0000
- AUPRC: ~1.0000

### Optimized Model (dengan CV)
- **Holdout Test Set**: F1=1.0, Precision=1.0, Recall=1.0, AUPRC=1.0
- **Cross-Validation (5×3)**:
  - F1 mean: 0.984 ± 0.015
  - Precision mean: 1.000 ± 0.000
  - Recall mean: 0.969 ± 0.029
  - AUPRC mean: 0.997 ± 0.006

### Best Hyperparameters
```python
{
    'model__learning_rate': 0.1,
    'model__max_depth': 3,
    'model__n_estimators': 200,
    'model__scale_pos_weight': 5.0
}
```

## 🔍 Artifacts di MLflow

Setiap run menghasilkan:
1. **Model folder** (`model/`):
   - `MLmodel` - Metadata model
   - `model.pkl` - Serialized pipeline
   - `requirements.txt` / `conda.yaml` - Environment spec

2. **Visualisasi** (`.png`):
   - `confusion_matrix.png` - Confusion matrix
   - `precision_recall_curve.png` - PR curve
   - `feature_importance.png` - Feature importance (hanya di tuning)
   - `cv_metrics_hist.png` - Distribusi CV scores (hanya di tuning)

3. **Dataset**:
   - `data/creditcardfraud_preprocessing.csv`
   - `data/dataset_profile.json`
   - Dataset registry (train/test splits)

4. **Metrics**:
   - Holdout: `f1_score`, `precision`, `recall`, `auprc`
   - CV: `cv_f1_mean`, `cv_f1_std`, `cv_precision_mean`, `cv_precision_std`, dll.

## 🛠️ Teknologi yang Digunakan

- **Python 3.13**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning framework
- **XGBoost** - Gradient boosting
- **MLflow** - Experiment tracking
- **DagsHub** - Remote tracking server
- **matplotlib & seaborn** - Visualisasi

## 📊 MLOps & Production Monitoring

Proyek ini dilengkapi dengan infrastruktur monitoring lengkap:

### 🏗️ Architecture

```
Flask Inference API (Port 5000)
    ↓
MLflow Model Registry (Run ID: 54050a2976204c79bae1a103b41e6753)
    ↓
Prometheus Exporter (Port 8000)
    ↓
Prometheus Time-Series DB (Port 9090)
    ↓
Grafana Dashboard (Port 3000) + Alert Rules
```

### 📁 Monitoring Components

1. **7.inference.py** - Flask API untuk real-time prediction
   - Endpoint: `POST /predict`
   - Input: Transaction features (amount, velocity, device_trust_score, etc.)
   - Output: `{fraud: bool, confidence: float}`
   - Metrics tracking ke `metrics.json`

2. **3.prometheus_exporter.py** - Metrics exporter
   - 12+ Prometheus metrics:
     - `fraud_detection_total_fraud_detected` (gauge)
     - `fraud_detection_rate_percent` (gauge)
     - `fraud_detection_model_confidence_score` (gauge)
     - `fraud_detection_prediction_latency_seconds` (histogram)
     - `system_cpu_usage_percent` (gauge)
     - `system_memory_usage_percent` (gauge)
     - Dan lainnya...

3. **2.prometheus.yml** - Prometheus configuration
   - Scrape interval: 5 detik
   - Targets: `localhost:8000/metrics`

4. **Grafana Dashboard**
   - Real-time fraud detection visualization
   - 3 Alert Rules:
     * Fraud Rate > 10%
     * Total Fraud Detected > 50
     * Model Confidence Score < 0.7
   - Contact Points: Discord/Slack/Webhook

5. **scheduler_api_trigger.py** - Real-time transaction simulation
   - Generates realistic transactions every 2 seconds
   - 95% normal, 5% suspicious distribution
   - Logs results dengan confidence scores

### 🚀 Menjalankan Stack Monitoring

```bash
# Terminal 1: Start Flask API
cd Monitoring\ dan\ Logging
python 7.inference.py

# Terminal 2: Start Prometheus Exporter
python 3.prometheus_exporter.py

# Terminal 3: Start Prometheus
prometheus --config.file=2.prometheus.yml

# Terminal 4: Start Grafana
docker run -d -p 3000:3000 grafana/grafana

# Terminal 5: Start Transaction Scheduler
python scheduler_api_trigger.py
```

### 📈 Testing Alerts

Trigger fraud detection alerts dengan suspisious transactions:

```bash
python trigger_fraud_batch.py  # 100+ transactions
# Result: Fraud Rate → 77.2%, Total Fraud → 105
# Alerts trigger → Discord/Slack notifications sent
```

## 📝 Catatan Penting

1. **Data Leakage Prevention**:
   - Kolom proxy (transaction_id) dihapus sebelum modeling
   - StandardScaler fit hanya pada training set melalui Pipeline
   - Stratified splitting untuk menjaga proporsi kelas

2. **Cross-Validation**:
   - Repeated Stratified K-Fold digunakan untuk mengurangi variance risk
   - Semua score distributions disimpan dalam `cv_scores.json`

3. **MLflow Tracking**:
   - Semua eksperimen ter-track di DagsHub
   - View experiments: https://dagshub.com/Dekhsa/SMSML_Muhamad-Dekhsa-Afnan.mlflow

4. **Production Monitoring**:
   - Model confidence scores dipantau real-time
   - Fraud rate metrics terupdate setiap 5 detik
   - Alerts dapat dikonfigurasi threshold-nya di Grafana

## 👤 Author

**Muhamad Dekhsa Afnan**
- GitHub: [@Dekhsa](https://github.com/Dekhsa)
- DagsHub: [Dekhsa](https://dagshub.com/Dekhsa)

## 📄 License

Proyek ini dibuat untuk keperluan pembelajaran dan evaluasi Machine Learning.

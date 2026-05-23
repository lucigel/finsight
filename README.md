# FinSight 📈

> **Finance ML Platform** — Stock Price Prediction + Fraud Detection  
> Built on Databricks Community Edition with Delta Lake + MLflow

---

## Overview

FinSight là một **end-to-end Machine Learning platform** mô phỏng hệ thống phân tích tài chính thực tế. Project giải quyết 2 bài toán ML phổ biến trong ngành Finance:

| Bài toán | Approach | Model |
|---|---|---|
| 📈 Stock Price Prediction | Dự đoán chiều giá 7 ngày tới | XGBoost, Prophet |
| 🚨 Fraud Detection | Phát hiện giao dịch bất thường | Isolation Forest, Autoencoder |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
│                                                                     │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │
│  │  yfinance   │  │  Alpha Vantage   │  │   Synthetic Generator  │ │
│  │             │  │                  │  │                        │ │
│  │ OHLCV daily │  │ Indicators, news │  │ 100k transactions      │ │
│  │ 5 stocks    │  │ sentiment        │  │ (fraud labeled)        │ │
│  └──────┬──────┘  └────────┬─────────┘  └───────────┬────────────┘ │
└─────────┼──────────────────┼────────────────────────┼─────────────┘
          │                  │                        │
          └──────────────────┼────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│              01_data_ingestion.ipynb                                │
│         Fetch → Validate schema → Checkpoint Bronze                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    DELTA LAKE (DBFS)                                │
│                                                                     │
│  /bronze/  →  /silver/  →  /gold/                                  │
│  Raw data     Features      Predictions                             │
│  append-only  engineered    model-ready                             │
└──────────┬────────────────────────────────┬────────────────────────┘
           ↓                                ↓
┌──────────────────────┐        ┌───────────────────────────┐
│ 02_stock_features    │        │ 03_fraud_features          │
│                      │        │                           │
│ RSI, MACD, Bollinger │        │ Velocity, patterns        │
│ Rolling stats        │        │ Geographic anomaly        │
│ Lag features         │        │ Merchant deviation        │
└──────────┬───────────┘        └──────────────┬────────────┘
           ↓                                   ↓
┌──────────────────────┐        ┌───────────────────────────┐
│ 04_stock_model       │        │ 05_fraud_model             │
│                      │        │                           │
│ XGBoost classifier   │        │ Isolation Forest          │
│ Prophet forecast     │        │ Autoencoder               │
│ Backtest strategy    │        │ Threshold optimization    │
└──────────┬───────────┘        └──────────────┬────────────┘
           └──────────────┬─────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    MLflow Model Registry                            │
│         Staging → Production promotion workflow                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│                    06_batch_predict.ipynb                        │
│              Load Production model → Predict → Gold              │
└───────────────────────────────┬──────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      serving/api.py                             │
│                                                                 │
│  GET  /predict/stock/{ticker}  →  next 7-day direction          │
│  POST /predict/fraud           →  anomaly score                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool | Ghi chú |
|---|---|---|
| Platform | Databricks Community Edition | Free, no credit card |
| Storage | Delta Lake (DBFS) | ACID, time travel, schema enforcement |
| Processing | PySpark + pandas | Spark 3.5 built-in |
| ML Tracking | MLflow | Built-in Databricks |
| Stock models | XGBoost, Prophet, scikit-learn | Prediction + forecasting |
| Fraud models | scikit-learn, PyTorch | Anomaly detection |
| Feature Eng | pandas-ta | Technical indicators |
| Data fetch | yfinance, requests | Free APIs |
| Serving | FastAPI + uvicorn | Local demo |
| Testing | pytest | Unit tests cho features + models |

---

## Project Structure

```
finsight/
├── notebooks/
│   ├── 01_data_ingestion.ipynb       # Fetch + validate + Bronze Delta
│   ├── 02_stock_features.ipynb       # Technical indicators → Silver
│   ├── 03_fraud_features.ipynb       # Transaction features → Silver
│   ├── 04_stock_model.ipynb          # Train + MLflow + Registry
│   ├── 05_fraud_model.ipynb          # Train + MLflow + Registry
│   └── 06_batch_predict.ipynb        # Load model → predict → Gold
├── src/
│   ├── features/
│   │   ├── stock_indicators.py       # RSI, MACD, Bollinger functions
│   │   └── fraud_features.py         # Velocity, pattern features
│   ├── models/
│   │   ├── stock_model.py            # XGBoost + Prophet wrappers
│   │   └── fraud_model.py            # IsolationForest + Autoencoder
│   └── utils/
│       ├── delta_utils.py            # Read/write Delta helpers
│       └── mlflow_utils.py           # MLflow logging helpers
├── serving/
│   └── api.py                        # FastAPI prediction endpoints
├── data/
│   └── generate_transactions.py      # Synthetic transaction generator
├── tests/
│   ├── test_features.py
│   └── test_models.py
├── docs/
│   ├── model_card_stock.md           # Stock model documentation
│   ├── model_card_fraud.md           # Fraud model documentation
│   └── analytics_requirements.md    # Business requirements → DE analysis
├── .env.example                      # API keys template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Phases

### ✅ Phase 1 — Setup & Data Ingestion
- Databricks cluster + library setup
- Fetch OHLCV data cho 5 cổ phiếu (VNM.VN, HPG.VN, AAPL, GOOGL, MSFT)
- Generate 100k synthetic transactions
- Bronze Delta tables

### ✅ Phase 2 — Feature Engineering
- Technical indicators: RSI (14), MACD (12/26/9), Bollinger Bands (20)
- Rolling statistics: mean/std (5, 10, 20, 60 days)
- Lag features: 1, 3, 5, 7 days
- Transaction velocity, merchant deviation, geographic anomaly
- Silver Delta tables

### ✅ Phase 3 — Model Training + MLflow
- Stock: XGBoost classifier (up/flat/down) + Prophet forecast
- Fraud: Isolation Forest + Autoencoder reconstruction error
- MLflow experiment tracking: params, metrics, artifacts
- Model Registry: Staging → Production workflow

### ✅ Phase 4 — Batch Prediction + Serving
- Daily batch predictions → Gold Delta tables
- FastAPI local endpoint cho stock + fraud prediction

### ✅ Phase 5 — Analysis + Documentation
- Strategy backtest: model vs buy-and-hold baseline
- Fraud analysis: precision/recall at different thresholds
- Model cards: limitations, intended use, evaluation results

---

## Data Sources

| Source | Data | API Key cần? | Rate limit |
|---|---|---|---|
| yfinance | OHLCV daily prices | ❌ Không | Không giới hạn |
| Alpha Vantage | Economic indicators | ✅ Free key | 5 req/phút |
| SEC EDGAR | Financial statements | ❌ Không | 10 req/giây |
| Synthetic | Transaction data | ❌ Không | N/A |

---

## ML Metrics

### Stock Prediction
| Metric | Ý nghĩa |
|---|---|
| Accuracy | Tỷ lệ dự đoán đúng chiều giá |
| F1 (macro) | Balanced metric cho 3 classes (up/flat/down) |
| Sharpe Ratio | Return / Risk của strategy backtest |
| Max Drawdown | Mức giảm tối đa của portfolio |

### Fraud Detection
| Metric | Ý nghĩa | Tại sao không dùng Accuracy |
|---|---|---|
| AUC-PR | Area under Precision-Recall curve | Imbalanced data (fraud ~1%) |
| Precision@K | Precision trong top K cảnh báo | Giới hạn analyst bandwidth |
| Recall | Tỷ lệ fraud được phát hiện | Miss fraud = mất tiền |

> ⚠️ **Quan trọng:** Accuracy vô nghĩa với fraud detection — model dự đoán "không fraud" 100% sẽ đạt ~99% accuracy nhưng hoàn toàn vô dụng.

---

## Giới hạn Databricks Community Edition

| Giới hạn | Impact | Workaround |
|---|---|---|
| Cluster tắt sau 2h idle | Mất computation state | Checkpoint Delta sau mỗi bước |
| 1 cluster duy nhất | Không parallel train | Train lần lượt |
| ~15GB RAM, 2 cores | Chậm với dataset lớn | Giữ scope nhỏ (5 stocks, 3 years) |
| Không có Jobs scheduler | Không auto-run | Chạy manual hoặc GitHub Actions |
| Không có Model Serving | Không có managed endpoint | FastAPI local |

---

## Setup

### 1. Clone repo

```bash
git clone https://github.com/<username>/finsight.git
cd finsight
```

### 2. Cài dependencies local (cho serving + tests)

```bash
pip install -r requirements.txt
```

### 3. Setup API keys

```bash
cp .env.example .env
# Điền ALPHA_VANTAGE_API_KEY vào .env
# Lấy free key tại: https://www.alphavantage.co/support/#api-key
```

### 4. Databricks setup

1. Đăng nhập [Databricks Community Edition](https://community.cloud.databricks.com)
2. Tạo cluster: **Runtime 14.x ML** (bao gồm MLflow, Spark 3.5)
3. Install libraries trên cluster:
   ```
   yfinance
   pandas-ta
   prophet
   fastapi
   uvicorn
   ```
4. Import notebooks từ `notebooks/` vào Databricks workspace
5. Chạy theo thứ tự: `01` → `02` → `03` → `04` → `05` → `06`

### 5. Chạy API local (sau khi có model)

```bash
cd serving
uvicorn api:app --reload --port 8000
```

Endpoints:
```
GET  http://localhost:8000/predict/stock/AAPL
POST http://localhost:8000/predict/fraud
     Body: {"amount": 5000, "merchant": "ATM", "hour": 3, ...}
```

---

## Kết quả (cập nhật sau khi train xong)

### Stock Prediction

| Model | Accuracy | F1 | Sharpe |
|---|---|---|---|
| Baseline (MA crossover) | - | - | - |
| XGBoost | - | - | - |
| Prophet | - | - | - |

### Fraud Detection

| Model | AUC-PR | Precision@100 | Recall |
|---|---|---|---|
| Baseline (random) | - | - | - |
| Isolation Forest | - | - | - |
| Autoencoder | - | - | - |

---

## Cảnh báo quan trọng

> ⚠️ Project này được xây dựng với mục đích **học tập**. Kết quả dự đoán giá cổ phiếu **KHÔNG** phải là lời khuyên đầu tư. Past performance does not guarantee future results.

---

## License

MIT License — free to use for learning purposes.

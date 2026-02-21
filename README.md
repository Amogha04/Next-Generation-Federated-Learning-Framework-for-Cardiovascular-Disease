# Next-Generation Federated Learning Framework for Cardiovascular Disease Prediction with Cancer Intelligence Extension

---

## 📌 Overview

This project implements a **privacy-preserving federated learning framework** for multi-disease medical prediction.

The framework supports:

- ✅ Centralized baseline training  
- ✅ Federated learning using Flower  
- ✅ Differential Privacy using Opacus  
- ✅ Multi-disease modular architecture  
- ✅ Privacy–utility tradeoff analysis  

---

## 🏗 Architecture

### System Components

- Federated Server (Flower)
- Multiple Hospital Clients
- Differential Privacy Engine (Opacus)
- Modular Disease Data Loader

### Supported Diseases

- ❤️ Cardiovascular Disease (UCI Heart Dataset)
- 🎗 Breast Cancer (Scikit-learn Dataset)

---

## 📊 Experimental Results

### Heart Disease

| Model | Accuracy | AUC |
|-------|----------|------|
| Centralized | ~0.79 | ~0.90 |
| Federated (Vanilla) | ~0.886 | ~0.946 |
| Federated + DP | ~0.843 | ~0.923 |

### Cancer (DP Federated)

| Model | Accuracy | AUC |
|-------|----------|------|
| Federated + DP | ~0.931 | ~0.976 |

---

## 🔐 Privacy–Utility Tradeoff

Introducing Differential Privacy slightly reduces model performance while significantly improving data confidentiality across distributed medical institutions.

---

## 🚀 How to Run

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Run Server

```bash
python -m src.federated.server
```

### 3️⃣ Run Clients (Open 3 Separate Terminals)

```bash
python -m src.federated.dp_client0
python -m src.federated.dp_client1
python -m src.federated.dp_client2
```

---

## 🔮 Future Work

- Secure aggregation  
- Non-IID client simulation  
- Real hospital datasets integration  
- Edge-device deployment  

---

## 👨‍💻 Team Members

- Amoghavarsha (22BCS009)  
- B S Aathreya Sharma (22BCS024)  
- Yellaling Kalyane (22BCS140)  
- Bharath L (22BDS013)  

---


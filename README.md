# 🚗✨ Automotive Sensor-Based Failure Prediction System

> 🔥 A deployable AI-powered Streamlit application that predicts **automotive component failure** using sensor data (Temperature, RPM, Load, Vibration).  
Built for **real-world predictive maintenance systems in the automotive industry.**

---

## 🌟 Project Overview

This project uses **Machine Learning + Sensor Data Simulation** to predict whether an automotive component is:

- ✅ Healthy (No Failure)
- ⚠ At Risk / Failure Detected

It simulates real industrial conditions using **synthetic but realistic sensor patterns**.

---

## 🧠 Key Features

🚀 8,000+ synthetic sensor data points  
⚙️ Multi-sensor input (Temperature, RPM, Load, Vibration)  
🤖 RandomForestClassifier for high accuracy predictions  
📊 StandardScaler normalization for stable predictions  
📈 Real-time probability score output  
📉 Feature importance visualization  
🌐 Interactive Streamlit web dashboard  
💾 Model persistence using Pickle (`model.pkl`, `scaler.pkl`)  
⚡ Fully deployment-ready architecture  

---

## 🏗️ System Architecture


Sensor Data Simulation → Preprocessing → Feature Scaling → Model Training → Prediction Engine → Streamlit UI


---

## 📁 Project Structure


automotive-failure-prediction/
│
├── app.py # Streamlit Web App
├── model.pkl # Trained ML Model
├── scaler.pkl # Feature scaler
├── requirements.txt # Dependencies
├── README.md # Documentation
│
├── data/
│ └── synthetic_data.csv # Generated dataset
│
├── model/
│ └── train_model.py # Training script
│
└── utils/
└── preprocessing.py # Data preprocessing logic


---

## ⚙️ Tech Stack

🟡 Python  
🟢 NumPy & Pandas  
🔵 Scikit-learn  
🟣 Matplotlib & Seaborn  
🟠 Streamlit  
⚫ Pickle  

---

## 🧪 Model Workflow

✔ Data generation (synthetic sensor signals)  
✔ Feature preprocessing (scaling + cleaning)  
✔ Train-test split  
✔ RandomForest model training  
✔ Evaluation (Accuracy + ROC AUC + Classification Report)  
✔ Model saving for deployment  

---

## 📊 Streamlit App Features

🌐 Interactive UI dashboard  
📥 Real-time sensor input simulation  
📊 Failure probability score display  
📈 Feature contribution visualization  
⚠ Instant failure detection alerts  

---

## 🚀 How to Run Locally

### 1️⃣ Clone repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd automotive-failure-prediction
2️⃣ Create virtual environment
python -m venv .venv
.venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Train the model
python model/train_model.py
5️⃣ Run Streamlit app
streamlit run app.py
🌍 Deployment (Streamlit Cloud)

🚀 Steps:

Push project to GitHub
Go to 👉 https://streamlit.io/cloud
Click New App
Select repository
Set entry point → app.py
Click Deploy
📈 Model Highlights

✔ High accuracy RandomForest model
✔ Robust against noisy sensor data
✔ Real-time prediction capability
✔ Feature-scaled stable performance

🏭 Real-World Applications

🚗 Automotive predictive maintenance
🏭 Industrial machine monitoring
⚙️ Engine health tracking
📡 IoT-based smart vehicle systems
🧠 AI-driven fault detection systems

🔮 Future Improvements

🔥 Deep Learning (LSTM for sensor time-series)
🔥 Real IoT sensor integration
🔥 Live streaming failure detection
🔥 Cloud dashboard analytics
🔥 Multi-class failure classification

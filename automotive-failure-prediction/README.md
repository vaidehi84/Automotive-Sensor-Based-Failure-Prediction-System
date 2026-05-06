# Automotive Sensor-Based Failure Prediction System

A deployable, industry-level Streamlit application that predicts automotive component failure using synthetic sensor data from temperature, RPM, load, and vibration.

## Project Structure

```
automotive-failure-prediction/
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
├── data/
│   └── synthetic_data.csv
├── model/
│   └── train_model.py
└── utils/
    └── preprocessing.py
```

## Features

- Synthetic dataset with 8,000+ rows
- RandomForestClassifier for robust binary failure prediction
- StandardScaler preprocessing to normalize sensor features
- Train/test split with performance reporting
- Streamlit UI with real-time prediction, probability score, and feature chart
- Model persistence using `pickle`

## Installation

1. Clone or download the repository.
2. Open the project folder in VS Code.
3. Create and activate a Python virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

4. Install dependencies.

```bash
pip install -r requirements.txt
```

## Train the model

Run the training script to generate `model.pkl`, `scaler.pkl`, and the synthetic dataset CSV.

```bash
python model/train_model.py
```

## Run the Streamlit app locally

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit in your browser.

## Deployment to Streamlit Cloud

1. Push the repository to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Create a new app and connect your GitHub repository.
4. Set the main file path to `app.py`.
5. Use the default branch and click Deploy.

> Ensure `model.pkl`, `scaler.pkl`, and `requirements.txt` are committed to GitHub.

## GitHub Push Instructions

```bash
git init
git add .
git commit -m "Initial commit: automotive sensor failure prediction project"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Notes

- The dataset and model are synthetic but follow realistic failure patterns.
- The model performance includes accuracy, ROC AUC, and classification metrics.
- If you want to retrain with different parameters, update `model/train_model.py`.

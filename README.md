# TruthLens 🕵️‍♂️📰

TruthLens is a Streamlit-based Fake News Detection App that helps users identify whether a news article is real or fake using machine learning models.

## 🔍 Features

- Classifies news as **Real** or **Fake**.
- Supports prediction via:
  - News Title
  - Full News (Title + Body)
- Probability-based confidence scoring.
- Clean, interactive Streamlit interface.

## 📦 Tech Stack

- Python, Streamlit
- scikit-learn, joblib, re
- pandas, numpy, matplotlib, seaborn
- TF-IDF Vectorization
- Random Forest Classifier
- LightGBM Classifier

### 📦 Dataset
We used a custom dataset `combined.csv` which merges multiple publicly available fake and real news sources.

- ✅ It includes title, text, and label columns.
- 📁 Stored locally in the project repository for seamless deployment.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py


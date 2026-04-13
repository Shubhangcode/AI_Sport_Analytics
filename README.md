# 🏏 AI Sports Analytics Dashboard (Advanced ML Version)

An end-to-end AI/ML system for cricket player analysis, segmentation, performance prediction, and intelligent team building — enhanced with improved model accuracy and production-level ML pipeline.

---

## 🚀 Overview

This project combines **Machine Learning, Data Science, and Sports Analytics** to help coaches, analysts, and teams make data-driven decisions.

The system processes player performance data, clusters players into meaningful groups, predicts future performance, and recommends optimal team combinations.

---

## 🔥 Key Features

### 📊 Data Overview

* Dataset summary and statistics
* Interactive visualizations using Plotly
* Distribution and correlation analysis

---

### 🧠 Player Segmentation (Unsupervised Learning)

* PCA for dimensionality reduction
* KMeans clustering for grouping players
* 3D visualization of player clusters
* Identifies player roles:
  * Batsmen
  * Bowlers
  * All-rounders

---

### 🔮 Advanced Performance Prediction (Improved)

* Upgraded ML models (Random Forest / Gradient Boosting)
* Feature scaling and normalization
* Enhanced feature engineering:
  * Batting impact
  * Bowling efficiency
  * Player consistency
* Realistic and stable predictions
* Evaluated using:
  * R² Score
  * Mean Absolute Error (MAE)

---

### 🤝 Similar Player Recommendation

* Cosine similarity-based recommendation system
* Finds players with similar performance profiles
* Useful for scouting and replacements

---

### 🏏 Team Builder (Coach Mode)

* Select team strategy:
  * Batting-focused
  * Bowling-focused
  * Balanced
* AI selects optimal team based on:
  * Performance score
  * Player roles
  * Cluster distribution
* Provides explanation for selections

---

## 🧠 Tech Stack

* **Python**
* **Pandas, NumPy** (Data Processing)
* **Scikit-learn** (ML Models, PCA, Clustering)
* **Plotly** (Visualization)
* **Streamlit** (Web App)

---

## 📁 Project Structure

```text
AI-Sports_Analytics/
│
├── data/
│   └── player_stats.csv
│
├── Src/
│   ├── preprocess.py        # Data cleaning & feature engineering
│   ├── clustering.py        # PCA + KMeans
│   ├── prediction.py        # ML model (improved)
│   └── recommendation.py    # Similarity system
│
├── streamlit_app.py         # Main app
├── requirements.txt
└── README.md
```

---

## ⚙️ ML Pipeline

```text
Raw Data → Preprocessing → Feature Engineering → Scaling → PCA → Clustering → Prediction → Recommendation → Streamlit App
```

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🌐 Deployment

Deployed using Streamlit Community Cloud

👉 **Live App:** [https://aisport-wtxwjmfvylvvnpdbuv5wkt.streamlit.app/](https://aisport-wtxwjmfvylvvnpdbuv5wkt.streamlit.app/)

---

## 📊 Model Improvements (Latest Update)

* Replaced basic model with advanced ML algorithms (Random Forest)
* Added feature scaling and normalization
* Improved feature engineering for realistic outputs
* Handled outliers and noisy data
* Ensured consistent pipeline across all modules
* Fixed all GitHub and runtime errors (including `KeyError` & `Unicode` fixes)
* Improved prediction stability and reliability

---

## 📌 Use Cases

* Cricket team selection
* Player performance analysis
* Talent scouting
* Strategy building for coaches
* Sports analytics research

---

## 📈 Future Enhancements

* Real-time match data integration
* Deep learning-based prediction models
* Match outcome prediction
* Player injury risk analysis

---

## 👨‍💻 Author

Shubhang

---

## ⭐ Support

If you found this project useful, give it a ⭐ on GitHub!

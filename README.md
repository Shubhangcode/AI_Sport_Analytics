# 🏏 AI Sports Analytics Dashboard

An end-to-end AI/ML project that analyzes cricket player performance, segments players using unsupervised learning, predicts performance scores, and helps coaches build optimal teams.

---

## 🚀 Features

### 📊 Data Overview

* Summary of player dataset
* Key statistics (Runs, Wickets, Strike Rate, Economy)
* Interactive visualizations using Plotly

---

### 🧠 Player Segmentation (Clustering)

* PCA for dimensionality reduction
* KMeans clustering to group players
* 3D interactive visualization of player clusters
* Identify player types (batsman, bowler, all-rounder)

---

### 🔮 Performance Prediction

* Machine Learning regression model
* Predicts player performance score
* Takes inputs like runs, wickets, strike rate, economy

---

### 🤝 Similar Player Recommendation

* Uses cosine similarity
* Finds players with similar performance profiles
* Useful for replacements and scouting

---

### 🏏 Team Builder (Coach Mode)

* Select team strategy:

  * Batting-focused
  * Bowling-focused
  * Balanced
* Automatically generates optimal team
* Based on performance + clustering

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn (PCA, KMeans, Regression)
* Plotly (visualization)
* Streamlit (web app)

---

## 📁 Project Structure

```
AI-Sports_Analytics/
│
├── data/
│   └── player_stats.csv
│
├── Src/
│   ├── preprocess.py
│   ├── clustering.py
│   ├── prediction.py
│   └── recommendation.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. Raw cricket data is preprocessed
2. Features like strike rate, economy are engineered
3. PCA reduces feature dimensions
4. KMeans groups players into clusters
5. Regression model predicts performance score
6. Similarity model recommends players
7. Streamlit app visualizes everything

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🌐 Deployment

Deployed using Streamlit Community Cloud

👉 Live App: *(https://aisportanalytics.streamlit.app/)*

---

## 📌 Use Cases

* Team selection for coaches
* Player performance analysis
* Talent scouting
* Sports analytics research

---

## 📈 Future Improvements

* Real-time match data integration
* Deep learning models
* Player injury prediction
* Match outcome prediction

---

## 👨‍💻 Author

Shubhang Trivedi



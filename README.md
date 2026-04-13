🏏 AI Sports Analytics Dashboard

An end-to-end AI/ML project that analyzes cricket player performance, segments players using unsupervised learning, predicts performance scores, and helps coaches build optimal teams.

🚀 Features
📊 Data Overview
Summary of player dataset
Key statistics (Runs, Wickets, Strike Rate, Economy)
Interactive visualizations using Plotly
🧠 Player Segmentation (Clustering)
PCA for dimensionality reduction
KMeans clustering to group players
3D interactive visualization of player clusters
Identify player types (batsman, bowler, all-rounder)
🔮 Performance Prediction
Machine Learning regression model
Predicts player performance score
Takes inputs like runs, wickets, strike rate, economy
🤝 Similar Player Recommendation
Uses cosine similarity
Finds players with similar performance profiles
Useful for replacements and scouting
🏏 Team Builder (Coach Mode)
Select team strategy:
Batting-focused
Bowling-focused
Balanced
Automatically generates optimal team
Based on performance + clustering
🧠 Tech Stack
Python
Pandas, NumPy
Scikit-learn (PCA, KMeans, Regression)
Plotly (visualization)
Streamlit (web app)
📁 Project Structure
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
⚙️ How It Works
Raw cricket data is preprocessed
Features like strike rate, economy are engineered
PCA reduces feature dimensions
KMeans groups players into clusters
Regression model predicts performance score
Similarity model recommends players
Streamlit app visualizes everything

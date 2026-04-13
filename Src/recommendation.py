import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class PlayerRecommender:
    def __init__(self, df):
        self.df = df
        
        # Prepare features for similarity calculation
        features = df.drop(columns=["Player", "Cluster"], errors='ignore')
        numeric_features = features.select_dtypes(include=['float64', 'int64'])
        
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(numeric_features)
        self.similarity_matrix = cosine_similarity(self.scaled_features)

    def recommend_players(self, player_name, top_n=5):
        if player_name not in self.df["Player"].values:
            return None
        
        player_index = self.df[self.df["Player"] == player_name].index[0]
        similarity_scores = self.similarity_matrix[player_index]
        
        # Get indices of top similar players (excluding self)
        similar_indices = similarity_scores.argsort()[-(top_n+1):-1][::-1]
        
        # Build results DataFrame
        result_df = self.df.iloc[similar_indices].copy()
        result_df["SimilarityScore"] = similarity_scores[similar_indices].round(3) * 100
        
        columns_to_show = ["Player", "Runs", "Wickets", "StrikeRate", "Economy", "PerformanceScore", "SimilarityScore"]
        return result_df[[col for col in columns_to_show if col in result_df.columns]]

    def get_similarity(self, player1, player2):
        if player1 not in self.df["Player"].values or player2 not in self.df["Player"].values:
            return None
        
        idx1 = self.df[self.df["Player"] == player1].index[0]
        idx2 = self.df[self.df["Player"] == player2].index[0]
        
        score = self.similarity_matrix[idx1][idx2]
        return round(score, 3)

if __name__ == "__main__":
    df = pd.read_csv("../data/player_stats.csv")
    recommender = PlayerRecommender(df)
    player = "AB de Villiers"
    print(f"Players similar to {player}:\n", recommender.recommend_players(player))
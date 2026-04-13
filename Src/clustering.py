import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_player_clusters(df, n_clusters=4):
    features = df.drop(columns=["Player", "Cluster"], errors="ignore")
    numeric_features = features.select_dtypes(include=['float64', 'int64'])
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(scaled_data)
    
    df_result = df.copy()
    df_result["Cluster"] = clusters
    return df_result, kmeans, scaler

if __name__ == "__main__":
    df = pd.read_csv("../data/player_stats.csv")
    clustered_df, _, _ = get_player_clusters(df)
    print(clustered_df[["Player", "Cluster"]].head())
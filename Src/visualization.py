import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_3d_cluster_plot(df):
    if "Cluster" not in df.columns:
        raise ValueError("Dataset must be clustered first via get_player_clusters.")
        
    features = df.drop(columns=["Player", "Cluster"], errors='ignore')
    numeric_features = features.select_dtypes(include=['float64', 'int64'])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_features)

    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(scaled_data)

    plot_df = df.copy()
    plot_df["PC1"] = pca_data[:, 0]
    plot_df["PC2"] = pca_data[:, 1]
    plot_df["PC3"] = pca_data[:, 2]

    # Clean Hover Data
    hover_cols = ["Runs", "Wickets", "StrikeRate", "Economy"]
    hover_data = {col: True for col in hover_cols if col in plot_df.columns}

    fig = px.scatter_3d(
        plot_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color=plot_df["Cluster"].astype(str),
        hover_name="Player",
        hover_data=hover_data,
        title="🏏 AI Cricket Player Clustering (3D API)",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Improve aesthetics
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title='Power (PC1)',
            yaxis_title='Consistency (PC2)',
            zaxis_title='Bowling Impact (PC3)'
        )
    )
    return fig

if __name__ == "__main__":
    from clustering import get_player_clusters
    df = pd.read_csv("../data/player_stats.csv")
    df, _, _ = get_player_clusters(df)
    fig = create_3d_cluster_plot(df)
    fig.show()

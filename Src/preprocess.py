import pandas as pd
import os

def clean_data(input_path="Data/player_stats.csv", output_path="Data/player_stats.csv"):
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        return None

    # Ensure consistent column naming
    if "name_x" in df.columns:
        df.rename(columns={"name_x": "Player"}, inplace=True)
    
    # Check if raw data needs grouping
    if df.duplicated(subset=["Player"]).any():
        print("Raw data detected. Aggregating...")
        batting = df.groupby("Player").agg({
            "Runs": "sum",
            "Fours": "sum",
            "Sixes": "sum",
            "Dots": "sum"
        }).reset_index()

        bowling = df.groupby("Player").agg({
            "Wickets": "sum",
            "RunsConceded": "sum",
            "Overs": "sum"
        }).reset_index()
        
        df = pd.merge(batting, bowling, on="Player", how="outer").fillna(0)
    
    # Recalculate features to ensure consistency
    df["StrikeRate"] = (df["Runs"] / (df["Dots"] + 1)) * 100
    df["Economy"] = df["RunsConceded"] / (df["Overs"] + 1)
    df["PerformanceScore"] = df["Runs"] + (df["Wickets"] * 25)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("✅ Preprocessing Done")
    return df

if __name__ == "__main__":
    clean_data()
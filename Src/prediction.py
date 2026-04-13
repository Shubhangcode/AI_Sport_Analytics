import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_performance_model(df):
    if "PerformanceScore" not in df.columns:
        raise ValueError("Missing 'PerformanceScore' in dataset.")
        
    X = df.drop(columns=["Player", "PerformanceScore", "Cluster"], errors='ignore')
    X = X.select_dtypes(include=['float64', 'int64'])
    y = df["PerformanceScore"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, scaler, mae, r2, X.columns

if __name__ == "__main__":
    df = pd.read_csv("../Data/player_stats.csv")
    model, scaler, mae, r2, _ = train_performance_model(df)
    print(f"Random Forest MAE: {mae:.2f}")
    print(f"Random Forest R2: {r2:.2f}")
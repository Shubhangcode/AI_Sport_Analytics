import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set up page configurations
st.set_page_config(
    page_title="AI Sports Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dashboard aesthetics
st.markdown("""
<style>
    /* Global Styles */
    .reportview-container { background: #f4f6f9; }
    h1, h2, h3 { color: #1e3a8a; }
    
    /* Metrics Styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff, #f0fdf4);
        border-right: 4px solid #22c55e;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #3b82f6;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Callout Boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Imports
try:
    from Src.preprocess import clean_data
    from Src.clustering import get_player_clusters
    from Src.recommendation import PlayerRecommender
    from Src.prediction import train_performance_model
    from Src.visualization import create_3d_cluster_plot
except ModuleNotFoundError as e:
    st.error(f"Error loading source modules. Module missing: {e}")
    st.stop()

# Helper function to load data
@st.cache_data
def load_and_preprocess_data():
    file_path = "Data/player_stats.csv"
    if not os.path.exists(file_path):
        file_path = "data/player_stats.csv"
        if not os.path.exists(file_path):
            return None
    # Preprocess
    df = clean_data(input_path=file_path, output_path=file_path)
    return df

st.title("🤖 AI-Powered Sports Analytics Dashboard")
st.markdown("A state-of-the-art predictive and analytical toolkit for modern cricket coaching and scouting. Navigate the sidebar to explore data, predict performance, discover similar players, or auto-build a team.")

df = load_and_preprocess_data()

if df is not None:
    # Sidebar Navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/5325/5325442.png", width=120)
        st.markdown("### 🧭 Dashboard Navigation")
        app_mode = st.radio("Select Module:", [
            "📊 Data Overview", 
            "🧠 Player Segmentation", 
            "🔮 Performance Prediction", 
            "🤝 Team Builder",
            "🔍 Similar Player Recommendation"
        ])
        st.markdown("---")
        st.caption("Developed for Sports Analytics Professionals.")

    if app_mode == "📊 Data Overview":
        st.header("📊 Dataset Overview & Key Statistics")
        st.markdown("Easily view high-level KPIs and explore the foundational dataset powering the AI models.")
        
        # Key Stat Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Players", len(df))
        col2.metric("Total Runs Scored", f"{int(df['Runs'].sum()):,}")
        col3.metric("Total Wickets Taken", f"{int(df['Wickets'].sum()):,}")
        col4.metric("Avg Economy Rate", f"{df['Economy'].mean():.2f}")
        
        st.markdown("---")
        
        # Interactive Plotly Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Runs Distribution")
            fig_hist = px.histogram(df, x="Runs", 
                                    nbins=20, 
                                    color_discrete_sequence=['#3b82f6'],
                                    title="Distribution of Player Runs")
            fig_hist.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with c2:
            st.subheader("Batting vs Bowling Impact")
            fig_scatter = px.scatter(df, x="Runs", y="Wickets", 
                                     color="PerformanceScore",
                                     size="PerformanceScore",
                                     hover_name="Player",
                                     color_continuous_scale="Viridis",
                                     title="Wickets vs Runs (Sized & Colored by Performance Score)")
            fig_scatter.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(50), use_container_width=True)

    elif app_mode == "🧠 Player Segmentation":
        st.header("🧠 AI Player Segmentation (K-Means Clustering)")
        st.write("Using PCA and K-Means Clustering to group players into functional archetypes based on multi-dimensional statistics.")
        
        # Perform clustering
        clustered_df, kmeans_model, _ = get_player_clusters(df, n_clusters=4)
        
        # Dynamically assign meaningful names to clusters based on attributes
        cluster_means = clustered_df.groupby("Cluster")[["Runs", "Wickets"]].mean()
        overall_runs_mean = df["Runs"].mean()
        overall_wickets_mean = df["Wickets"].mean()
        
        role_map = {}
        for c in cluster_means.index:
            r = cluster_means.loc[c, 'Runs']
            w = cluster_means.loc[c, 'Wickets']
            # Determine role naming logic strictly by highest relative traits
            if r > overall_runs_mean * 1.1 and w > overall_wickets_mean * 1.1:
                role_map[c] = "Premium All-Rounders 🌟"
            elif r > overall_runs_mean * 1.1:
                role_map[c] = "Aggressive Batsmen 🏏"
            elif w > overall_wickets_mean * 1.1:
                role_map[c] = "Specialist Bowlers 🎯"
            else:
                role_map[c] = "Bench/Developing Players 🌱"
                
        # Ensure unique mapped names if logic overlaps
        for c, name in role_map.items():
            if list(role_map.values()).count(name) > 1:
                role_map[c] = f"{name} (Type {c})"
                
        clustered_df["Archetype"] = clustered_df["Cluster"].map(role_map)
        
        # 3D Plot
        c_cols = st.columns([2, 1])
        with c_cols[0]:
            fig = create_3d_cluster_plot(clustered_df)
            st.plotly_chart(fig, use_container_width=True)
            
        with c_cols[1]:
            st.subheader("Cluster Explanations")
            for cluster_id, role_name in role_map.items():
                count = (clustered_df["Cluster"] == cluster_id).sum()
                st.info(f"**Cluster {cluster_id}: {role_name}**\n\nTotal Players: {count}")
                
        # Interactive Filter
        st.markdown("---")
        st.subheader("🔍 Filter by Archetype")
        unique_archetypes = list(set(role_map.values()))
        selected_archetypes = st.multiselect("Select Archetypes:", options=unique_archetypes, default=unique_archetypes)
        
        if selected_archetypes:
            filtered_df = clustered_df[clustered_df["Archetype"].isin(selected_archetypes)]
            st.dataframe(filtered_df[["Player", "Archetype", "Runs", "Wickets", "StrikeRate", "Economy", "PerformanceScore"]].sort_values("PerformanceScore", ascending=False), use_container_width=True)


    elif app_mode == "🔮 Performance Prediction":
        st.header("🔮 Dynamic Performance Prediction")
        st.write("Leverage a trained Random Forest Regressor to forecast a player's underlying Performance Score from standard statistics.")
        
        with st.spinner("Training Model..."):
            model, scaler, mae, r2, feature_cols = train_performance_model(df)
            
        model_col, form_col = st.columns([1, 2])
        
        with model_col:
            st.success("🤖 Model Trained & Ready!")
            st.metric("Mean Absolute Error", f"{mae:.2f}")
            st.metric("R² Confidence Score", f"{r2:.2f}")
            st.info("The Performance Score is a composite metric indicating overall match impact.")
            
        with form_col:
            st.subheader("Input Player Capabilities")
            c1, c2 = st.columns(2)
            input_runs = c1.number_input("Expected Runs", min_value=0, value=500, step=50)
            input_sr = c1.number_input("Batting Strike Rate", min_value=0.0, value=130.0, step=5.0)
            input_wickets = c2.number_input("Expected Wickets", min_value=0, value=15, step=1)
            input_econ = c2.number_input("Bowling Economy", min_value=0.0, value=7.5, step=0.5)
            
            if st.button("🔮 Predict Performance Score"):
                # Fill missing features with dataset means
                sample_data = df.mean(numeric_only=True).to_dict()
                
                # Override with user inputs
                if "Runs" in sample_data: sample_data["Runs"] = input_runs
                if "StrikeRate" in sample_data: sample_data["StrikeRate"] = input_sr
                if "Wickets" in sample_data: sample_data["Wickets"] = input_wickets
                if "Economy" in sample_data: sample_data["Economy"] = input_econ
                
                sample_df = pd.DataFrame([sample_data])[feature_cols]
                scaled_sample = scaler.transform(sample_df)
                pred = model.predict(scaled_sample)[0]
                
                # Interpretation logic
                if pred > df["PerformanceScore"].quantile(0.8):
                    interpretation = "High Performer 🚀"
                    color_code = "success"
                elif pred > df["PerformanceScore"].quantile(0.4):
                    interpretation = "Average/Solid Pro ⚖️"
                    color_code = "info"
                else:
                    interpretation = "Needs Improvement 📉"
                    color_code = "warning"
                
                st.markdown("### Prediction Result")
                if color_code == "success":
                    st.success(f"**Predicted Score:** {pred:.2f} — {interpretation}")
                elif color_code == "info":
                    st.info(f"**Predicted Score:** {pred:.2f} — {interpretation}")
                else:
                    st.warning(f"**Predicted Score:** {pred:.2f} — {interpretation}")
                st.balloons() if color_code == "success" else None


    elif app_mode == "🤝 Team Builder":
        st.header("🤝 Smart Team Builder (Coach Mode)")
        st.write("Generate an optimized squad based on AI clustering, performance metrics, and role composition.")
        
        # Perform clustering so we have roles
        clustered_df, _, _ = get_player_clusters(df, n_clusters=4)
        cluster_means = clustered_df.groupby("Cluster")[["Runs", "Wickets"]].mean()
        runs_m, wick_m = df["Runs"].mean(), df["Wickets"].mean()
        
        def identify_role(row):
            c = row["Cluster"]
            r = cluster_means.loc[c, 'Runs']
            w = cluster_means.loc[c, 'Wickets']
            if r > runs_m * 1.1 and w > wick_m * 1.1: return "All-Rounder"
            elif r > runs_m * 1.1: return "Batsman"
            elif w > wick_m * 1.1: return "Bowler"
            return "Backup/Developing"
            
        clustered_df["DesignatedRole"] = clustered_df.apply(identify_role, axis=1)
        
        # Sidebar controls for Team Builder
        st.markdown("### Set Squad Preferences")
        col_t1, col_t2 = st.columns(2)
        squad_size = col_t1.slider("Squad Size (Number of Players)", min_value=5, max_value=15, value=11)
        focus = col_t2.selectbox("Team Strategy Focus", ["Balanced", "Batting Heavy", "Bowling Heavy"])
        
        if st.button("🚀 Auto-Generate Squad"):
            with st.spinner("AI evaluating combinations..."):
                # Define rules based on focus
                if focus == "Balanced":
                    bat_target = int(squad_size * 0.45)
                    bowl_target = int(squad_size * 0.40)
                    allr_target = squad_size - bat_target - bowl_target
                elif focus == "Batting Heavy":
                    bat_target = int(squad_size * 0.55)
                    bowl_target = int(squad_size * 0.30)
                    allr_target = squad_size - bat_target - bowl_target
                else: # Bowling heavy
                    bat_target = int(squad_size * 0.30)
                    bowl_target = int(squad_size * 0.55)
                    allr_target = squad_size - bat_target - bowl_target
                
                # Select top players per role by PerformanceScore
                squad = pd.DataFrame()
                
                # Batsmen
                batsmen = clustered_df[clustered_df["DesignatedRole"] == "Batsman"].nlargest(bat_target, "PerformanceScore")
                squad = pd.concat([squad, batsmen])
                
                # Bowlers
                bowlers = clustered_df[clustered_df["DesignatedRole"] == "Bowler"].nlargest(bowl_target, "PerformanceScore")
                squad = pd.concat([squad, bowlers])
                
                # All-Rounders
                allrounders = clustered_df[(clustered_df["DesignatedRole"] == "All-Rounder") & (~clustered_df["Player"].isin(squad["Player"]))].nlargest(allr_target, "PerformanceScore")
                squad = pd.concat([squad, allrounders])
                
                # If short, fill with next best overall
                if len(squad) < squad_size:
                    remaining = clustered_df[~clustered_df["Player"].isin(squad["Player"])].nlargest(squad_size - len(squad), "PerformanceScore")
                    squad = pd.concat([squad, remaining])
                    
                squad = squad.drop_duplicates(subset=["Player"]).head(squad_size)
                
                # Add Explanation Column
                squad["Selection Reason"] = squad.apply(
                    lambda x: f"Top Ranked {x['DesignatedRole']} (Score: {x['PerformanceScore']:.1f})", axis=1
                )
                
                st.success(f"**Successfully Built {focus} Squad of {len(squad)} Players!**")
                
                c_role_pie, c_role_table = st.columns([1, 2])
                with c_role_pie:
                    # Visual summary
                    role_counts = squad["DesignatedRole"].value_counts()
                    fig_roles = px.pie(values=role_counts.values, names=role_counts.index, 
                                       title="Squad Composition", hole=0.4,
                                       color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig_roles, use_container_width=True)
                
                with c_role_table:
                    # Final Table
                    st.markdown("### Final Roster")
                    st.dataframe(squad[["Player", "DesignatedRole", "Selection Reason", "Runs", "Wickets", "PerformanceScore"]].sort_values("PerformanceScore", ascending=False), use_container_width=True, hide_index=True)

    elif app_mode == "🔍 Similar Player Recommendation":
        st.header("🔍 Similar Player Search & Scouting")
        st.write("Using Cosine Similarity logic to find equivalents across multi-dimensional statistical profiles.")
        
        player_list = sorted(df["Player"].unique().tolist())
        
        sel_col1, sel_col2 = st.columns(2)
        selected_player = sel_col1.selectbox("Select Target Player", player_list)
        top_n = sel_col2.slider("Number of Recommendations", 3, 10, 5)
        
        if st.button("Target Lock: Find Similars 🎯"):
            with st.spinner("Running similarity matrix..."):
                recommender = PlayerRecommender(df)
                results = recommender.recommend_players(selected_player, top_n)
                
            if results is not None and not results.empty:
                st.success(f"Closest Matches for **{selected_player}**:")
                
                # Comparison table
                st.dataframe(results[["Player", "SimilarityScore", "Runs", "Wickets", "StrikeRate", "Economy"]]
                             .style.format({"SimilarityScore": "{:.1f}%", "StrikeRate": "{:.2f}", "Economy": "{:.2f}"}), 
                             use_container_width=True, hide_index=True)
                
                # Radar Chart for stats comparison
                st.subheader("Statistical Footprint Comparison")
                
                # Normalize features for radar chart visually
                target_stats = df[df["Player"] == selected_player][["Runs", "Wickets", "StrikeRate", "PerformanceScore"]].iloc[0]
                
                fig_radar = go.Figure()
                
                # Add Target Player
                fig_radar.add_trace(go.Scatterpolar(
                    r=[target_stats["Runs"]/df["Runs"].max(), 
                       target_stats["Wickets"]/df["Wickets"].max(), 
                       target_stats["StrikeRate"]/df["StrikeRate"].max(), 
                       target_stats["PerformanceScore"]/df["PerformanceScore"].max()],
                    theta=['Runs (Norm)', 'Wickets (Norm)', 'Strike Rate (Norm)', 'Performance (Norm)'],
                    fill='toself',
                    name=f"{selected_player} (Target)",
                    line_color="red"
                ))
                
                # Add top 2 recommended
                for i in range(min(2, len(results))):
                    rec = results.iloc[i]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[rec["Runs"]/df["Runs"].max(), 
                           rec["Wickets"]/df["Wickets"].max(), 
                           rec["StrikeRate"]/df["StrikeRate"].max(), 
                           rec["PerformanceScore"]/df["PerformanceScore"].max()],
                        theta=['Runs (Norm)', 'Wickets (Norm)', 'Strike Rate (Norm)', 'Performance (Norm)'],
                        fill='toself',
                        name=rec["Player"]
                    ))

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Top 2 Alternatives vs Target Player",
                    margin=dict(t=40, b=40)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
            else:
                st.error("Engine failed to compute similarities (player not found or data error).")
else:
    st.warning("⚠️ Critical Missing Data: 'Data/player_stats.csv' not found. Please ingest data to launch dashboard.")

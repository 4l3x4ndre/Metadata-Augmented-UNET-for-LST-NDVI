import streamlit as st
import pandas as pd
import os

def get_temporal_distance(t1_year):
    """Categorizes t1_year into temporal distance buckets."""
    if t1_year <= 2021:
        return 'long_distance'
    elif t1_year in [2022, 2023]:
        return 'mid_distance'
    elif t1_year > 2023:
        return 'short_distance'
    else:
        return 'other'

def get_color_for_interpretation(interpretation: str) -> str:
    """Returns a color based on the interpretation text."""
    if "Excellent" in interpretation:
        return "green"
    if "Good" in interpretation:
        return "orange"
    if "Needs improvement" in interpretation:
        return "red"
    if "overly noisy" in interpretation or "overly smooth" in interpretation:
        return "orange"
    if "realistic level" in interpretation:
        return "green"
    return "black"


def interpret_metrics_streamlit(df: pd.DataFrame, model_name: str):
    """Provides a Streamlit-based interpretation for key metrics."""
    st.subheader(f"Interpreting metrics for {model_name}")
    st.info("For regression tasks, 'accuracy' and 'precision' are captured by error metrics like MAE and RMSE. Lower is better.")

    if 'is_known_city' not in df.columns:
        df['is_known_city'] = True # Assume all are known if column is missing
    
    if 't1_year' in df.columns:
        df['temporal_distance'] = df['t1_year'].apply(get_temporal_distance)
    else:
        df['temporal_distance'] = 'not_available'

    # Create tabs for known/unknown cities
    if 'is_known_city' in df.columns and df['is_known_city'].nunique() > 1:
        known_tabs = st.tabs(["Known Cities", "Unknown Cities"])
        tab_map = {0: True, 1: False}
    else:
        # If only one type (or column doesn't exist), just show one view
        known_tabs = [st.container()]
        tab_map = {0: df['is_known_city'].iloc[0] if 'is_known_city' in df.columns else True}


    for i, tab in enumerate(known_tabs):
        with tab:
            is_known = tab_map[i]
            city_df = df[df['is_known_city'] == is_known]
            
            if city_df.empty:
                st.write("No data for this category.")
                continue

            temporal_groups = city_df.groupby('temporal_distance')
            
            for temp_dist, group_df in temporal_groups:
                with st.expander(f"Temporal Distance: {temp_dist}", expanded=True):
                    # Filter for overall metrics, as per-class can be noisy
                    overall_metrics = group_df[group_df['dw_class'] == 'overall'].groupby('channel')[['mae', 'rmse', 'laplacian_var_pred', 'laplacian_var_gt']].mean()

                    if overall_metrics.empty:
                        st.write("Could not find 'overall' metrics to interpret for this group.")
                        continue

                    for channel, row in overall_metrics.iterrows():
                        st.markdown(f"#### Channel: {channel}")
                        
                        cols = st.columns(2)
                        cols[0].metric(label=f"MAE ({channel})", value=f"{row['mae']:.4f}")
                        cols[1].metric(label=f"RMSE ({channel})", value=f"{row['rmse']:.4f}")

                        interpretation = ""
                        if 'temp' in channel:
                            if row['mae'] < 2.0:
                                interpretation = "Excellent. The model's temperature predictions are highly accurate (avg. error < 2°C)."
                            elif row['mae'] < 4.0:
                                interpretation = "Good. The model's temperature predictions are reasonably accurate (avg. error < 4°C)."
                            else:
                                interpretation = "Needs improvement. The model's temperature predictions have a notable deviation (avg. error >= 4°C)."
                            
                        elif 'ndvi' in channel:
                            if row['mae'] < 0.05:
                                interpretation = "Excellent. The model's NDVI predictions are very precise (avg. error < 2.5% of the range)."
                            elif row['mae'] < 0.1:
                                interpretation = "Good. The model's NDVI predictions are reasonably precise (avg. error < 5% of the range)."
                            else:
                                interpretation = "Needs improvement. The model's NDVI predictions show significant deviation (avg. error >= 5% of the range)."
                        
                        if interpretation:
                            color = get_color_for_interpretation(interpretation)
                            st.markdown(f"**Interpretation (Error):** <span style='color: {color};'>{interpretation}</span>", unsafe_allow_html=True)


                        # Smoothness
                        if pd.notna(row['laplacian_var_pred']) and pd.notna(row['laplacian_var_gt']) and row['laplacian_var_gt'] > 0:
                            smoothness_ratio = row['laplacian_var_pred'] / row['laplacian_var_gt']
                            st.markdown(f"**Smoothness (Laplacian Var):** Pred={row['laplacian_var_pred']:.4f}, GT={row['laplacian_var_gt']:.4f}, Ratio={smoothness_ratio:.2f}")
                            
                            if smoothness_ratio > 1.5:
                                smoothness_interp = "The model's predictions may be overly noisy or contain artifacts."
                            elif smoothness_ratio < 0.5:
                                smoothness_interp = "The model's predictions may be overly smooth, losing fine details."
                            else:
                                smoothness_interp = "The model's predictions have a realistic level of detail and texture."
                            
                            color = get_color_for_interpretation(smoothness_interp)
                            st.markdown(f"**Interpretation (Smoothness):** <span style='color: {color};'>{smoothness_interp}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("**Smoothness:** Not available (GT Laplacian variance is zero or NaN).")
                        
                        st.divider()


def main():
    st.set_page_config(layout="wide", page_title="Metric Analysis")
    st.title("Model Evaluation Metric Analysis")

    # --- Sidebar for Configuration ---
    st.sidebar.header("Configuration")
    default_csv_path = "reports/tests/2c"
    EVALUATION_DIR = st.sidebar.text_input("Path to folder of evaluation CSV", value=default_csv_path)

    # Find CSV files
    try:
        csv_files = [os.path.join(EVALUATION_DIR, f) for f in os.listdir(EVALUATION_DIR) if f.endswith('.csv') and not f.endswith('_info.csv')]
        if not csv_files:
            st.warning(f"No evaluation CSV files found in '{EVALUATION_DIR}'.")
            st.stop()
    except FileNotFoundError:
        st.error(f"Directory not found: '{EVALUATION_DIR}'. Please ensure the path is correct.")
        st.stop()

    # --- Comparative Model Analysis ---
    st.header("Comparative Model Analysis")

    @st.cache_data
    def load_and_process_all_csvs(files):
        all_dfs = []
        for csv_path in files:
            df = pd.read_csv(csv_path)
            df['model'] = os.path.basename(csv_path).replace('_evaluation.csv', '')
            
            # Try to load info file
            info_path = csv_path.replace('_evaluation.csv', '_info.csv')
            # Fallback for files that don't end in _evaluation.csv but might have info? Unlikely given evaluate.py logic.
            if not os.path.exists(info_path) and csv_path.endswith('.csv'):
                 info_path = csv_path[:-4] + '_info.csv'

            model_variant = "unknown"
            if os.path.exists(info_path):
                try:
                    info_df = pd.read_csv(info_path)
                    if not info_df.empty:
                        row = info_df.iloc[0]
                        emb = row.get('model_embedding_type', 'unknown')
                        study = row.get('study_name', '')
                        # Check for ++ in study name
                        plus = "++" if "++" in str(study) else ""
                        model_variant = f"{emb}{plus}"
                except Exception:
                    pass
            
            df['model_variant'] = model_variant

            if 'is_known_city' not in df.columns:
                df['is_known_city'] = True
            if 't1_year' in df.columns:
                df['temporal_distance'] = df['t1_year'].apply(get_temporal_distance)
            else:
                df['temporal_distance'] = 'not_available'
            all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True)

    combined_df = load_and_process_all_csvs(csv_files)
    overall_metrics_df = combined_df[combined_df['dw_class'] == 'overall'].copy()

    # Create filters
    col1, col2 = st.columns(2)
    city_type_filter = col1.selectbox("Filter by City Type", ["All", "Known", "Unknown"])
    temporal_filter = col2.selectbox("Filter by Temporal Distance", ["All", "long_distance", "mid_distance", "short_distance"])

    # Filter data
    filtered_df = overall_metrics_df.copy()
    if city_type_filter == "Known":
        filtered_df = filtered_df[filtered_df['is_known_city'] == True]
    elif city_type_filter == "Unknown":
        filtered_df = filtered_df[filtered_df['is_known_city'] == False]

    if temporal_filter != "All":
        filtered_df = filtered_df[filtered_df['temporal_distance'] == temporal_filter]

    # Group and display summary
    summary_df = filtered_df.groupby(['model', 'model_variant', 'channel'])[['mae', 'rmse', 'laplacian_var_pred', 'laplacian_var_gt']].mean().reset_index()

    st.dataframe(summary_df.style.format({
        'mae': '{:.4f}',
        'rmse': '{:.4f}',
        'laplacian_var_pred': '{:.4f}',
        'laplacian_var_gt': '{:.4f}',
    }), use_container_width=True)

    st.divider()

    # --- Detailed Single Model Analysis ---
    st.header("Detailed Single Model Analysis")
    
    selected_csv = st.selectbox("Select an evaluation CSV file for detailed analysis", csv_files, format_func=lambda x: os.path.basename(x))

    if selected_csv:
        df = pd.read_csv(selected_csv)
        model_name = os.path.basename(selected_csv).replace('_evaluation.csv', '')
        
        interpret_metrics_streamlit(df, model_name)

if __name__ == "__main__":
    main()

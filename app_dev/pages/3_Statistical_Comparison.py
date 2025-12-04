# app/pages/3_Statistical_Comparison.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon, pearsonr, mannwhitneyu

from src.utils.plot_utils import get_styled_figure_ax, convert_label, style_legend, DATASET_COLORS_5V5

def save_fig_to_report(fig, filename):
    """Saves the given figure to the reports directory."""
    output_dir = "reports/tests/app/statistics"
    os.makedirs(output_dir, exist_ok=True)
    # Sanitize filename
    safe_filename = filename.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('.', '')
    if not safe_filename.endswith('.pdf'):
        safe_filename += '.pdf'
    path = os.path.join(output_dir, safe_filename)
    fig.savefig(path, bbox_inches='tight')

def load_and_process_data(uploaded_files):
    dfs = []
    for uploaded_file in uploaded_files:
        raw_model_name = os.path.basename(uploaded_file.name).replace('_evaluation.csv', '')
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
            continue
            
        # Parse model names
        parts = raw_model_name.split('_')
        if len(parts) >= 3:
            model_type = parts[1].upper()
            embedding_type_raw = parts[2]
            embedding_type_formatted = convert_label(embedding_type_raw)
            if model_type == 'UNET++':
                # model_label = f"{embedding_type_formatted} (UNet++)"
                model_label = f"UNet++"
            else:
                model_label = f"{embedding_type_formatted}"
        else:
            model_label = raw_model_name

        df['model'] = model_label
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Calculate City Sample Count
    if 'city' in full_df.columns:
        city_counts = full_df.groupby('city')['sample_idx'].count() # Approx count (sum of all samples for that city across splits/channels?)
        first_model = full_df['model'].unique()[0]
        ref_df = full_df[full_df['model'] == first_model]
        real_city_counts = ref_df.groupby('city')['sample_idx'].nunique().to_dict()
        
        full_df['city_sample_count'] = full_df['city'].map(real_city_counts)
    else:
        full_df['city_sample_count'] = 0

    return full_df

def run_analysis_for_channel(df, channel_name, metric):
    st.markdown(f"## Analysis for Channel: **{channel_name}** ({metric.upper()})")
    
    # Filter for this channel and overall dw_class
    sub_df = df[(df['channel'] == channel_name) & (df['dw_class'] == 'overall')].copy()
    
    if sub_df.empty:
        st.warning(f"No data found for channel {channel_name} with dw_class='overall'.")
        return

    models = sorted(sub_df['model'].unique())
    
    model_colors = {model: DATASET_COLORS_5V5[i % len(DATASET_COLORS_5V5)] for i, model in enumerate(models)}

    # Y-axis scaling (98th percentile to ignore extreme outliers)
    ymin = sub_df[metric].min()
    ymax_robust = np.percentile(sub_df[metric], 98)
    padding = (ymax_robust - ymin) * 0.05
    
    ylim_min = max(0, ymin - padding)
    ylim_max = ymax_robust + padding
    
    # --- 1. Global Performance Summary ---
    st.subheader("1. Global Performance Summary")
    summary = sub_df.groupby('model')[metric].agg(['mean', 'std', 'min', 'max', 'count']).sort_values('mean')
    summary.columns = ['Mean', 'Std Dev', 'Min', 'Max', 'Samples']
    st.dataframe(summary.style.highlight_min(subset=['Mean'], color='lightgreen', axis=0).format("{:.4f}"))

    # --- 2. Error Distribution (Global) ---
    st.subheader("2. Error Distribution Comparison")
    fig_box, ax_box = get_styled_figure_ax(figsize=(12, 6), aspect='auto')
    sns.boxplot(data=sub_df, x='model', y=metric, ax=ax_box, palette=model_colors, order=models)
    ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45, ha='right')
    ax_box.set_title(f"{channel_name}: {metric.upper()} Distribution by Model")
    ax_box.set_ylim(ylim_min, ylim_max) 
    save_fig_to_report(fig_box, f"dist_global_{channel_name}_{metric}")
    st.pyplot(fig_box)

    # --- 3. Known vs Unknown Cities ---
    if 'is_known_city' in sub_df.columns:
        st.subheader("3. Known vs Unknown Cities")
        st.markdown("Does the model generalize to unseen cities?")
        
        fig_known, ax_known = get_styled_figure_ax(figsize=(12, 6), aspect='auto')
        sns.boxplot(
            data=sub_df, 
            x='model', 
            y=metric, 
            hue='is_known_city', 
            ax=ax_known, 
            palette="coolwarm",
            order=models
        )
        ax_known.set_title(f"{channel_name}: {metric.upper()} - Known vs Unknown Cities")
        ax_known.set_xticklabels(ax_known.get_xticklabels(), rotation=45, ha='right')
        ax_known.set_ylim(ylim_min, ylim_max)
        save_fig_to_report(fig_known, f"dist_known_unknown_{channel_name}_{metric}")
        st.pyplot(fig_known)

        # Statistical Test for Known vs Unknown
        st.markdown("#### Statistical Significance (Mann-Whitney U Test)")
        st.markdown("Testing if the error distribution significantly differs between Known and Unknown cities.")
        
        known_unknown_stats = []
        for model in models:
            model_data = sub_df[sub_df['model'] == model]
            
            # identify groups assuming boolean or 0/1
            known_mask = model_data['is_known_city'].astype(bool) == True
            
            known_group = model_data[known_mask][metric]
            unknown_group = model_data[~known_mask][metric]
            
            res = {'Model': model}
            
            if len(known_group) > 0 and len(unknown_group) > 0:
                # Mann-Whitney U test
                try:
                    stat, p_val = mannwhitneyu(known_group, unknown_group, alternative='two-sided')
                except ValueError:
                    p_val = 1.0
                    
                res['Mean (Known)'] = known_group.mean()
                res['Mean (Unknown)'] = unknown_group.mean()
                res['Diff'] = res['Mean (Unknown)'] - res['Mean (Known)']
                res['p-value'] = p_val
            else:
                res['Mean (Known)'] = known_group.mean() if len(known_group) > 0 else np.nan
                res['Mean (Unknown)'] = unknown_group.mean() if len(unknown_group) > 0 else np.nan
                res['Diff'] = np.nan
                res['p-value'] = np.nan
            
            known_unknown_stats.append(res)
        
        if known_unknown_stats:
            ku_df = pd.DataFrame(known_unknown_stats).set_index('Model')
            
            def color_p_value(val):
                if isinstance(val, float) and val < 0.05:
                    return 'color: red; font-weight: bold'
                return ''
            
            st.dataframe(
                ku_df.style.format("{:.4f}", subset=['Mean (Known)', 'Mean (Unknown)', 'Diff', 'p-value'])
                .applymap(color_p_value, subset=['p-value'])
            )

    # --- 4. Statistical Correlations (Lat/Lon/Count/Year) ---
    st.subheader("4. Statistical Correlations (Lat/Lon/Count/Year)")
    st.markdown("Pearson correlation between performance metric and key metadata.")
    
    correlation_results = []
    for model in models:
        model_df = sub_df[sub_df['model'] == model]
        
        # Metric vs Lat
        if 'lat' in model_df.columns and len(model_df) > 1:
            r_lat, p_lat = pearsonr(model_df[metric], model_df['lat'])
        else:
            r_lat, p_lat = np.nan, np.nan
            
        # Metric vs Lon
        if 'lon' in model_df.columns and len(model_df) > 1:
            r_lon, p_lon = pearsonr(model_df[metric], model_df['lon'])
        else:
            r_lon, p_lon = np.nan, np.nan
            
        # Metric vs Sample Count
        if 'city_sample_count' in model_df.columns and len(model_df) > 1:
            r_count, p_count = pearsonr(model_df[metric], model_df['city_sample_count'])
        else:
            r_count, p_count = np.nan, np.nan

        # Metric vs Year (t1_year)
        if 't1_year' in model_df.columns and len(model_df) > 1 and model_df['t1_year'].nunique() > 1:
            r_year, p_year = pearsonr(model_df[metric], model_df['t1_year'])
        else:
            r_year, p_year = np.nan, np.nan

        # Metric vs Year (Known Cities)
        if 't1_year' in model_df.columns and 'is_known_city' in model_df.columns:
            known_mask = model_df['is_known_city'].astype(bool) == True
            known_df = model_df[known_mask]
            if len(known_df) > 1 and known_df['t1_year'].nunique() > 1:
                r_year_k, p_year_k = pearsonr(known_df[metric], known_df['t1_year'])
            else:
                r_year_k, p_year_k = np.nan, np.nan
        else:
            r_year_k, p_year_k = np.nan, np.nan

        # Metric vs Year (Unknown Cities)
        if 't1_year' in model_df.columns and 'is_known_city' in model_df.columns:
            unknown_mask = model_df['is_known_city'].astype(bool) == False
            unknown_df = model_df[unknown_mask]
            if len(unknown_df) > 1 and unknown_df['t1_year'].nunique() > 1:
                r_year_u, p_year_u = pearsonr(unknown_df[metric], unknown_df['t1_year'])
            else:
                r_year_u, p_year_u = np.nan, np.nan
        else:
            r_year_u, p_year_u = np.nan, np.nan
            
        correlation_results.append({
            'Model': model,
            'Lat (r)': r_lat, 'Lat (p)': p_lat,
            'Lon (r)': r_lon, 'Lon (p)': p_lon,
            'Count (r)': r_count, 'Count (p)': p_count,
            'Year (r)': r_year, 'Year (p)': p_year,
            'Year_Known (r)': r_year_k, 'Year_Known (p)': p_year_k,
            'Year_Unknown (r)': r_year_u, 'Year_Unknown (p)': p_year_u
        })
        
    corr_df = pd.DataFrame(correlation_results).set_index('Model')
    
    # Formatting for display
    def color_significant(val):
        if isinstance(val, float) and val < 0.05:
            return 'color: red; font-weight: bold'
        return ''

    st.dataframe(
        corr_df.style.format("{:.3f}")
        .applymap(color_significant, subset=['Lat (p)', 'Lon (p)', 'Count (p)', 'Year (p)', 'Year_Known (p)', 'Year_Unknown (p)'])
    )


    # --- 5. Spatial Analysis (Lat/Lon) ---
    st.subheader("5. Spatial Analysis (Lat/Lon)")
    col1, col2 = st.columns(2)
    
    if channel_name.lower() == 'after_temp':
        _ymax = 11
        _ymin = 2
    else:
        _ymax = 0.15
        _ymin = 0.05
    with col1:
        if 'lat' in sub_df.columns:
            st.markdown("**Performance vs Latitude**")
            fig_lat, ax_lat = get_styled_figure_ax(figsize=(8, 6), aspect='auto')
            for model in models:
                model_data = sub_df[sub_df['model'] == model]
                sns.regplot(
                    data=model_data, x='lat', y=metric, 
                    ax=ax_lat, scatter=False, label=model, ci=None,
                    line_kws={'color': model_colors[model]}
                )
            ax_lat.set_title(f"Trend: {metric.upper()} vs Latitude")
            ax_lat.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_lat.set_ylim(_ymin, _ymax)
            style_legend(ax_lat, ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
            save_fig_to_report(fig_lat, f"trend_lat_{channel_name}_{metric}")
            st.pyplot(fig_lat)

    with col2:
        if 'lon' in sub_df.columns:
            st.markdown("**Performance vs Longitude**")
            fig_lon, ax_lon = get_styled_figure_ax(figsize=(8, 6), aspect='auto')
            for model in models:
                model_data = sub_df[sub_df['model'] == model]
                sns.regplot(
                    data=model_data, x='lon', y=metric, 
                    ax=ax_lon, scatter=False, label=model, ci=None,
                    line_kws={'color': model_colors[model]}
                )
            ax_lon.set_title(f"Trend: {metric.upper()} vs Longitude")
            ax_lon.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_lon.set_ylim(_ymin, _ymax)
            style_legend(ax_lon, ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
            save_fig_to_report(fig_lon, f"trend_lon_{channel_name}_{metric}")
            st.pyplot(fig_lon)

    # --- 6. Impact of Data Availability (City Sample Count) ---
    if 'city_sample_count' in sub_df.columns:
        st.subheader("6. Impact of Data Availability (City Sample Count)")
        st.markdown("Do models perform better on cities with more data?")
        
        fig_cnt, ax_cnt = get_styled_figure_ax(figsize=(10, 6), aspect='auto')
        for model in models:
            model_data = sub_df[sub_df['model'] == model]
            sns.regplot(
                data=model_data, x='city_sample_count', y=metric, 
                ax=ax_cnt, scatter=False, label=model, ci=None,
                line_kws={'color': model_colors[model]} # Use assigned color
            )
        ax_cnt.set_title(f"Trend: {metric.upper()} vs City Sample Count")
        # ax_cnt.legend()
        style_legend(ax_cnt, ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
        # ax_cnt.set_ylim(ylim_min, sub_df[metric].max())
        if channel_name.lower() == 'after_ndvi':
            _ymin=0.01
        ax_cnt.set_ylim(_ymin, _ymax)
        save_fig_to_report(fig_cnt, f"trend_count_{channel_name}_{metric}")
        st.pyplot(fig_cnt)

    # --- 7. Pairwise Statistical Tests ---
    st.subheader("7. Statistical Significance (Pairwise Wilcoxon)")
    
    # Pivot for paired tests
    sub_df['unique_id'] = sub_df['sample_idx'].astype(str) + "_" + sub_df['city'] # Ensure unique ID
    pivot_df = sub_df.pivot_table(index='unique_id', columns='model', values=metric).dropna()
    
    if pivot_df.empty:
        st.warning("Not enough overlapping samples for statistical tests.")
    else:
        p_values = pd.DataFrame(index=models, columns=models, dtype=float)
        for m1 in models:
            for m2 in models:
                if m1 == m2:
                    p_values.loc[m1, m2] = np.nan
                else:
                    try:
                        stat, p = wilcoxon(pivot_df[m1], pivot_df[m2])
                        p_values.loc[m1, m2] = p
                    except ValueError:
                        p_values.loc[m1, m2] = 1.0

        fig_sig, ax_sig = get_styled_figure_ax(figsize=(10, 8), aspect='auto', grid=False)
        sns.heatmap(
            p_values.astype(float), 
            annot=True, 
            fmt=".1e", 
            cmap="Greens_r", 
            ax=ax_sig,
            vmax=0.05,
            cbar_kws={'label': 'p-value'}
        )
        ax_sig.set_title(f"Pairwise Wilcoxon P-Values ({channel_name})")
        save_fig_to_report(fig_sig, f"sig_matrix_{channel_name}_{metric}")
        st.pyplot(fig_sig)

        # --- 8. Sample-wise Error Correlation (Exact Location) ---
        st.subheader("8. Sample-wise Error Correlation (Exact Location)")
        st.markdown("Do models make errors on the same samples? (Pearson Correlation of sample-wise errors)")
    
        if pivot_df.empty:
            st.warning("Not enough overlapping samples for correlation analysis.")
        else:
            # Calculate correlation matrix
            corr_matrix = pivot_df.corr(method='pearson')
    
            # Mask the upper triangle to avoid duplicates
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
            # Adjust matrix and mask to remove the masked first row and last column from being drawn
            if not corr_matrix.empty and corr_matrix.shape[0] > 1 and corr_matrix.shape[1] > 1:
                corr_matrix_display = corr_matrix.iloc[1:, :-1]
                mask_display = mask[1:, :-1]
            else:
                corr_matrix_display = corr_matrix
                mask_display = mask

            # Before plotting, remove the 'model' title from the axes
            corr_matrix_display.index.name = None
            corr_matrix_display.columns.name = None

            fig_corr, ax_corr = get_styled_figure_ax(figsize=(10, 8), aspect='auto', grid=False)
            sns.heatmap(
                corr_matrix_display, # Use the sliced matrix
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax_corr,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson r'},
                mask=mask_display # Use the sliced mask
            )
            # ax_corr.set_title(f"Sample-wise Error Correlation ({channel_name})")
            plt.xticks(rotation=30) 
            save_fig_to_report(fig_corr, f"error_corr_matrix_{channel_name}_{metric}")
            st.pyplot(fig_corr)
    
        # --- 9. Temporal Error Correlation (Time Delta) ---
        st.subheader("9. Temporal Error Correlation (Time Delta)")
    
        if 'time_delta' not in sub_df.columns:
             st.info("Column 'time_delta' not found in data. Please re-run evaluation with the latest script.")
        else:
            st.markdown("Do models share the same performance trends across different time intervals?")
    
            # Group by time_delta and model, calculate mean metric
            temporal_df = sub_df.groupby(['time_delta', 'model'])[metric].mean().reset_index()
    
            # Pivot: Index=time_delta, Columns=model, Values=mean_error
            temporal_pivot = temporal_df.pivot(index='time_delta', columns='model', values=metric).dropna()
    
            if temporal_pivot.shape[0] < 2:
                st.warning("Not enough unique time deltas for correlation.")
            else:
                t_corr = temporal_pivot.corr(method='pearson')
                t_mask = np.triu(np.ones_like(t_corr, dtype=bool))
    
                # Adjust matrix and mask to remove the masked first row and last column from being drawn
                if not t_corr.empty and t_corr.shape[0] > 1 and t_corr.shape[1] > 1:
                    t_corr_display = t_corr.iloc[1:, :-1]
                    t_mask_display = t_mask[1:, :-1]
                else:
                    t_corr_display = t_corr
                    t_mask_display = t_mask
                    
                # Before plotting, remove the 'model' title from the axes
                t_corr_display.index.name = None
                t_corr_display.columns.name = None

                fig_t, ax_t = get_styled_figure_ax(figsize=(10, 8), aspect='auto', grid=False)
                sns.heatmap(
                    t_corr_display, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax_t, vmin=-1, vmax=1, mask=t_mask_display,
                    cbar_kws={'label': 'Pearson r'}
                )
                plt.xticks(rotation=30) 
                # ax_t.set_title(f"Temporal Trend Correlation ({channel_name})")
                save_fig_to_report(fig_t, f"temporal_corr_{channel_name}_{metric}")
                st.pyplot(fig_t)
    
        # --- 10. Regional Error Correlation (Latitude Bands) ---
        st.subheader("10. Regional Error Correlation (Latitude Bands)")
    
        if 'lat' not in sub_df.columns:
            st.info("Column 'lat' not found in data.")
        else:
            st.markdown("Do models share the same performance trends across latitudes? (Binned by 5 degrees)")
    
            # Create bins
            # Use pandas cut
            sub_df['lat_bin'] = pd.cut(sub_df['lat'], bins=range(-90, 95, 5))
    
            # Group
            lat_df = sub_df.groupby(['lat_bin', 'model'])[metric].mean().reset_index()
    
            # Pivot
            lat_pivot = lat_df.pivot(index='lat_bin', columns='model', values=metric).dropna()
    
            if lat_pivot.shape[0] < 2:
                 st.warning("Not enough populated latitude bins for correlation.")
            else:
                l_corr = lat_pivot.corr(method='pearson')
                l_mask = np.triu(np.ones_like(l_corr, dtype=bool))
    
                # Adjust matrix and mask to remove the masked first row and last column from being drawn
                if not l_corr.empty and l_corr.shape[0] > 1 and l_corr.shape[1] > 1:
                    l_corr_display = l_corr.iloc[1:, :-1]
                    l_mask_display = l_mask[1:, :-1]
                else:
                    l_corr_display = l_corr
                    l_mask_display = l_mask

                # Before plotting, remove the 'model' title from the axes
                l_corr_display.index.name = None
                l_corr_display.columns.name = None

                fig_l, ax_l = get_styled_figure_ax(figsize=(10, 8), aspect='auto', grid=False)
                sns.heatmap(
                    l_corr_display, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax_l, vmin=-1, vmax=1, mask=l_mask_display,
                    cbar_kws={'label': 'Pearson r'}
                )
                plt.xticks(rotation=30) 
                # ax_l.set_title(f"Regional Trend Correlation ({channel_name})")
                save_fig_to_report(fig_l, f"regional_corr_{channel_name}_{metric}")
                st.pyplot(fig_l)
    
    # --- 11. 2D Spatial Error Correlation (Lat-Lon Grid) ---
    st.subheader("11. 2D Spatial Error Correlation (Lat-Lon Grid)")

    if 'lat' not in sub_df.columns or 'lon' not in sub_df.columns:
        st.info("Column 'lat' or 'lon' not found in data.")
    else:
        st.markdown("Do models share the same performance trends across spatial regions? (Binned by 10x10 degrees)")
        
        # Create bins
        sub_df['lat_bin_2d'] = pd.cut(sub_df['lat'], bins=range(-90, 95, 10))
        sub_df['lon_bin_2d'] = pd.cut(sub_df['lon'], bins=range(-180, 185, 10))
        
        # Group
        # observed=True ensures we only get bins that actually have data
        spatial_df = sub_df.groupby(['lat_bin_2d', 'lon_bin_2d', 'model'], observed=True)[metric].mean().reset_index()
        
        # Create a combined string for the index
        spatial_df['grid_cell'] = spatial_df['lat_bin_2d'].astype(str) + " x " + spatial_df['lon_bin_2d'].astype(str)
        
        # Pivot
        spatial_pivot = spatial_df.pivot(index='grid_cell', columns='model', values=metric).dropna()
        
        if spatial_pivot.shape[0] < 2:
             st.warning("Not enough populated spatial grid cells for correlation.")
        else:
            s_corr = spatial_pivot.corr(method='pearson')
            s_mask = np.triu(np.ones_like(s_corr, dtype=bool))
            
            # Adjust matrix and mask
            if not s_corr.empty and s_corr.shape[0] > 1 and s_corr.shape[1] > 1:
                s_corr_display = s_corr.iloc[1:, :-1]
                s_mask_display = s_mask[1:, :-1]
            else:
                s_corr_display = s_corr
                s_mask_display = s_mask

            # Remove axes labels
            s_corr_display.index.name = None
            s_corr_display.columns.name = None

            fig_s, ax_s = get_styled_figure_ax(figsize=(10, 8), aspect='auto', grid=False)
            sns.heatmap(
                s_corr_display, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax_s, vmin=-1, vmax=1, mask=s_mask_display,
                cbar_kws={'label': 'Pearson r'}
            )
            plt.xticks(rotation=30) 
            # ax_s.set_title(f"2D Spatial Trend Correlation ({channel_name})")
            save_fig_to_report(fig_s, f"spatial_2d_corr_{channel_name}_{metric}")
            st.pyplot(fig_s)
    
    st.markdown("---")
def statistical_page():
    st.title("Statistical Model Comparison")

    st.sidebar.header("Data Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload evaluation CSVs",
        type=['csv'],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload one or more evaluation CSV files (e.g. from `reports/tests/`).")
        st.stop()

    # --- Data Loading ---
    full_df = load_and_process_data(uploaded_files)
    if full_df.empty:
        st.error("Could not load data.")
        st.stop()

    # --- Global Settings ---
    st.sidebar.header("Analysis Settings")
    metric = st.sidebar.selectbox("Metric", ['mae', 'rmse'])
    
    available_channels = sorted(full_df['channel'].unique())
    
    # Channel Selection
    channel_mode = st.sidebar.radio("Channel Analysis Mode", ["All Separately", "Select Specific"])
    
    selected_channels = []
    if channel_mode == "All Separately":
        selected_channels = available_channels
    else:
        selected_channels = st.sidebar.multiselect("Select Channels", available_channels, default=available_channels[:1])

    if not selected_channels:
        st.warning("Please select at least one channel.")
        st.stop()

    # --- Run Analysis per Channel ---
    for channel in selected_channels:
        run_analysis_for_channel(full_df, channel, metric)

if __name__ == "__main__":
    statistical_page()

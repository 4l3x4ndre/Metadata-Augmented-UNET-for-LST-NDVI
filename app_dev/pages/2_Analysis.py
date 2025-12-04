# app/pages/2_Analysis.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from src.utils.plot_utils import style_legend, get_styled_figure_ax, DATASET_COLORS, DATASET_COLORS_5V5, convert_label,DATASET_COLORS_5


def save_fig_to_report(fig, filename):
    """Saves the given figure to the reports directory."""
    output_dir = "reports/tests/app/analysis"
    os.makedirs(output_dir, exist_ok=True)
    safe_filename = filename.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    if not safe_filename.endswith('.pdf'):
        safe_filename += '.pdf'
    path = os.path.join(output_dir, safe_filename)
    fig.savefig(path, bbox_inches='tight')

def analysis_page():
    # Deterministic mapping for Dynamic World classes to numerical IDs
    DW_CLASS_MAPPING = {
        'Water': 1,
        'Trees': 2,
        'Grass': 3,
        'F.V.': 4, # flooded_vegetation
        'Crops': 5,
        'Shrub': 6, # shrub_and_scrub
        'Bare': 7,
        'Built': 8,
        'Snow': 9 # snow_ice
    }

    st.title("Model Performance Analysis")

    st.sidebar.header("Data Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload evaluation CSVs",
        type=['csv'],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload one or more evaluation CSV files to begin analysis.")
        st.stop()

    dfs = []
    for uploaded_file in uploaded_files:
        raw_model_name = os.path.basename(uploaded_file.name).replace('_evaluation.csv', '')
        df = pd.read_csv(uploaded_file)
        
        # Parse the raw_model_name to extract model and embedding types
        parts = raw_model_name.split('_')
        
        # Check if the filename has enough parts to extract model and embedding type
        # Expected format: prefix_model_embedding_...
        # Example: lgs-2c-future-fixed-eval_unet_emb_1_job23676307
        if len(parts) >= 3:
            model_type = parts[1].upper() # Capitalize model type (e.g., 'UNET')
            embedding_type_raw = parts[2] # Raw embedding type (e.g., 'emb')
            embedding_type_formatted = convert_label(embedding_type_raw) # Formatted embedding type (e.g., 'Embeddings')
            
            if model_type == 'UNET++':
                model_label = f"UNet++"
            else:
                model_label = f"{embedding_type_formatted}"
        else:
            # Fallback if the filename format is not as expected
            model_label = raw_model_name

        df['model'] = model_label
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    # --- Overall Model Comparison ---
    st.header("Overall Model Comparison")
    overall_metrics = full_df[full_df['dw_class'] == 'overall'].copy()
    
    if overall_metrics.empty:
        st.warning("No 'overall' metrics found in the uploaded file(s).")
    else:
        metric_to_plot = st.selectbox("Select Metric for Overview", ['mae', 'rmse'], key='overview_metric')

        if metric_to_plot:
            st.subheader(f"Distribution of {metric_to_plot.upper()}")
            
            channels = overall_metrics['channel'].unique()
            temp_channels = [c for c in channels if 'temp' in c.lower()]
            ndvi_channels = [c for c in channels if 'ndvi' in c.lower()]
            other_channels = [c for c in channels if c not in temp_channels and c not in ndvi_channels]

            def create_violin_plot(df, title, filename_suffix):
                fig, ax = get_styled_figure_ax(figsize=(12, 6), aspect='auto')
                sns.violinplot(
                    data=df,
                    x='channel',
                    y=metric_to_plot,
                    hue='model',
                    ax=ax,
                    palette=DATASET_COLORS_5V5 if len(df['model'].unique()) <= len(DATASET_COLORS_5V5) else None
                )
                # ax.set_title(title)
                ax.set_ylabel(metric_to_plot.upper())
                ax.set_xlabel('Channel')
                style_legend(ax, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
                # style_legend(ax, ncol=3, bbox_to_anchor=(0.5, 1.23), loc='upper center')
                
                save_fig_to_report(fig, f"violin_{metric_to_plot}_{filename_suffix}")
                st.pyplot(fig)

            if temp_channels:
                create_violin_plot(overall_metrics[overall_metrics['channel'].isin(temp_channels)], f"Temperature Channels Performance ({metric_to_plot.upper()})", "temp")
            
            if ndvi_channels:
                create_violin_plot(overall_metrics[overall_metrics['channel'].isin(ndvi_channels)], f"NDVI Channels Performance ({metric_to_plot.upper()})", "ndvi")

            if other_channels:
                create_violin_plot(overall_metrics[overall_metrics['channel'].isin(other_channels)], f"Other Channels Performance ({metric_to_plot.upper()})", "other")

    # --- Temporal Sample Distribution ---
    st.header("Temporal Sample Distribution")
    if 't1_year' in full_df.columns and 'is_known_city' in full_df.columns:
        # Deduplicate to get unique samples
        unique_samples = full_df[['sample_idx', 't1_year', 'is_known_city']].drop_duplicates()
        
        # Prepare data for plotting
        years = sorted(unique_samples['t1_year'].unique())
        plot_data = []
        
        for year in years:
            year_samples = unique_samples[unique_samples['t1_year'] == year]
            
            # Known
            plot_data.append({
                'Year': year,
                'Count': len(year_samples[year_samples['is_known_city'] == True]),
                'Category': 'Known Cities'
            })
            
            # Unseen
            plot_data.append({
                'Year': year,
                'Count': len(year_samples[year_samples['is_known_city'] == False]),
                'Category': 'Unseen Cities'
            })
            
        plot_df = pd.DataFrame(plot_data)
        
        fig_temp_dist, ax1 = get_styled_figure_ax(figsize=(12, 6), aspect='auto')
        ax2 = ax1.twinx()
        
        known_data = plot_df[plot_df['Category'] == 'Known Cities']
        unseen_data = plot_df[plot_df['Category'] == 'Unseen Cities']
        
        # Plot Known on ax1 (Solid)
        l1 = ax1.plot(known_data['Year'], known_data['Count'], color='black', linestyle='-', marker='o', label='Known Cities (Left)')
        
        # Plot Unseen on ax2 (Dashed)
        l2 = ax2.plot(unseen_data['Year'], unseen_data['Count'], color='black', linestyle='--', marker='s', label='Unseen Cities (Right)')
        
        ax1.set_ylabel("Count (Known Cities)")
        ax2.set_ylabel("Count (Unseen Cities)")
        ax1.set_xlabel("Year")
        
        # Combine legends
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
        
        save_fig_to_report(fig_temp_dist, "temporal_sample_distribution")
        st.pyplot(fig_temp_dist)

    # --- Temporal Distance Analysis ---
    st.header("Performance over Temporal Distance")
    if 't1_year' not in full_df.columns or 'is_known_city' not in full_df.columns:
        st.warning("Temporal analysis requires 't1_year' and 'is_known_city' columns in the CSV.")
    else:
        metric_temporal = st.selectbox("Select Metric", ['mae', 'rmse'], key='temporal_metric')

        if metric_temporal:
            temporal_df = full_df[full_df['dw_class'] == 'overall']
            temporal_agg = temporal_df.groupby(['t1_year', 'is_known_city', 'model', 'channel'])[metric_temporal].mean().reset_index()
            temporal_agg['is_known_city'] = temporal_agg['is_known_city'].map({True: 'Known Cities', False: 'Unknown Cities'})

            channels = temporal_agg['channel'].unique()
            for channel in channels:
                st.subheader(f"Temporal Performance: {channel}")
                channel_data = temporal_agg[temporal_agg['channel'] == channel]
                
                fig, ax = get_styled_figure_ax(figsize=(12, 6), aspect='auto')
                sns.lineplot(
                    data=channel_data,
                    x='t1_year',
                    y=metric_temporal,
                    hue='model',
                    style='is_known_city',
                    markers=True,
                    dashes=True,
                    ax=ax,
                    palette=DATASET_COLORS_5 if len(channel_data['model'].unique()) <= len(DATASET_COLORS_5) else None,
                    linewidth=2.5,
                )
                # ax.set_title(f"Temporal Performance ({metric_temporal.upper()}) - {channel}")
                ax.set_ylabel(metric_temporal.upper())
                ax.set_xlabel('Year of First Image (t1)')
                
                # Get legend handles and labels
                handles, labels = ax.get_legend_handles_labels()

                # Separate handles and labels based on exact matches with data values
                unique_models = set(temporal_agg['model'].unique())
                unique_cities = set(temporal_agg['is_known_city'].unique())

                model_handles = []
                model_labels = []
                city_handles = []
                city_labels = []

                for h, l in zip(handles, labels):
                    if l in unique_models:
                        model_handles.append(h)
                        model_labels.append(l)
                    elif l in unique_cities:
                        city_handles.append(h)
                        city_labels.append(l)

                # Remove the original combined legend
                if ax.get_legend():
                    ax.get_legend().remove()
                
                # Create the first legend for 'Model'
                if model_handles:
                    legend1 = ax.legend(model_handles, model_labels, title='Model', loc='center left', bbox_to_anchor=(1, 0.75), ncol=1, frameon=False)
                    ax.add_artist(legend1)
                
                # Create the second legend for 'City Category'
                if city_handles:
                    ax.legend(city_handles, city_labels, title='City Category', loc='center left', bbox_to_anchor=(1, 0.25), ncol=1, frameon=False)
                
                save_fig_to_report(fig, f"temporal_{metric_temporal}_{channel}")
                st.pyplot(fig)

    # --- Seasonal Performance Analysis ---
    st.header("Performance over Seasonality (Month)")
    if 't2_month' not in full_df.columns:
         st.warning("Seasonal analysis requires 't2_month' column in the CSV. Please ensure your evaluation CSV contains this column.")
    else:
        metric_seasonal = st.selectbox("Select Metric for Seasonality", ['mae', 'rmse'], key='seasonal_metric')
        
        if metric_seasonal:
            seasonal_df = full_df[full_df['dw_class'] == 'overall']
            # Group by month, ignoring year (averaging across years)
            seasonal_agg = seasonal_df.groupby(['t2_month', 'is_known_city', 'model', 'channel'])[metric_seasonal].mean().reset_index()
            seasonal_agg['is_known_city'] = seasonal_agg['is_known_city'].map({True: 'Known Cities', False: 'Unknown Cities'})
            
            channels = seasonal_agg['channel'].unique()
            for channel in channels:
                st.subheader(f"Seasonal Performance: {channel}")
                channel_data = seasonal_agg[seasonal_agg['channel'] == channel]
                
                fig, ax = get_styled_figure_ax(figsize=(12, 6), aspect='auto')
                sns.lineplot(
                    data=channel_data,
                    x='t2_month',
                    y=metric_seasonal,
                    hue='model',
                    style='is_known_city',
                    markers=True,
                    dashes=True,
                    ax=ax,
                    palette=DATASET_COLORS_5 if len(channel_data['model'].unique()) <= len(DATASET_COLORS_5) else None,
                    linewidth=2.5,
                )
                # ax.set_title(f"Seasonal Performance ({metric_seasonal.upper()}) - {channel}")
                ax.set_ylabel(metric_seasonal.upper())
                ax.set_xlabel('Month of Target Image (t2)')
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                
                # Get legend handles and labels
                handles, labels = ax.get_legend_handles_labels()

                # Separate handles and labels based on exact matches with data values
                unique_models = set(seasonal_agg['model'].unique())
                unique_cities = set(seasonal_agg['is_known_city'].unique())

                model_handles = []
                model_labels = []
                city_handles = []
                city_labels = []

                for h, l in zip(handles, labels):
                    if l in unique_models:
                        model_handles.append(h)
                        model_labels.append(l)
                    elif l in unique_cities:
                        city_handles.append(h)
                        city_labels.append(l)

                # Remove the original combined legend
                if ax.get_legend():
                    ax.get_legend().remove()
                
                # Create the first legend for 'Model'
                if model_handles:
                    legend1 = ax.legend(model_handles, model_labels, title='Model', loc='center left', bbox_to_anchor=(1, 0.75), ncol=1, frameon=False)
                    ax.add_artist(legend1)
                
                # Create the second legend for 'City Category'
                if city_handles:
                    ax.legend(city_handles, city_labels, title='City Category', loc='center left', bbox_to_anchor=(1, 0.25), ncol=1, frameon=False)
                
                save_fig_to_report(fig, f"seasonal_{metric_seasonal}_{channel}")
                st.pyplot(fig)

    # --- Performance by Dynamic World Category ---
    st.header("Performance by Dynamic World Category")
    dw_classes = [c for c in full_df['dw_class'].unique() if c != 'overall']
    
    if dw_classes:
        metric_dw = st.selectbox("Select Metric", ['mae', 'rmse'], key='dw_metric')

        if metric_dw:
            dw_metrics = full_df[full_df['dw_class'] != 'overall']
            channels = dw_metrics['channel'].unique()

            for channel in channels:
                st.subheader(f"DW Performance for Channel: {channel}")
                channel_dw_metrics = dw_metrics[dw_metrics['channel'] == channel]

                if channel_dw_metrics.empty:
                    st.warning(f"No data for channel {channel} in DW analysis.")
                    continue

                dw_agg = channel_dw_metrics.groupby(['dw_class', 'model'])[metric_dw].mean().reset_index()
                
                # Apply renaming of classes
                class_renaming_map = {
                    'shrub_and_scrub': 'Shrub',
                    'flooded_vegetation': 'F.V.',
                    'snow_and_ice': 'Snow'
                }
                dw_agg['dw_class'] = dw_agg['dw_class'].replace(class_renaming_map)
                dw_agg['dw_class'] = dw_agg['dw_class'].apply(lambda x: x.title() if isinstance(x, str) else x)
                
                min_metric_per_dw_class = dw_agg.groupby('dw_class')[metric_dw].min().sort_values()
                dw_class_order_str = min_metric_per_dw_class.index.tolist()
                

                fig_dw, ax = get_styled_figure_ax(figsize=(20, 8), aspect='auto')
                sns.barplot(
                    data=dw_agg,
                    x='dw_class', 
                    y=metric_dw,
                    hue='model',
                    ax=ax,
                    palette=DATASET_COLORS_5V5 if len(dw_agg['model'].unique()) <= len(DATASET_COLORS_5V5) else None,
                    order=dw_class_order_str 
                )
                # ax.set_title(f"Performance by DW Class for {channel} ({metric_dw.upper()})")
                if metric_dw.lower() == 'mae':
                    ax.set_ylabel('Mean Absolute Error (MAE)')
                elif metric_dw.lower() == 'rmse':
                    ax.set_ylabel('Root Mean Square Error (RMSE)')
                else:
                    ax.set_ylabel(metric_dw.upper())
                ax.set_xlabel('Dynamic World Class') 
                # ax.tick_params(axis='x', rotation=45) # Re-enabled rotation
                # style_legend(ax, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
                style_legend(ax, ncol=5, bbox_to_anchor=(0.5, 1.1), loc='upper center')
                
                save_fig_to_report(fig_dw, f"dw_performance_{metric_dw}_{channel}")
                st.pyplot(fig_dw)


    # --- Best and Worst Performing Samples ---
    st.header("Best and Worst Performing Samples")
    metric_bw = st.selectbox("Select Metric for Best/Worst Analysis", ['mae', 'rmse'], key='bw_metric')
    if metric_bw:
        sample_metrics = full_df[full_df['dw_class'] == 'overall'].groupby(['model', 'sample_idx'])[metric_bw].mean().reset_index()

        if not sample_metrics.empty:
            best_sample_row = sample_metrics.loc[sample_metrics[metric_bw].idxmin()]
            st.subheader(f"🏆 Best Performing Sample (Lowest Mean {metric_bw.upper()})")
            st.write(f"Model: **{best_sample_row['model']}**")
            st.write(f"Sample Index: **{best_sample_row['sample_idx']}**")
            st.write(f"Mean {metric_bw.upper()}: **{best_sample_row[metric_bw]:.4f}**")

            worst_sample_row = sample_metrics.loc[sample_metrics[metric_bw].idxmax()]
            st.subheader(f"📉 Worst Performing Sample (Highest Mean {metric_bw.upper()})")
            st.write(f"Model: **{worst_sample_row['model']}**")
            st.write(f"Sample Index: **{worst_sample_row['sample_idx']}**")
            st.write(f"Mean {metric_bw.upper()}: **{worst_sample_row[metric_bw]:.4f}**")

    # --- Geospatial Performance Analysis ---
    st.header("Geospatial Performance Analysis")
    
    # Check for necessary columns
    if 'lat' not in full_df.columns or 'lon' not in full_df.columns:
        st.warning("Geospatial analysis requires 'lat' and 'lon' columns in the CSV.")
    else:
        # Model Selector
        models = full_df['model'].unique()
        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox("Select Model for Geospatial Analysis", models, key='geo_model')
        with col2:
            channel_type = st.selectbox("Select Channel Type", ["Temperature", "NDVI", "All"], key='geo_channel')
        
        # Bin Selector
        n_bins = st.slider("Number of Spatial Bins (Grid Resolution)", min_value=5, max_value=50, value=50, step=1)
        
        # Filter data
        geo_df = full_df[(full_df['model'] == selected_model) & (full_df['dw_class'] == 'overall')].copy()

        # Filter by channel type
        if channel_type == "Temperature":
            geo_df = geo_df[geo_df['channel'].str.contains('temp', case=False)]
        elif channel_type == "NDVI":
            geo_df = geo_df[geo_df['channel'].str.contains('ndvi', case=False)]
        
        if geo_df.empty:
             st.warning(f"No data found for model {selected_model} with dw_class='overall' and channel_type='{channel_type}'.")
        else:
            # Binning logic
            geo_df['lat_bin'] = pd.cut(geo_df['lat'], bins=n_bins)
            geo_df['lon_bin'] = pd.cut(geo_df['lon'], bins=n_bins)
            
            # 1. Metric Heatmap
            st.subheader(f"Spatial Distribution of {metric_to_plot.upper()} ({channel_type})")
            pivot_metric = geo_df.pivot_table(
                index='lat_bin', 
                columns='lon_bin', 
                values=metric_to_plot, 
                aggfunc='mean'
            )
            # Sort index/columns for proper display (descending lat, ascending lon)
            pivot_metric = pivot_metric.sort_index(ascending=False)
            
            fig_geo_metric, ax_geo_metric = get_styled_figure_ax(figsize=(12, 10), aspect='auto', grid=False)
            ax_geo_metric.set_facecolor('white') # Ensure axes background is white

            # Create a colormap and set NaN values to white
            cmap_metric = sns.color_palette("flare", as_cmap=True) # Use the current colormap
            cmap_metric.set_bad('white') # Set color for NaN values to white
            # Pivot table index/columns are CategoricalIndex of Interval
            
            # Helper to get midpoints safely
            def get_bin_midpoints(index):
                return [(interval.left + interval.right)/2 for interval in index]

            lat_midpoints = get_bin_midpoints(pivot_metric.index)
            lon_midpoints = get_bin_midpoints(pivot_metric.columns)
            
            
            # Set fixed colorbar range for NDVI MAE
            if channel_type == "NDVI" and metric_to_plot == "mae":
                heatmap_kwargs = {'vmin': 0.01, 'vmax': 0.2}
            else:
                heatmap_kwargs = {}

            sns.heatmap(
                pivot_metric, 
                ax=ax_geo_metric, 
                cmap=cmap_metric, # Use the modified colormap
                cbar_kws={'label': f'Mean {metric_to_plot.upper()}'},
                mask=pivot_metric.isnull(), # Explicitly mask NaN values
                **heatmap_kwargs
            )
            # ax_geo_metric.set_title(f"{selected_model} - {metric_to_plot.upper()} ({channel_type}) by Location")
            ax_geo_metric.set_xlabel("Longitude")
            ax_geo_metric.set_ylabel("Latitude")
            
            # Set ticks to show midpoints instead of bin intervals
            # sns.heatmap puts ticks at 0.5, 1.5, etc.
            step = max(1, len(lon_midpoints) // 10)
            ax_geo_metric.set_xticks(np.arange(len(lon_midpoints))[::step] + 0.5)
            ax_geo_metric.set_xticklabels([f"{x:.1f}" for x in lon_midpoints[::step]], rotation=45)
            
            step_y = max(1, len(lat_midpoints) // 10)
            ax_geo_metric.set_yticks(np.arange(len(lat_midpoints))[::step_y] + 0.5)
            ax_geo_metric.set_yticklabels([f"{y:.1f}" for y in lat_midpoints[::step_y]], rotation=0)
            
            save_fig_to_report(fig_geo_metric, f"geo_metric_{metric_to_plot}_{selected_model}_{channel_type.lower()}")
            st.pyplot(fig_geo_metric)
            
            # 2. Sample Count Heatmap
            st.subheader("Sample Count Distribution")
            # Use the first model's data as a proxy for location distribution since test sets match
            geo_df_global = full_df[full_df['dw_class'] == 'overall'].drop_duplicates(subset=['sample_idx'])
            
                
            # Re-drop duplicates after filtering because full_df has multiple rows per sample (one per channel)
            geo_df_global = geo_df_global.drop_duplicates(subset=['sample_idx'])

            geo_df_global['lat_bin'] = pd.cut(geo_df_global['lat'], bins=n_bins)
            geo_df_global['lon_bin'] = pd.cut(geo_df_global['lon'], bins=n_bins)
            
            pivot_count = geo_df_global.pivot_table(
                index='lat_bin', 
                columns='lon_bin', 
                values='sample_idx', # Count any column
                aggfunc='nunique' # Count unique samples
            )
            pivot_count = pivot_count.sort_index(ascending=False)
            
            fig_geo_count, ax_geo_count = get_styled_figure_ax(figsize=(12, 10), aspect='auto', grid=False)
            # Set facecolor to white for empty cells
            ax_geo_count.set_facecolor('white')
            
            # Compute midpoints for labels
            lat_midpoints_c = get_bin_midpoints(pivot_count.index)
            lon_midpoints_c = get_bin_midpoints(pivot_count.columns)
            
            mask = pivot_count.isnull() | (pivot_count == 0)
            sns.heatmap(
                pivot_count, 
                ax=ax_geo_count, 
                cmap=sns.color_palette("rocket_r", as_cmap=True),
                cbar_kws={'label': 'Number of Samples'},
                annot=False,
                fmt='g',     # General format
                mask=mask
            )
            # ax_geo_count.set_title(f"Global Sample Count by Location ({channel_type})")
            ax_geo_count.set_xlabel("Longitude")
            ax_geo_count.set_ylabel("Latitude")

            step = max(1, len(lon_midpoints_c) // 10)
            ax_geo_count.set_xticks(np.arange(len(lon_midpoints_c))[::step] + 0.5)
            ax_geo_count.set_xticklabels([f"{x:.1f}" for x in lon_midpoints_c[::step]], rotation=45)
            
            step_y = max(1, len(lat_midpoints_c) // 10)
            ax_geo_count.set_yticks(np.arange(len(lat_midpoints_c))[::step_y] + 0.5)
            ax_geo_count.set_yticklabels([f"{y:.1f}" for y in lat_midpoints_c[::step_y]], rotation=0)
            
            save_fig_to_report(fig_geo_count, f"geo_count_global_{channel_type.lower()}")
            st.pyplot(fig_geo_count)



if __name__ == "__main__":
    analysis_page()

import streamlit as st
import pandas as pd
import leafmap.foliumap as lm
import ipywidgets as widgets
import matplotlib.pyplot as plt
import os
import folium
import geopandas as gpd
import io

from src.utils.plot_utils import style_legend
DATASET_COLORS = ['#95BB63', '#BCBCE0', '#6a408d', '#EA805D']

def dataset_page():
    st.set_page_config(layout="wide")
    st.title("Dataset Geographical Distribution")

    # --- Sidebar for Configuration ---
    st.sidebar.header("Configuration")
    default_csv_path = "reports/eda_thresholded/dataset_processed_metrics.csv"
    csv_path = st.sidebar.text_input("Path to dataset metrics CSV", value=default_csv_path)

    if not os.path.exists(csv_path):
        st.error(f"File not found: {csv_path}")
        st.stop()

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()
    if 'meta_lat' not in df.columns or 'meta_lon' not in df.columns:
        st.error("CSV must contain 'meta_lat' and 'meta_lon' columns.")
        st.stop()
    df = df.rename(columns={"meta_lat": "latitude", "meta_lon": "longitude"})
    
    # --- Extract city name from filepath ---
    def extract_city(filepath):
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        return " ".join(parts[:-8])

    df['city'] = df['filepath'].apply(extract_city)
    
    st.write(f"Loaded {len(df)} samples from `{os.path.basename(csv_path)}`")
    st.write(f"Total unique cities in dataset: {df['city'].nunique()}")

    # --- Preprocess Split Info ---
    # Get all splits per city
    city_splits = df.groupby('city')['split'].apply(set).to_dict()

    # Identify exclusive cities
    exclusive_val_cities = {c for c, splits in city_splits.items() if splits == {'val'}}
    exclusive_test_cities = {c for c, splits in city_splits.items() if splits == {'test'}}
    exclusive_train_cities = {c for c, splits in city_splits.items() if splits == {'train'}}

    # --- Tabular Info ---
    st.subheader("Dataset Statistics")
    
    splits = ["train", "val", "test"]
    stats_data = []
    
    for split in splits:
        subset = df[df['split'] == split]
        n_samples = len(subset)
        n_cities = subset['city'].nunique()
        
        n_exclusive_cities = 0
        n_exclusive_samples = 0
        
        if split == "train":
            target_cities = exclusive_train_cities
        elif split == "val":
            target_cities = exclusive_val_cities
        elif split == "test":
            target_cities = exclusive_test_cities
        else:
            target_cities = set()
            
        n_exclusive_cities = len(target_cities)
        n_exclusive_samples = subset[subset['city'].isin(target_cities)].shape[0]

        stats_data.append({
            "Split": split.capitalize(),
            "Total Samples": n_samples,
            "Unique Cities": n_cities,
            "Exclusive Cities": n_exclusive_cities,
            "Exclusive Samples": n_exclusive_samples
        })
    
    st.table(pd.DataFrame(stats_data))

    col1, col2 = st.columns(2)
    with col1:
        val_cities = df[df['split'] == 'val'].drop_duplicates('city')[['city', 'latitude', 'longitude']].reset_index(drop=True)
        with st.expander(f"All Validation Cities ({len(val_cities)})"):
            st.dataframe(val_cities)
        
        # Exclusive Val
        ex_val_df = df[df['city'].isin(exclusive_val_cities)].drop_duplicates('city')[['city', 'latitude', 'longitude']].reset_index(drop=True)
        with st.expander(f"Exclusive Validation Cities ({len(ex_val_df)})"):
            st.dataframe(ex_val_df)

    with col2:
        test_cities = df[df['split'] == 'test'].drop_duplicates('city')[['city', 'latitude', 'longitude']].reset_index(drop=True)
        with st.expander(f"All Test Cities ({len(test_cities)})"):
            st.dataframe(test_cities)

        # Exclusive Test
        ex_test_df = df[df['city'].isin(exclusive_test_cities)].drop_duplicates('city')[['city', 'latitude', 'longitude']].reset_index(drop=True)
        with st.expander(f"Exclusive Test Cities ({len(ex_test_df)})"):
            st.dataframe(ex_test_df)

    # --- Preprocess for Maps ---
    # Define split priority
    split_priority = {"train": 3, "val": 2, "test": 1}
    
    # Determine dominant split using priority
    def dominant_split(splits):
        if not splits:
            return "unknown"
        # pick the split with the highest priority
        return max(splits, key=lambda s: split_priority.get(s, 0))

    df['dominant_split'] = df['city'].map(lambda c: dominant_split(city_splits[c]))


    # --- Keep unique city points ---
    df_unique = df.drop_duplicates(subset='city').reset_index(drop=True)

    # --- Color map by split ---
    color_map = {
        "train": DATASET_COLORS[0],
        "test":   DATASET_COLORS[2],
        "val":  DATASET_COLORS[3],
    }
    mpl_color_map = {
        "train": DATASET_COLORS[0],
        "test": DATASET_COLORS[2],
        "val": DATASET_COLORS[3],
    }

    # --- Interactive Map ---
    st.subheader("Geographical Distribution (Interactive)")
    if st.button("Generate Interactive Map"):
        m = lm.Map(center=(20, 0), zoom=2)
        m.add_basemap("HYBRID")
        m.add_basemap("Esri.NatGeoWorldMap")

        # Add colored points
        for _, row in df_unique.iterrows():
            city = row['city']
            # Use the 'dominant_split' column, which reflects the highest priority split
            d_split = row.get('dominant_split', 'unknown') 

            # Default popup text
            popup_text = f"{city} ({d_split})"

            # Override color and popup if it's an exclusive test city
            if city in exclusive_test_cities:
                color = color_map['test']
                popup_text = f"{city} (test)"
            else:
                color = color_map.get(d_split, "#999999")

            folium.CircleMarker(
                location=(row["latitude"], row["longitude"]),
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=popup_text
            ).add_to(m)
            
        # Add legend
        m.add_legend(
            title="Dataset Split",
            labels=["Train", "Validation", "Test"],
            colors=[
                color_map["train"],
                color_map["val"],
                color_map["test"]
            ]
        )

        # Render in Streamlit
        m.to_streamlit(height=600)

    # --- PDF Generation ---
    st.subheader("Static Map (PDF Export)")
    st.write("Generates a high-quality PDF map with continent contours suitable for research papers.")
    
    if st.button("Generate Static PDF Map"):
        with st.spinner("Loading world map and generating PDF..."):
            try:
                # Attempt to load a world map
                # Using a reliable URL for GeoJSON countries
                world_url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json"
                world = gpd.read_file(world_url)
                
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # Plot the world map
                world.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.5)
                
                # Determine plotting category for each unique city
                import matplotlib.patches as mpatches
                
                pie_radius = 1.5

                for _, row in df_unique.iterrows():
                    city = row['city']
                    lat, lon = row['latitude'], row['longitude']
                    
                    # Get splits for this city
                    splits = list(city_splits.get(city, []))
                    
                    # Sort to ensure consistent orientation (Train -> Val -> Test)
                    sort_order = {"train": 0, "val": 1, "test": 2}
                    splits.sort(key=lambda s: sort_order.get(s, 99))
                    
                    if not splits:
                        continue
                    
                    n_splits = len(splits)
                    angle_per_split = 360 / n_splits
                    start_angle = 90 # Start from top
                    
                    for split in splits:
                        color = mpl_color_map.get(split, 'gray')
                        
                        # Draw wedge
                        wedge = mpatches.Wedge(
                            (lon, lat), 
                            pie_radius, 
                            start_angle, 
                            start_angle + angle_per_split, 
                            facecolor=color, 
                            edgecolor='white',
                            linewidth=0.1,
                            zorder=5
                        )
                        ax.add_patch(wedge)
                        start_angle += angle_per_split

                # Custom Legend
                legend_handles = [
                    mpatches.Patch(color=mpl_color_map["train"], label="Train"),
                    mpatches.Patch(color=mpl_color_map["val"], label="Validation"),
                    mpatches.Patch(color=mpl_color_map["test"], label="Test"),
                ]
                legend_labels = [h.get_label() for h in legend_handles]
                
                style_legend(ax, ncol=3, handles=legend_handles, 
                             labels=legend_labels, 
                             frameon=False,
                             bbox_to_anchor=(0.5, 1.02)
                             )
                
                # ax.set_title("Geographical Distribution of Dataset Cities")
                ax.set_axis_off()

                # Save to BytesIO buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='pdf', bbox_inches='tight', dpi=300)
                buf.seek(0)
                
                st.download_button(
                    label="Download PDF",
                    data=buf,
                    file_name="dataset_geographical_distribution.pdf",
                    mime="application/pdf"
                )
                st.success("PDF generated successfully.")
                
            except Exception as e:
                st.error(f"Failed to generate PDF map: {e}")
                st.write("Ensure you have internet access to fetch the world map base layer, or that `geopandas` is correctly installed.")

if __name__ == "__main__":
    dataset_page()

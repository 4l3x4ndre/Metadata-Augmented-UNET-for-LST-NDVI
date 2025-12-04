import streamlit as st
import os
import ee
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import torch

import streamlit.elements.image as st_image
try:
    from streamlit.elements.lib.image_utils import image_to_url as original_image_to_url
    
    # Fix for streamlit-drawable-canvas with newer Streamlit versions
    class ImageConfig:
        def __init__(self, width):
            self.width = width

    def patched_image_to_url(image, width, clamp, channels, output_format, image_id, allow_emoji=False):
        # st_canvas passes int for width, which causes error in new streamlit
        if isinstance(width, int):
            width = ImageConfig(width)
        # Drop allow_emoji as it's removed in newer Streamlit versions
        return original_image_to_url(image, width, clamp, channels, output_format, image_id)
        
    st_image.image_to_url = patched_image_to_url
except ImportError:
    pass # specific for newer streamlit versions

from streamlit_drawable_canvas import st_canvas

from gee_utils import get_satellite_data
from processing_utils import DW_PALETTE, DW_PALETTE_INV, prepare_input, denormalize_output, canvas_to_dw_map
from model_utils import load_model, run_inference

st.set_page_config(page_title="Urban Greening Planner", layout="wide")

# --- Session State ---
if 'paths' not in st.session_state:
    st.session_state['paths'] = None
if 'image_key' not in st.session_state:
    st.session_state['image_key'] = 0

st.title("🏙️ Urban Greening Planner")
st.markdown("Predict the microclimatic impact (LST & NDVI) of your urban scenarios.")

st.info("👋 Welcome! Please visit the 'Readme' page for detailed information about this application, the data used, and important considerations about data quality.")

from dotenv import load_dotenv
load_dotenv()

default_project = os.getenv("GEE_PROJECT_ID", "")
default_service_account = os.getenv("GEE_SERVICE_ACCOUNT", "")

expand_credentials = not default_project or not default_service_account

with st.expander("Google Earth Engine Credentials", expanded=expand_credentials):
    if not default_project or not default_service_account:
        st.warning("Credentials not found in .env file. Please enter them below. Also create a .private-key.json file for your [service account](https://developers.google.com/earth-engine/guides/service_account).")

    project_id = st.text_input("GEE Project ID", value=default_project)
    service_account = st.text_input("GEE Service Account", value=default_service_account)

    # Update environment variables so gee_utils uses the user input
    os.environ["GEE_PROJECT_ID"] = project_id
    os.environ["GEE_SERVICE_ACCOUNT"] = service_account

# Check if credentials are available
credentials_present = bool(os.getenv("GEE_PROJECT_ID")) and bool(os.getenv("GEE_SERVICE_ACCOUNT"))

# Defaults
default_lat = 41.8990
default_lon = 12.4690
default_year_t1 = 2019
default_month_t1 = 8

# Cache-Only Logic
if not credentials_present:
    st.warning("⚠️ No credentials provided. Switching to **Cache-Only Mode**. You can only use the pre-loaded location/data.")
    
    # Try to find cached files
    cache_pattern = os.path.join("app/cache", "*_dw.tif")
    cache_files = glob.glob(cache_pattern)
    
    if cache_files:
        # Pick the first one
        first_file = cache_files[0]
        basename = os.path.basename(first_file).replace("_dw.tif", "")
        # Filename format: lat_lon_year_month (e.g., 41.8990_12.4690_2019_08)
        try:
            parts = basename.split('_')
            
            c_year = int(parts[-2])
            c_month = int(parts[-1])
            c_lon = float(parts[-3])
            c_lat = float(parts[-4])
            
            default_lat = c_lat
            default_lon = c_lon
            default_year_t1 = c_year
            default_month_t1 = c_month
            
            # Populate session state paths if empty
            if st.session_state['paths'] is None:
                st.session_state['paths'] = {
                    'dw': os.path.join("app/cache", f"{basename}_dw.tif"),
                    'rgb': os.path.join("app/cache", f"{basename}_rgb.tif"),
                    'ndvi': os.path.join("app/cache", f"{basename}_ndvi.tif"),
                    'temp': os.path.join("app/cache", f"{basename}_temp.tif"),
                }
                st.session_state['image_key'] += 1
                
            # Look up population for cached city
            try:
                cities_path = "data/processed/cities/worldcities_processed.csv"
                if os.path.exists(cities_path):
                    cities_df = pd.read_csv(cities_path)
                    cities_df['dist'] = np.sqrt((cities_df['lat'] - default_lat)**2 + (cities_df['lng'] - default_lon)**2)
                    nearest_city = cities_df.loc[cities_df['dist'].idxmin()]
                    st.session_state['fetched_population'] = float(nearest_city['population'])
            except Exception as e:
                pass
                
        except Exception as e:
            st.error(f"Error parsing cached file: {e}")

# --- Sidebar ---
st.sidebar.header("1. Configure Inputs")
st.sidebar.markdown("""
Adjust the **location** (Latitude, Longitude), **current time** (Year T1, Month T1), and **future target time** (Year T2, Month T2).
""")

# Location
lat = st.sidebar.number_input("Latitude", value=default_lat, format="%.4f", disabled=not credentials_present)
lon = st.sidebar.number_input("Longitude", value=default_lon, format="%.4f", disabled=not credentials_present)

# Population is now automatic based on nearest city
if 'fetched_population' not in st.session_state:
    st.session_state['fetched_population'] = 2000000 # Default fallback

# Time
col1, col2 = st.sidebar.columns(2)
year_t1 = col1.number_input("Year (Current)", value=default_year_t1, min_value=2015, max_value=2025, disabled=not credentials_present)
month_t1 = col2.number_input("Month (Current)", value=default_month_t1, min_value=1, max_value=12, disabled=not credentials_present)

year_t2 = st.sidebar.number_input("Target Year (Future)", value=2025, min_value=2023, max_value=2030)
month_t2 = st.sidebar.number_input("Target Month", value=8, min_value=1, max_value=12)

st.sidebar.header("2. Fetch Current Data")
st.sidebar.markdown("""
Click the **"Fetch Satellite Data"** button on the main page to retrieve current (T1) satellite imagery and determine the local population.
""")

st.sidebar.header("3. Design Future Scenario")
st.sidebar.markdown("""
On the main page, use the **canvas** to draw your proposed land use changes for the future (T2). Select brush colors from the palette.
""")

st.sidebar.header("4. Select Model & Predict")
st.sidebar.markdown("""
Choose a **trained model** from the dropdown menu, then click **"Run Prediction"** on the main page to forecast the impact of your changes.
""")

# Model Picker
models_dir = "models"
model_files = sorted(glob.glob(os.path.join(models_dir, "*.pth")))
if not model_files:
    st.sidebar.warning("No models found in 'models/'. Please add a .pth file.")
    selected_model_path = None
else:
    selected_model_path = st.sidebar.selectbox("Select Model", model_files, index=0)


# --- Session State ---
if 'paths' not in st.session_state:
    st.session_state['paths'] = None
if 'image_key' not in st.session_state:
    st.session_state['image_key'] = 0


# --- Step 1: Fetch Data ---
st.header("1. Retrieve Current State")
if st.button("Fetch Satellite Data", disabled=not credentials_present):
    # Clear cache before fetching
    cache_dir = "app/cache"
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    # Look up nearest city and population
    try:
        cities_path = "data/processed/cities/worldcities_processed.csv"
        if os.path.exists(cities_path):
            cities_df = pd.read_csv(cities_path)
            # Euclidean distance for finding nearest (sufficient for this scale)
            cities_df['dist'] = np.sqrt((cities_df['lat'] - lat)**2 + (cities_df['lng'] - lon)**2)
            nearest_city = cities_df.loc[cities_df['dist'].idxmin()]
            
            st.session_state['fetched_population'] = float(nearest_city['population'])
            st.info(f"Nearest City: {nearest_city['city']} (Lat: {nearest_city['lat']:.2f}, Lon: {nearest_city['lng']:.2f}) - Population: {int(nearest_city['population']):,}")
        else:
            st.warning(f"Cities database not found at {cities_path}. Using default population.")
    except Exception as e:
        st.error(f"Error looking up city data: {e}")

    with st.spinner("Fetching data from Earth Engine..."):
        try:
            paths, error = get_satellite_data(lat, lon, year_t1, month_t1)
            if error:
                st.error(error)
            else:
                st.session_state['paths'] = paths
                st.session_state['image_key'] += 1 # Force canvas reload
                st.success("Data fetched successfully!")
        except ee.EEException as e:
            st.error(f"Earth Engine Error: {e}.\n\nPlease check your credentials in the 'Google Earth Engine Credentials' section above. Go to [Service Account page](https://developers.google.com/earth-engine/guides/service_account) to create your private key.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

import rasterio

# Helper to safely load and colorize/normalize images for display
def load_for_display(path, cmap=None, vmin=None, vmax=None):
    with rasterio.open(path) as src:
        img = src.read(1)  # Read first band
        
        # If it's floating point (likely NDVI or Temp), normalize or apply cmap
        if cmap:
            # Use matplotlib to apply colormap
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap_func = plt.get_cmap(cmap)
            mapped = cmap_func(norm(img)) # Returns RGBA float 0-1
            return (mapped[:, :, :3] * 255).astype(np.uint8) # RGB uint8
        
        if src.count == 3:
            return np.moveaxis(src.read(), 0, -1) # (H, W, 3)
            
        return img

if st.session_state['paths']:
    paths = st.session_state['paths']
    
    # Layout: Display T1 Data
    c1, c2, c3, c4 = st.columns(4)
    
    # RGB
    # Check if RGB exists and is valid
    if os.path.exists(paths['rgb']):
        c1.image(paths['rgb'], caption="RGB Imagery", width='stretch')
    
    # DW (Colorize)
    dw_img = Image.open(paths['dw']) 
    dw_arr = np.array(dw_img)
    h, w = dw_arr.shape
    colorized_dw = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color_hex in DW_PALETTE_INV.items():
        rgb = [int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
        colorized_dw[dw_arr == cls] = rgb
    c2.image(colorized_dw, caption="Land Use (Dynamic World)", width='stretch')

    # NDVI (Float) -> Colormap
    if os.path.exists(paths['ndvi']):
        ndvi_disp = load_for_display(paths['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
        c3.image(ndvi_disp, caption="NDVI", width='stretch')

    # Temp (Float) -> Colormap
    if os.path.exists(paths['temp']):
        with rasterio.open(paths['temp']) as src:
            t_data = src.read(1)
            t_min, t_max = np.nanmin(t_data), np.nanmax(t_data)
        temp_disp = load_for_display(paths['temp'], cmap='magma', vmin=t_min, vmax=t_max)
        c4.image(temp_disp, caption="Land Surface Temp", width='stretch')
    
    # --- Step 2: Edit Scenario ---
    st.header("2. Design Future Scenario")
    st.markdown("Modify the Land Use map below to simulate changes (e.g., plant trees, build structures).")
    
    raw_dw_img = Image.open(paths['dw'])
    dw_arr = np.array(raw_dw_img)

    # The DW export from GEE is grayscale (0-8). We need to colorize it for the canvas to look good.
    if dw_arr.ndim == 2:
        # Colorize
        h, w = dw_arr.shape
        colorized = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color_hex in DW_PALETTE_INV.items():
            rgb = [int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
            colorized[dw_arr == cls] = rgb
        bg_image = Image.fromarray(colorized).convert("RGBA")
    else:
        bg_image = raw_dw_img.convert("RGBA")

    
    # Palette for picking
    st.subheader("Brush Palette")
    cols = st.columns(9)
    
    # Create a state for selected color
    if 'stroke_color' not in st.session_state:
        st.session_state['stroke_color'] = "#397d49" # Trees default

    labels = ["Water", "Trees", "Grass", "Flood", "Crops", "Shrub", "Built", "Bare", "Snow"]
    
    for i, (hex_code, cls_id) in enumerate(DW_PALETTE.items()):
        with cols[i]:
            if st.button(f"{labels[cls_id]}", key=f"btn_{cls_id}"):
                st.session_state['stroke_color'] = hex_code
            st.markdown(f"<div style='width:20px;height:20px;background-color:{hex_code};border:1px solid black'></div>", unsafe_allow_html=True)

    stroke_width = st.slider("Brush Size", 1, 50, 10)
    
    canvas_result = st_canvas(
        fill_color=st.session_state['stroke_color'],
        stroke_width=stroke_width,
        stroke_color=st.session_state['stroke_color'],
        background_image=bg_image,
        update_streamlit=True,
        height=512, # Fixed for now, or match image aspect
        width=512,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state['image_key']}",
    )
    
    # --- Step 3: Predict ---
    st.header("3. Forecast Impact")
    
    if st.button("Run Prediction"):
        if canvas_result.image_data is not None and selected_model_path:
            with st.spinner("Running inference..."):
                # 1. Prepare Input
                input_tensor, meta_tensor, temp_tensor = prepare_input(
                    paths, 
                    canvas_result.image_data, 
                    lat, lon, st.session_state['fetched_population'], 
                    year_t1, month_t1, year_t2, month_t2
                )
                
                # Visualize Model Input DW
                # Extract DW_t2 from input_tensor (last 9 channels)
                # Shape: (1, 23, 512, 512)
                # DW_t1 (9) + RGB (3) + NDVI (1) + Temp (1) + DW_t2 (9)
                # Index start = 9+3+1+1 = 14
                dw_t2_ohe = input_tensor[0, 14:23, :, :].numpy()
                dw_t2_idx = np.argmax(dw_t2_ohe, axis=0)
                
                h_vis, w_vis = dw_t2_idx.shape
                dw_t2_vis = np.zeros((h_vis, w_vis, 3), dtype=np.uint8)
                for cls, color_hex in DW_PALETTE_INV.items():
                    rgb = [int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                    dw_t2_vis[dw_t2_idx == cls] = rgb
                
                st.subheader("Model Input Check")
                st.image(dw_t2_vis, caption="Future Land Use (Model Input)", width=300)

                # 2. Load Model
                device = 'cpu'
                model = load_model(selected_model_path, device=device)
                
                # 3. Predict
                output = run_inference(model, input_tensor, meta_tensor, temp_tensor, device=device)
                
                # 4. Display
                ndvi_pred = output[0, 0, :, :]
                temp_pred = output[0, 1, :, :]
                
                ndvi_denorm, temp_denorm = denormalize_output(ndvi_pred, temp_pred)
                
                # Calculate Stats
                # Load original for delta
                with rasterio.open(paths['temp']) as src:
                    temp_orig = src.read(1)
                
                # Display Results
                res_col1, res_col2, res_col3 = st.columns(3)
                
                # Plotting
                fig_ndvi, ax_ndvi = plt.subplots()
                im = ax_ndvi.imshow(ndvi_denorm, cmap='RdYlGn', vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax_ndvi)
                ax_ndvi.set_title("Predicted NDVI")
                ax_ndvi.axis('off')
                res_col1.pyplot(fig_ndvi)
                
                fig_temp, ax_temp = plt.subplots()
                im = ax_temp.imshow(temp_denorm, cmap='magma')
                plt.colorbar(im, ax=ax_temp, label="°C")
                ax_temp.set_title("Predicted LST (°C)")
                ax_temp.axis('off')
                res_col2.pyplot(fig_temp)

                # Delta LST
                # Ensure shapes match
                if temp_denorm.shape == temp_orig.shape:
                    delta_temp = temp_denorm - temp_orig
                    fig_delta, ax_delta = plt.subplots()
                    # Diverging colormap centered on 0
                    max_delta = max(abs(np.min(delta_temp)), abs(np.max(delta_temp)))
                    im = ax_delta.imshow(delta_temp, cmap='coolwarm', vmin=-max_delta, vmax=max_delta)
                    plt.colorbar(im, ax=ax_delta, label="Δ°C")
                    ax_delta.set_title("Temperature Change")
                    ax_delta.axis('off')
                    res_col3.pyplot(fig_delta)
                    
                    avg_cooling = np.mean(delta_temp)
                    st.metric("Average Temperature Change", f"{avg_cooling:.2f} °C")
                else:
                    st.warning(f"Original and predicted shapes mismatch, cannot compute delta. Original: {temp_orig.shape}, Predicted: {temp_denorm.shape}")

        else:
            st.warning("Please draw on the canvas and select a model.")

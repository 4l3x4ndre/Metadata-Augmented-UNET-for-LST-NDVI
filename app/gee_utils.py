import ee
import geemap
import os
import logging
import rasterio
from rasterio.enums import Resampling
from dotenv import load_dotenv
from urban_planner.config import CONFIG

load_dotenv()

logger = logging.getLogger(__name__)

def authenticate():
    try:
        ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))
        logger.info("Earth Engine already authenticated and initialized.")
    except Exception as e:
        logger.info(f"Authentication needed or project not set. Attempting authentication flow... {e}")
        try:
            service_account = os.getenv('GEE_SERVICE_ACCOUNT')
            credentials = ee.ServiceAccountCredentials(service_account, '.private-key.json')
            ee.Initialize(credentials)
        except Exception as inner_e:
            logger.error(f"Failed to authenticate: {inner_e}")
            # Fallback to standard flow if service account fails or not provided
            ee.Authenticate()
            ee.Initialize()

def maskL8sr(image):
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 5).eq(0))
    return image.updateMask(mask)

def apply_scale_landsat(image):
    optical = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    thermal = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)

def resize_and_overwrite(path, target_size):
    """
    Reads the image at path, resizes it to (target_size, target_size),
    updates the transform to maintain the geographic extent, and overwrites the file.
    """
    try:
        with rasterio.open(path) as src:
            # Determine resampling method
            # Nearest for DW (categorical), Bilinear for others (continuous)
            resampling = Resampling.nearest if '_dw.tif' in path else Resampling.bilinear
            
            data = src.read(
                out_shape=(src.count, target_size, target_size),
                resampling=resampling
            )
            
            profile = src.profile.copy()
            
            # Update transform to account for new resolution
            new_transform = src.transform * src.transform.scale(
                (src.width / target_size),
                (src.height / target_size)
            )
            
            profile.update({
                'height': target_size,
                'width': target_size,
                'transform': new_transform,
            })
            
        # Write back
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data)
            
        logger.info(f"Resized {path} to {target_size}x{target_size}")
        
    except Exception as e:
        logger.error(f"Failed to resize {path}: {e}")
        raise e

def process_saved_images(paths):
    """
    Iterates through saved paths and enforces the model's image size.
    """
    target_size = CONFIG.model.img_size
    for key, path in paths.items():
        if os.path.exists(path):
            resize_and_overwrite(path, target_size)

def get_satellite_data(lat, lon, year, month, output_dir="app/cache"):
    """
    Fetches Sentinel-2 (RGB, NDVI), Dynamic World (LULC), and Landsat 8 (LST)
    for a given location and time.
    Uses a 1000m buffer logic (2km x 2km box) and exports at native/specified scales.
    Then resizes to CONFIG.model.img_size.
    """
    authenticate()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1000m buffer logic from gee_functions_future.py
    meters = 1000
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(meters).bounds()
    
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(60, 'days')
    
    basename = f"{lat:.4f}_{lon:.4f}_{year}_{month:02d}"
    
    # --- Dynamic World ---
    dw_col = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(region).filterDate(start, end)
    
    # --- Sentinel-2 ---
    s2_col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(region).filterDate(start, end)
    
    # --- Landsat 8 ---
    l8_col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(region).filterDate(start, end).map(apply_scale_landsat).map(maskL8sr)
    
    # Check availability
    if dw_col.size().getInfo() == 0:
        return None, "No Dynamic World data found for this date/location."
    if s2_col.size().getInfo() == 0:
        return None, "No Sentinel-2 data found for this date/location."
    if l8_col.size().getInfo() == 0:
        return None, "No Landsat 8 data found for this date/location."

    # Mosaic/Mean/Mode
    # Use mode for DW to keep integer classes
    dw_img = dw_col.mode().clip(region) 
    s2_img = s2_col.median().clip(region)
    l8_img = l8_col.mean().clip(region)

    paths = {}
    
    # Export DW (Label)
    dw_path = os.path.join(output_dir, f"{basename}_dw.tif")
    try:
        geemap.ee_export_image(
            dw_img.select('label'),
            filename=dw_path,
            scale=10, # 10m scale
            region=region,
            file_per_band=False
        )
        paths['dw'] = dw_path
    except Exception as e:
        return None, f"Failed to export DW: {e}"

    # Export RGB
    rgb_path = os.path.join(output_dir, f"{basename}_rgb.tif")
    try:
        s2_rgb = s2_img.select(['B4', 'B3', 'B2']).visualize(min=0, max=3000)
        geemap.ee_export_image(
            s2_rgb,
            filename=rgb_path,
            scale=10, # 10m scale
            region=region,
            file_per_band=False
        )
        paths['rgb'] = rgb_path
    except Exception as e:
        return None, f"Failed to export RGB: {e}"

    # Export NDVI
    ndvi_path = os.path.join(output_dir, f"{basename}_ndvi.tif")
    try:
        ndvi = s2_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        geemap.ee_export_image(
            ndvi,
            filename=ndvi_path,
            scale=10, # 10m scale
            region=region,
            file_per_band=False
        )
        paths['ndvi'] = ndvi_path
    except Exception as e:
        return None, f"Failed to export NDVI: {e}"

    # Export Temperature
    temp_path = os.path.join(output_dir, f"{basename}_temp.tif")
    try:
        st_kelvin = l8_img.select('ST_B10')
        st_celsius = st_kelvin.subtract(273.15)
        geemap.ee_export_image(
            st_celsius,
            filename=temp_path,
            scale=30, # 30m scale for L8
            region=region,
            file_per_band=False
        )
        paths['temp'] = temp_path
    except Exception as e:
        return None, f"Failed to export Temperature: {e}"
    
    # Resize images to match model input size
    try:
        process_saved_images(paths)
    except Exception as e:
        return None, f"Failed to resize images: {e}"
        
    return paths, None

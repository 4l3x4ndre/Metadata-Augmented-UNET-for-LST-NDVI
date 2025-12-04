import numpy as np
from PIL import Image
import matplotlib.patches as mpatches

DW_CLASSES = {
    0: "water", 1: "trees", 2: "grass", 3: "flooded_vegetation",
    4: "crops", 5: "shrub_and_scrub", 6: "built", 7: "bare", 8: "snow_and_ice"
}

HEX_COLORS = [ 
    '#419bdf', # 0 water
    '#547551', # 1 trees
    '#88b053', # 2 grass
    '#153d1a', # 3 flooded_vegetation
    '#e49635', # 4 crops
    '#517075', # 5 shrub_and_scrub
    '#616161', # 6 built
    '#4a3b25', # 7 bare
    '#fcfcfc'  # 8 snow_and_ice
]

def dw_to_rgb(dw_array, return_numpy=False):
    """
    Converts a Dynamic World array to an RGB image.

    Args:
        dw_array (np.ndarray): (H, W) array with values in [0, 8] (integer).
        return_numpy (bool): If True, returns a numpy array. Otherwise, returns a PIL Image.

    Returns:
        Union[Image.Image, np.ndarray]: RGB image.
    """
    rgb_colors = [tuple(int(h[i:i+2], 16) for i in (1, 3, 5)) for h in HEX_COLORS]
    h, w = dw_array.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(rgb_colors):
        rgb_image[dw_array == i] = color
    
    if return_numpy:
        return rgb_image
    
    return Image.fromarray(rgb_image, 'RGB')

def get_dw_legend_patches():
    """
    Returns a list of patches for a matplotlib legend for Dynamic World classes.
    """
    return [mpatches.Patch(color=HEX_COLORS[i], label=f"{i}: {DW_CLASSES[i]}") for i in range(len(DW_CLASSES))]
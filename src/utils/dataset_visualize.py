import os
import sys
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from cv2 import resize as cv2_resize, INTER_LINEAR

from urban_planner.config import CONFIG

def show_images_with_colorbar(root_folder, image_prefix):
    image_files = sorted([f for f in os.listdir(root_folder) if f.startswith(image_prefix)])
    n_images = len(image_files)
    n_cols = 4
    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 5*n_rows))
    axes = axes.flatten()

    for idx, (ax, img_file) in enumerate(zip(axes, image_files)):
        img_path = os.path.join(root_folder, img_file)
        # img = Image.open(img_path)

        with rasterio.open(img_path) as src:
            img_raw = src.read()
            img_raw = resize_channels(img_raw)
            if img_raw.shape[0] == 1:
                img = img_raw[0]
            else:
                img = np.transpose(img_raw, (1, 2, 0))
                if img.shape[2] >= 3:
                    img = img[:, :, :3]
                else:
                    raise ValueError("Unsupported number of channels: {}".format(img.shape[2]))

        img_arr = np.array(img)

        print("Image:", img_file, "min:", img_arr.min(), "max:", img_arr.max(), "mean:", img_arr.mean(), "std:", img_arr.std(), "nans:", np.isnan(img_arr).sum())
        if img_arr.max() > 1:
            img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())

        im = ax.imshow(img_arr)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(img_file)
        ax.axis('off')

    # Hide unused axes
    for ax in axes[n_images:]:
        ax.axis('off')

    # plt.tight_layout()
    plt.show()


def resize_image(img):
    img_max = img.max()
    img = img.astype(np.float32)
    target = CONFIG.dataset.image_shape_edge
    if img.shape[1:] == (target, target):
        return img
    interp = INTER_LINEAR
    if img.shape[0] == 1:
        img_r = cv2_resize(img.squeeze(), (target, target), interpolation=interp)[np.newaxis, ...]
        return img_r.astype(np.uint32) * img_max 
    img_r = cv2_resize(img.transpose(1, 2, 0), (target, target), interpolation=interp).transpose(2, 0, 1)
    return img_r.astype(np.uint32) * img_max

def resize_channels(image):
    # convert to float64 first (as using INTER_LINEAR):
    image = image.astype(np.float64)
    new_shape = (CONFIG.dataset.image_shape_edge, CONFIG.dataset.image_shape_edge)
    resized = cv2_resize(image.transpose(1, 2, 0), new_shape, interpolation=INTER_LINEAR)
    if len(resized.shape) == 2:
        resized = resized[:, :, np.newaxis]
    resized_channels = resized.transpose(2, 0, 1)
    return resized_channels



if __name__ == "__main__":
    # python dataset_visualize.py /path/to/images image_prefix 
    if len(sys.argv) != 3:
        print("Usage: python dataset_visualize.py <root_folder> <image prefix>")
        sys.exit(1)
    root = sys.argv[1]
    file_prefix = sys.argv[2]
    show_images_with_colorbar(root, file_prefix)

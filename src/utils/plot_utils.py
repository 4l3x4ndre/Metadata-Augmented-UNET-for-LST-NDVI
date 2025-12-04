"""emperature
This script is adapted from:
BeautifulFigures, Andrey Churkin, https://github.com/AndreyChurkin/BeautifulFigures
"""

import matplotlib.pyplot as plt
import re

# https://www.color-hex.com/color-palette/106106
# Defining colours for datasets
DATASET_COLORS = ['#95BB63', '#BCBCE0', '#77b5b6', '#EA805D']
DATASET_COLORS_5 = ['#95BB63', '#BCBCE0', '#77b5b6', '#EA805D', '#795A5A']
DATASET_COLORS_5V5 = [
    "#A9D8A1", "#94CFAE", "#7FC7BB", "#6ABEC8", "#57B4D2",   # best 5 (green/teal)
    "#F6C5C3", "#F4B1A8", "#F09E92", "#EB8B7C", "#E57869"    # worst 5 (rose/coral)]
]
DATASET_LINE_COLORS = ['#95BB63', '#6a408d', '#77b5b6', '#EA805D']
# DATASET_LINE_COLORS = ['#6a408d', '#4e4e4e', '#378d94', '#c04a2e']
REGRESSION_COLOR = '#8a8a8a'

def set_plot_style():
    """Sets a consistent style for matplotlib plots."""
    plt.rcParams.update({
        'font.family': 'monospace',
        # 'font.family': 'Courier New',
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20,

        # 'svg.fonttype': 'path'

        # PDF-specific settings
        'pdf.fonttype': 42,              # Embed fonts as TrueType (keeps text selectable)
        'ps.fonttype': 42,               # For PostScript 
    })

def get_styled_figure_ax(figsize=(10, 10), aspect='equal', datalim=True, grid=True):
    """
    Creates a matplotlib figure and axes with a consistent style.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    if aspect == 'equal':
        ax.set_aspect('equal', adjustable='datalim' if datalim else 'box')

    if grid:
        # Major grid
        ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25)

        # Minor ticks and grid
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.15)

        ax.set_axisbelow(True)
    return fig, ax

def convert_label(label):
    """Converts internal label names to more readable forms."""
    label = label.replace('after_ndvi', 'NDVI (T2)').replace('after ndvi', 'NDVI (T2)')
    label = label.replace('after_temp', 'Temperature (T2)').replace('after temp', 'Temperature (T2)')

    # Specific embedding type conversions
    label = label.replace('noemb', 'No embeddings')
    label = label.replace('metaemb', 'Metadata')
    label = label.replace('tempemb', 'Temperature')
    # label = label.replace('noemb', 'N')
    # label = label.replace('metaemb', 'M')
    # label = label.replace('tempemb', 'T')
    
    # Generic embedding replacement (whole word only to avoid replacing inside 'Embeddings')
    label = re.sub(r'\bemb\b', 'Embeddings', label)
    # label = re.sub(r'\bemb\b', 'E', label)

    if 'rmse' in label.lower():
        label = label.replace('-', '.')
    
    label = label.replace('_', ' ')
    return label

def style_legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False, order=None, format_labels=None, handles=None, labels=None):
    """Styles the legend of a plot."""
    if handles is None or labels is None:
        detected_handles, detected_labels = ax.get_legend_handles_labels()
        if handles is None:
            handles = detected_handles
        if labels is None:
            labels = detected_labels

    if handles and labels:
        if order:
            handles = [handles[i] for i in order]
            labels = []
            for i in order:
                if format_labels:
                    labels.append(format_labels(labels[i]))
                else:
                    labels.append(convert_label(labels[i]))
        else:
            if format_labels:
                labels = [format_labels(label) for label in labels]
            else:
                labels = [convert_label(label) for label in labels]
        ax.legend(
            handles,
            labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            frameon=frameon
        )

def adjust_plot_limits(ax, all_data_x, all_data_y, zoom_out=0.6):
    """Adjusts plot limits to be equal and centered around the data."""
    if not hasattr(all_data_x, '__len__') or len(all_data_x) == 0 or not hasattr(all_data_y, '__len__') or len(all_data_y) == 0:
        return

    x_min = min(all_data_x)
    x_max = max(all_data_x)
    x_median = (x_min + x_max) / 2
    x_range = x_max - x_min

    y_min = min(all_data_y)
    y_max = max(all_data_y)
    y_median = (y_min + y_max) / 2
    y_range = y_max - y_min

    plotting_range = max([x_range, y_range]) + zoom_out

    ax.set_xlim(x_median - plotting_range / 2, x_median + plotting_range / 2)
    ax.set_ylim(y_median - plotting_range / 2, y_median + plotting_range / 2)

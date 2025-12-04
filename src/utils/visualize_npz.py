import argparse
from matplotlib.artist import get
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from scipy.stats import entropy, chi2_contingency, pearsonr
from numpy.fft import fft, fftfreq
import glob
from tqdm import tqdm
import datetime
import seaborn as sns
from matplotlib.colors import LogNorm

from urban_planner.config import CONFIG
from src.utils.plot_utils import get_styled_figure_ax, style_legend, DATASET_COLORS, DATASET_COLORS_5V5, DATASET_LINE_COLORS

def extract_metrics(npz_path, metrics):
    """
    Extracts a comprehensive set of metrics from a single .npz file.
    """
    sample_metrics = {'filepath': os.path.basename(npz_path)}
    data = np.load(npz_path, allow_pickle=True)
    input_stack = data['input']
    target_stack = data['target']
    metadata_normalized = data['metadata']
    temp_series_normalized = data['temperature_serie']

    meta_std = np.array(metrics['meta_std'])
    meta_mean = np.array(metrics['meta_mean'])
    metadata = metadata_normalized * meta_std + meta_mean
    meta_keys = ['lat', 'lon', 'population', 'delta_time_years']
    meta_dict = dict(zip(meta_keys, metadata))
    for k, v in meta_dict.items():
        sample_metrics[f'meta_{k}'] = v

    dw_class_names = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    channel_names = []
    channel_names.extend([f'dw_t1_{c}' for c in dw_class_names])
    channel_names.extend(['rgb_r', 'rgb_g', 'rgb_b'])
    channel_names.append('ndvi_t1')
    channel_names.append('temp_t1')
    channel_names.extend([f'dw_t2_{c}' for c in dw_class_names])

    for i, name in enumerate(channel_names):
        channel = input_stack[i]
        sample_metrics[f'input_{name}_mean'] = channel.mean()
        sample_metrics[f'input_{name}_std'] = channel.std()
        sample_metrics[f'input_{name}_min'] = channel.min()
        sample_metrics[f'input_{name}_max'] = channel.max()

    for i, name in enumerate(['ndvi_t2', 'temp_t2']):
        channel = target_stack[i]
        sample_metrics[f'target_{name}_mean'] = channel.mean()
        sample_metrics[f'target_{name}_std'] = channel.std()
        sample_metrics[f'target_{name}_min'] = channel.min()
        sample_metrics[f'target_{name}_max'] = channel.max()

    dw_t1 = input_stack[:9]
    dw_t2 = input_stack[14:23]
    ndvi_t1 = input_stack[12]
    temp_t1_normalized = input_stack[13]
    temp_t1 = temp_t1_normalized * metrics['temp_std'] + metrics['temp_mean']
    ndvi_t2 = target_stack[0]
    temp_t2_normalized = target_stack[1]
    temp_t2 = temp_t2_normalized * metrics['temp_std'] + metrics['temp_mean']
    temp_series = temp_series_normalized * metrics['temp_series_std'] + metrics['temp_series_mean']

    dw_t1_proportions = dw_t1.mean(axis=(1, 2))
    dw_t2_proportions = dw_t2.mean(axis=(1, 2))
    for i, name in enumerate(dw_class_names):
        sample_metrics[f'dw_t1_prop_{name}'] = dw_t1_proportions[i]
        sample_metrics[f'dw_t2_prop_{name}'] = dw_t2_proportions[i]

    sample_metrics['dw_t1_entropy'] = entropy(dw_t1_proportions[dw_t1_proportions > 0], base=2)
    sample_metrics['dw_t2_entropy'] = entropy(dw_t2_proportions[dw_t2_proportions > 0], base=2)

    if len(temp_series) > 1:
        sample_metrics['temp_series_mean'] = temp_series.mean()
        sample_metrics['temp_series_std'] = temp_series.std()
        if not np.all(temp_series == temp_series[0]):
            sample_metrics['temp_series_slope'] = np.polyfit(np.arange(len(temp_series)), temp_series, 1)[0]
        else:
            sample_metrics['temp_series_slope'] = 0.0
        ts_series = pd.Series(temp_series)
        if ts_series.std() > 0:
            sample_metrics['temp_series_autocorr_1'] = ts_series.autocorr(lag=1)
        else:
            sample_metrics['temp_series_autocorr_1'] = np.nan
        if len(temp_series) > 12:
            detrended = temp_series - np.polyval(np.polyfit(np.arange(len(temp_series)), temp_series, 1), np.arange(len(temp_series)))
            N = len(detrended)
            yf = fft(detrended)
            xf = fftfreq(N, 1/12.0)
            pos_mask = xf > 0
            if np.any(pos_mask):
                freq_idx = np.argmin(np.abs(xf[pos_mask] - 1.0))
                true_idx = np.where(pos_mask)[0][freq_idx]
                amplitude = 2.0/N * np.abs(yf[true_idx])
                sample_metrics['temp_series_seasonal_amplitude'] = amplitude
            else:
                sample_metrics['temp_series_seasonal_amplitude'] = np.nan
        else:
            sample_metrics['temp_series_seasonal_amplitude'] = np.nan
    else:
        sample_metrics['temp_series_mean'] = np.nan if len(temp_series) == 0 else temp_series[0]
        sample_metrics['temp_series_std'] = 0.0
        sample_metrics['temp_series_slope'] = 0.0
        sample_metrics['temp_series_autocorr_1'] = np.nan
        sample_metrics['temp_series_seasonal_amplitude'] = np.nan

    ndvi_diff = ndvi_t2 - ndvi_t1
    temp_diff = temp_t2 - temp_t1
    dw_diff = dw_t2_proportions - dw_t1_proportions
    sample_metrics['delta_ndvi_l1_norm'] = np.linalg.norm(ndvi_diff.flatten(), ord=1)
    sample_metrics['delta_ndvi_l2_norm'] = np.linalg.norm(ndvi_diff.flatten(), ord=2)
    sample_metrics['delta_temp_l1_norm'] = np.linalg.norm(temp_diff.flatten(), ord=1)
    sample_metrics['delta_temp_l2_norm'] = np.linalg.norm(temp_diff.flatten(), ord=2)
    dw_t1_built_prop = dw_t1_proportions[6]
    sample_metrics['pop_density_proxy'] = meta_dict['population'] / (dw_t1_built_prop + 1e-9)
    sample_metrics['ndvi_diff_mean'] = ndvi_diff.mean()
    sample_metrics['ndvi_diff_std'] = ndvi_diff.std()
    sample_metrics['ndvi_diff_min'] = ndvi_diff.min()
    sample_metrics['ndvi_diff_max'] = ndvi_diff.max()
    sample_metrics['temp_diff_mean'] = temp_diff.mean()
    sample_metrics['temp_diff_std'] = temp_diff.std()
    sample_metrics['temp_diff_min'] = temp_diff.min()
    sample_metrics['temp_diff_max'] = temp_diff.max()
    sample_metrics['dw_diff_mean'] = dw_diff.mean()
    sample_metrics['dw_diff_std'] = dw_diff.std()
    sample_metrics['dw_diff_min'] = dw_diff.min()
    sample_metrics['dw_diff_max'] = dw_diff.max()
    return sample_metrics

def visualize_npz(npz_path):
    """
    Visualizes the content of a single .npz file from the processed dataset.
    """
    if not os.path.exists(npz_path):
        print(f"Error: File not found at {npz_path}")
        return
    metrics_path = os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'normalization_metrics.json')
    if not os.path.exists(metrics_path):
        print(f"Error: Normalization metrics not found at {metrics_path}")
        return
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    data = np.load(npz_path, allow_pickle=True)
    input_stack = data['input']
    target_stack = data['target']
    metadata_normalized = data['metadata']
    temp_series_normalized = data['temperature_serie']
    dw_t1_image = np.argmax(input_stack[:9], axis=0)
    rgb_normalized = input_stack[9:12]
    rgb_image = (rgb_normalized * np.array(metrics['rgb_std'])[:, np.newaxis, np.newaxis] + np.array(metrics['rgb_mean'])[:, np.newaxis, np.newaxis]) * 255.0
    rgb_image = np.clip(rgb_image.transpose(1, 2, 0), 0, 255).astype(np.uint8)
    ndvi_image = input_stack[12]
    temp_normalized = input_stack[13]
    temp_image = temp_normalized * metrics['temp_std'] + metrics['temp_mean']
    dw_t2_image = np.argmax(input_stack[14:23], axis=0)
    target_ndvi = target_stack[0]
    target_temp_normalized = target_stack[1]
    target_temp = target_temp_normalized * metrics['temp_std'] + metrics['temp_mean']
    metadata = metadata_normalized * metrics['meta_std'] + metrics['meta_mean']
    temp_series = temp_series_normalized * metrics['temp_series_std'] + metrics['temp_series_mean']
    print(f"Number of temperature points retrieved: {len(temp_series)}")
    plt.figure(figsize=(20, 20))
    plt.suptitle(f"Un-normalized Visualization of {os.path.basename(npz_path)}", fontsize=16)
    ax_meta = plt.subplot2grid((4, 4), (0, 0))
    ax_meta.text(0, 1, "Metadata (un-normalized)", va='top', ha='left', fontsize=14, weight='bold')
    meta_keys = ['lat', 'lon', 'population', 'delta_time_years']
    meta_text = '\n'.join([f"{k}: {v:.4f}" for k, v in zip(meta_keys, metadata)])
    ax_meta.text(0, 0.95, meta_text, va='top', ha='left', fontsize=10)
    ax_meta.axis('off')
    input_images = [dw_t1_image, rgb_image, ndvi_image, temp_image, dw_t2_image]
    titles_input = ['Input DW (t1)', 'Input RGB (t1)', 'Input NDVI (t1)', 'Input Temp (t1)', 'Input DW (t2, argmax)']
    for i, (img, title) in enumerate(zip(input_images, titles_input)):
        ax = plt.subplot2grid((4, 5), (1, i))
        if len(img.shape) == 2:
            if 'DW' in title:
                im = ax.imshow(img, cmap='tab10', vmin=0, vmax=8)
            else:
                im = ax.imshow(img, cmap='viridis')
            plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    target_images = [target_ndvi, target_temp]
    titles_target = ['Target NDVI (t2)', 'Target Temp (t2)']
    for i, (img, title) in enumerate(zip(target_images, titles_target)):
        ax = plt.subplot2grid((4, 4), (0, i + 1))
        im = ax.imshow(img, cmap='viridis')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
    ax_temp_series = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    ax_temp_series.plot(temp_series)
    ax_temp_series.set_title('Historical Temperature Series (un-normalized)')
    ax_temp_series.set_xlabel('Year index (from 1951)')
    ax_temp_series.set_ylabel('Temperature')
    ax_temp_series.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def visualize_npz_normalized(npz_path):
    """
    Visualizes the normalized content of a single .npz file as the model sees it.
    """
    if not os.path.exists(npz_path):
        print(f"Error: File not found at {npz_path}")
        return
    data = np.load(npz_path, allow_pickle=True)
    input_stack = data['input']
    target_stack = data['target']
    metadata_normalized = data['metadata']
    temp_series_normalized = data['temperature_serie']
    dw_t1_image = np.argmax(input_stack[:9], axis=0)
    rgb_image = input_stack[9:12].transpose(1, 2, 0)
    ndvi_image = input_stack[12]
    temp_image = input_stack[13]
    dw_t2_image = np.argmax(input_stack[14:23], axis=0)
    target_ndvi = target_stack[0]
    target_temp = target_stack[1]
    plt.figure(figsize=(20, 20))
    plt.suptitle(f"Normalized Visualization (Model's View) of {os.path.basename(npz_path)}", fontsize=16)
    ax_meta = plt.subplot2grid((4, 4), (0, 0))
    ax_meta.text(0, 1, "Metadata (normalized)", va='top', ha='left', fontsize=14, weight='bold')
    meta_keys = ['lat', 'lon', 'population', 'delta_time_years']
    meta_text = '\n'.join([f"{k}: {v:.4f}" for k, v in zip(meta_keys, metadata_normalized)])
    ax_meta.text(0, 0.95, meta_text, va='top', ha='left', fontsize=10)
    ax_meta.axis('off')
    input_images = [dw_t1_image, rgb_image, ndvi_image, temp_image, dw_t2_image]
    titles_input = ['Input DW (t1, argmax)', 'Input RGB (norm)', 'Input NDVI', 'Input Temp (norm)', 'Input DW (t2, argmax)']
    for i, (img, title) in enumerate(zip(input_images, titles_input)):
        ax = plt.subplot2grid((4, 5), (1, i))
        im = ax.imshow(img, cmap='viridis')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
    target_images = [target_ndvi, target_temp]
    titles_target = ['Target NDVI', 'Target Temp (norm)']
    for i, (img, title) in enumerate(zip(target_images, titles_target)):
        ax = plt.subplot2grid((4, 4), (0, i + 1))
        im = ax.imshow(img, cmap='viridis')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
    ax_temp_series = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    ax_temp_series.plot(temp_series_normalized)
    ax_temp_series.set_title('Historical Temperature Series (normalized)')
    ax_temp_series.set_xlabel('Year index (from 1951)')
    ax_temp_series.set_ylabel('Normalized Temperature')
    ax_temp_series.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def plot_geographical_distributions(df, output_dir):
    print("Plotting geographical distributions...")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='meta_lon', y='meta_lat', hue='split')
    plt.title('Geographical Distribution of Samples by Split')
    plt.savefig(os.path.join(output_dir, 'world_distribution_by_split.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(14, 8))
    pop_norm = LogNorm(vmin=max(1, df['meta_population'].min()), vmax=df['meta_population'].max())
    sns.scatterplot(data=df, x='meta_lon', y='meta_lat', hue='meta_population', size='meta_population', palette='viridis', hue_norm=pop_norm)
    plt.title('Geographical Distribution by Population')
    plt.savefig(os.path.join(output_dir, 'world_distribution_by_population.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='meta_lon', y='meta_lat', hue='dw_t1_prop_built', palette='magma')
    plt.title('Geographical Distribution by Built Area Proportion (t1)')
    plt.savefig(os.path.join(output_dir, 'world_distribution_by_built_prop.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='meta_lon', y='meta_lat', hue='green_prop_t1', palette='summer')
    plt.title('Geographical Distribution by Green Area Proportion (t1)')
    plt.savefig(os.path.join(output_dir, 'world_distribution_by_green_prop.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='meta_lon', y='meta_lat', hue='temp_diff_mean', palette='coolwarm')
    plt.title('Geographical Distribution by Mean Temperature Change (t2-t1)')
    plt.savefig(os.path.join(output_dir, 'world_distribution_by_temp_change.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='meta_lon', y='meta_lat', hue='ndvi_diff_mean', palette='RdYlGn')
    plt.title('Geographical Distribution by Mean NDVI Change (t2-t1)')
    plt.savefig(os.path.join(output_dir, 'world_distribution_by_ndvi_change.png'), dpi=300)
    plt.close()

def plot_correlation_heatmaps(df, output_dir):
    print("Plotting correlation heatmaps...")
    key_metrics = ['t2_year', 'meta_population', 'meta_delta_time_years', 'dw_t1_prop_built', 'dw_t1_prop_trees', 'dw_t1_prop_water', 'dw_t1_prop_crops','ndvi_diff_mean', 'temp_diff_mean', 'delta_ndvi_l2_norm', 'delta_temp_l2_norm','temp_series_mean', 'temp_series_slope', 'dw_t1_entropy', 'pop_density_proxy']
    corr = df[key_metrics].corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Matrix of Key Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap_main.png'), dpi=300)
    plt.close()

def plot_distributions(df, output_dir):
    print("Plotting metric distributions...")
    metrics_to_compare = ['meta_population', 'meta_delta_time_years', 'ndvi_diff_mean', 'temp_diff_mean', 'dw_t1_prop_built']
    fig, axes = plt.subplots(len(metrics_to_compare), 1, figsize=(10, 5 * len(metrics_to_compare)))
    for i, metric in enumerate(metrics_to_compare):
        sns.boxplot(data=df, x='split', y=metric, ax=axes[i])
        axes[i].set_title(f'Distribution of {metric} by Split')
        if metric == 'meta_population':
            axes[i].set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distributions_by_split.png'), dpi=300)
    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(df['ndvi_diff_mean'], kde=True, ax=axes[0], color='green')
    axes[0].set_title('Distribution of Mean NDVI Change')
    sns.histplot(df['temp_diff_mean'], kde=True, ax=axes[1], color='red')
    axes[1].set_title('Distribution of Mean Temperature Change')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'changes_distribution.png'), dpi=300)
    plt.close()

def plot_relationships(df, output_dir):
    print("Plotting key relationships...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.scatterplot(data=df, x='meta_population', y='temp_diff_mean', ax=axes[0], alpha=0.5)
    axes[0].set_xscale('log')
    axes[0].set_title('Population vs. Mean Temperature Change')
    sns.scatterplot(data=df, x='meta_population', y='ndvi_diff_mean', ax=axes[1], alpha=0.5)
    axes[1].set_xscale('log')
    axes[1].set_title('Population vs. Mean NDVI Change')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'population_vs_changes.png'), dpi=300)
    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.scatterplot(data=df, x='meta_delta_time_years', y='delta_temp_l2_norm', ax=axes[0], alpha=0.5)
    axes[0].set_title('Time Delta vs. Temperature Change (L2 Norm)')
    sns.scatterplot(data=df, x='meta_delta_time_years', y='delta_ndvi_l2_norm', ax=axes[1], alpha=0.5)
    axes[1].set_title('Time Delta vs. NDVI Change (L2 Norm)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_delta_vs_changes.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='dw_t1_prop_built', y='temp_diff_mean', alpha=0.5)
    plt.title('Built-up Area Proportion vs. Mean Temperature Change')
    plt.xlabel('Proportion of Built-up Area at t1')
    plt.ylabel('Mean Temperature Change (t2-t1)')
    plt.savefig(os.path.join(output_dir, 'built_up_vs_temp.png'), dpi=300)
    plt.close()

def plot_semantic_proportions(df, output_dir):
    print("Plotting semantic class proportions...")
    dw_class_names = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    prop_cols = [f'dw_t1_prop_{c}' for c in dw_class_names]
    mean_props = df[prop_cols].mean().sort_values(ascending=False)
    mean_props.index = [s.replace('dw_t1_prop_', '') for s in mean_props.index]
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=mean_props.index, y=mean_props.values)
    ax.set_title('Mean Semantic Class Proportions Across Dataset (t1)')
    ax.set_xlabel('Semantic Class')
    ax.set_ylabel('Mean Proportion')
    plt.xticks(rotation=45, ha='right')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'semantic_class_proportions.png'), dpi=300)
    plt.close()

def plot_change_interactions(df, output_dir):
    print("Plotting change interactions...")
    ndvi_quantiles = df['ndvi_diff_mean'].quantile([0.25, 0.75])
    temp_quantiles = df['temp_diff_mean'].quantile([0.25, 0.75])
    def categorize_change(x, quantiles, prefix):
        if x < quantiles.iloc[0]:
            return f'{prefix} High Neg'
        elif x > quantiles.iloc[1]:
            return f'{prefix} High Pos'
        else:
            return f'{prefix} Low'
    df['ndvi_change_cat'] = df['ndvi_diff_mean'].apply(lambda x: categorize_change(x, ndvi_quantiles, 'NDVI'))
    df['temp_change_cat'] = df['temp_diff_mean'].apply(lambda x: categorize_change(x, temp_quantiles, 'Temp'))
    contingency_table = pd.crosstab(df['ndvi_change_cat'], df['temp_change_cat'])
    cat_order = ['High Neg', 'Low', 'High Pos']
    ndvi_order = [f'NDVI {c}' for c in cat_order]
    temp_order = [f'Temp {c}' for c in cat_order]
    contingency_table = contingency_table.reindex(index=ndvi_order, columns=temp_order)
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='viridis')
    plt.title('Interaction between NDVI and Temperature Changes')
    plt.xlabel('Temperature Change Category')
    plt.ylabel('NDVI Change Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'change_interaction_heatmap.png'), dpi=300)
    plt.close()
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    sns.regplot(data=df, x='dw_prop_diff_built', y='temp_diff_mean', ax=axes[0, 0], scatter_kws={'alpha':0.3})
    axes[0, 0].set_title('Built Area Change vs. Temperature Change')
    axes[0, 0].set_xlabel('Change in Built Area Proportion')
    axes[0, 0].set_ylabel('Mean Temperature Change')
    sns.regplot(data=df, x='dw_prop_diff_built', y='ndvi_diff_mean', ax=axes[0, 1], scatter_kws={'alpha':0.3})
    axes[0, 1].set_title('Built Area Change vs. NDVI Change')
    axes[0, 1].set_xlabel('Change in Built Area Proportion')
    axes[0, 1].set_ylabel('Mean NDVI Change')
    sns.regplot(data=df, x='green_prop_diff', y='temp_diff_mean', ax=axes[1, 0], scatter_kws={'alpha':0.3})
    axes[1, 0].set_title('Green Area Change vs. Temperature Change')
    axes[1, 0].set_xlabel('Change in Green Area Proportion')
    axes[1, 0].set_ylabel('Mean Temperature Change')
    sns.regplot(data=df, x='green_prop_diff', y='ndvi_diff_mean', ax=axes[1, 1], scatter_kws={'alpha':0.3})
    axes[1, 1].set_title('Green Area Change vs. NDVI Change')
    axes[1, 1].set_xlabel('Change in Green Area Proportion')
    axes[1, 1].set_ylabel('Mean NDVI Change')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'semantic_change_vs_env_change.png'), dpi=300)
    plt.close()

def plot_incremental_zero_diffs(df: pd.DataFrame, output_dir: str, thresholds=None):
    """
    Reports how many samples have NDVI or Temp differences close to zero
    for a set of incremental thresholds.
    
    thresholds: list of thresholds (absolute value) around zero to check
    """
    if thresholds is None:
        thresholds = [0, 1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.05, 0.1]
    
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    for t in thresholds:
        ndvi_count = (df['ndvi_diff_mean'].abs() <= t).sum()
        temp_count = (df['temp_diff_mean'].abs() <= t).sum()
        both_count = ((df['ndvi_diff_mean'].abs() <= t) & (df['temp_diff_mean'].abs() <= t)).sum()
        dw_count = (df['dw_diff_mean'].abs() <= t).sum()
        total = len(df)
        summary_rows.append({
            'Threshold': t,
            'NDVI_count': ndvi_count,
            'NDVI_prop': ndvi_count / total,
            'Temp_count': temp_count,
            'Temp_prop': temp_count / total,
            'Both_count': both_count,
            'Both_prop': both_count / total,
            'DW_count': dw_count,
            'DW_prop': dw_count / total
        })
    summary = pd.DataFrame(summary_rows)
    print("Incremental zero diffs:")
    print(summary)
    plt.figure(figsize=(8, 5))
    plt.plot(summary['Threshold'], summary['NDVI_count'], marker='o', label='NDVI')
    plt.plot(summary['Threshold'], summary['Temp_count'], marker='o', label='Temp')
    plt.plot(summary['Threshold'], summary['Both_count'], marker='o', label='Both')
    plt.plot(summary['Threshold'], summary['DW_count'], marker='o', label='DW')
    plt.xlabel('Threshold around zero (absolute value)')
    plt.ylabel('Number of samples')
    plt.title('Incremental near-zero differences')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "incremental_zero_diffs_plot.png"))
    plt.close()

def plot_temporal_distributions(df, output_dir):
    """Plots distributions of temporal variables."""
    print("Plotting temporal distributions...")
    if 't1_year' not in df.columns or 't2_year' not in df.columns:
        print("Missing t1_year or t2_year columns.")
        return

    t1_counts = df['t1_year'].dropna().astype(int).value_counts().sort_index()
    t2_counts = df['t2_year'].dropna().astype(int).value_counts().sort_index()
    
    all_years = sorted(list(set(t1_counts.index) | set(t2_counts.index)))
    t1_series = t1_counts.reindex(all_years, fill_value=0)
    t2_series = t2_counts.reindex(all_years, fill_value=0)
    
    print(f"Years: {', '.join(map(str, all_years))}")
    print(f"t1: {', '.join(map(str, t1_series.values))}")
    print(f"t2: {', '.join(map(str, t2_series.values))}")
    
    fig, ax = get_styled_figure_ax(figsize=(12, 6), aspect='none', grid=True)
    
    ax.plot(all_years, t1_series, marker='o', label='T1 Year', color=DATASET_LINE_COLORS[0])
    ax.plot(all_years, t2_series, marker='o', label='T2 Year', color=DATASET_LINE_COLORS[3])
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    
    total_samples = len(df)
    ax.text(0.05, 0.95, f"Total Samples: {total_samples}", transform=ax.transAxes,
            fontsize=14, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
            
    style_legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_distributions.pdf'), dpi=300)
    plt.close()

def plot_year_distribution_split(df, year_col, output_name_prefix, title_suffix, output_dir):
    """
    Plots separate bar charts for Known and Unknown cities per year.
    Grouped by year.
    Bars for each split (Train, Val, Test).
    """
    print(f"Plotting split samples per year ({year_col}) - {title_suffix}...")
    if year_col not in df.columns or 'split' not in df.columns or 'filepath' not in df.columns:
        print(f"Missing required columns for plot: {year_col}, split, or filepath")
        return

    df_plot = df.dropna(subset=[year_col]).copy()
    df_plot[year_col] = df_plot[year_col].astype(int)
    
    # Extract city names: everything before the first digit
    def get_city_name(filepath):
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        city_parts = []
        for p in parts:
            if p.isdigit():
                break
            city_parts.append(p)
        return "_".join(city_parts)

    df_plot['city_name'] = df_plot['filepath'].apply(get_city_name)
    
    # Identify known cities from training set
    train_df = df_plot[df_plot['split'].str.contains('train', case=False)]
    known_cities = set(train_df['city_name'].unique())
    
    # Mark rows as known/unknown
    df_plot['is_known'] = df_plot['city_name'].isin(known_cities)
    
    years = sorted(df_plot[year_col].unique())
    unique_splits = df_plot['split'].unique()
    
    # Determine order: Train first, then others
    train_split = next((s for s in unique_splits if 'train' in s.lower()), None)
    split_order = []
    if train_split:
        split_order.append(train_split)
    split_order.extend([s for s in unique_splits if s != train_split])
    
    # Function to plot a subset (known or unknown)
    def plot_subset(is_known_val, label_suffix):
        subset_df = df_plot[df_plot['is_known'] == is_known_val]
        if subset_df.empty:
            print(f"No data for {label_suffix} cities in {year_col}.")
            return

        counts = subset_df.groupby([year_col, 'split']).size().unstack(fill_value=0)
        counts = counts.reindex(years, fill_value=0) # Ensure all years are present
        
        fig, ax = get_styled_figure_ax(figsize=(14, 7), aspect='none', grid=True)
        
        bar_width = 0.20 # Slightly thinner bars
        indices = np.arange(len(years))
        
        num_splits = len(split_order)
        total_width = num_splits * bar_width
        start_offset = -total_width / 2 + bar_width / 2
        
        legend_handles = []
        
        for i, split in enumerate(split_order):
            offset = start_offset + i * bar_width
            
            vals = counts.get(split, pd.Series([0]*len(years), index=years)).values
            
            # Assign colors
            color_idx = 0
            if 'train' in split.lower(): color_idx = 0
            elif 'val' in split.lower(): color_idx = 1
            elif 'test' in split.lower(): color_idx = 2
            else: color_idx = 3
            
            c = DATASET_COLORS[color_idx % len(DATASET_COLORS)]
            
            ax.bar(indices + offset, vals, bar_width, label=split.capitalize(), color=c, edgecolor='black')
            
            # For custom legend order
            from matplotlib.patches import Patch
            if not any(h.get_label() == split.capitalize() for h in legend_handles):
                 legend_handles.append(Patch(facecolor=c, edgecolor='black', label=split.capitalize()))

        ax.set_xlabel(f'Year ({title_suffix})')
        ax.set_ylabel('Count')
        # ax.set_title(f'Sample Counts per Year - {label_suffix} Cities')
        ax.set_xticks(indices)
        ax.set_xticklabels(years)
        
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(legend_handles), frameon=False)
        
        plt.tight_layout()
        output_filename = f"{output_name_prefix}_{label_suffix.lower()}.pdf"
        plt.savefig(os.path.join(output_dir, output_filename), dpi=300)
        plt.close()

    # Generate plots
    plot_subset(True, "Known")
    plot_subset(False, "Unknown")

def print_data_summary(df):
    """
    Prints a summary of sample counts for T1 and T2 years.
    Breakdown:
    - Total
    - Known Cities (Overall)
    - Unknown Cities (Overall)
    - Known Cities per Split (Train, Val, Test)
    - Unknown Cities per Split (Train, Val, Test)
    """
    print("\n" + "="*30)
    print("DATA SUMMARY")
    print("="*30)

    if 't1_year' not in df.columns or 't2_year' not in df.columns:
        print("Missing t1_year or t2_year columns.")
        return

    # Ensure years are int
    df = df.copy()
    df['t1_year'] = df['t1_year'].fillna(-1).astype(int)
    df['t2_year'] = df['t2_year'].fillna(-1).astype(int)

    # City extraction and known/unknown logic
    def get_city_name(filepath):
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        city_parts = []
        for p in parts:
            if p.isdigit():
                break
            city_parts.append(p)
        return "_".join(city_parts)

    df['city_name'] = df['filepath'].apply(get_city_name)
    train_df = df[df['split'].str.contains('train', case=False)]
    known_cities = set(train_df['city_name'].unique())
    df['is_known'] = df['city_name'].isin(known_cities)

    # Helper to print counts
    def print_counts(label, subset_df):
        t1_counts = subset_df['t1_year'].value_counts().sort_index()
        t2_counts = subset_df['t2_year'].value_counts().sort_index()
        
        # Filter out -1 (NaNs)
        t1_counts = t1_counts[t1_counts.index != -1]
        t2_counts = t2_counts[t2_counts.index != -1]

        print(f"\n{label}:")
        print(f"t1: {', '.join([f'{y}: {c}' for y, c in t1_counts.items()])}")
        print(f"t2: {', '.join([f'{y}: {c}' for y, c in t2_counts.items()])}")

    # 1. Total count
    print_counts("Total count", df)

    # 2. Total known/unknown
    print_counts("Total count of known cities", df[df['is_known']])
    print_counts("Total count of unknown cities", df[~df['is_known']])

    # 3. Per Split (Known)
    splits = sorted(df['split'].unique())
    for split in splits:
        print_counts(f"Total count of known cities (split={split})", df[(df['is_known']) & (df['split'] == split)])

    # 4. Per Split (Unknown)
    for split in splits:
        print_counts(f"Total count of unknown cities (split={split})", df[(~df['is_known']) & (df['split'] == split)])
    
    print("="*30 + "\n")

def visualize_metrics(csv_path, output_dir):    
    """
    Generates and saves a suite of visualizations from the metrics CSV file.
    """
    if sns is None:
        return
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    print(f"Loading metrics from {csv_path}")
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to {output_dir}")

    def extract_t2_year(filepath):
        try:
            return int(filepath.split('_')[-2])
        except (IndexError, ValueError):
            return np.nan
    
    def extract_t1_year(filepath):
        try:
            # Assumes format ..._t1year_t1month_to_t2year_t2month.npz
            # e.g. ..._2021_03_to_2024_05.npz
            return int(filepath.split('_')[-5])
        except (IndexError, ValueError):
            return np.nan

    df['t2_year'] = df['filepath'].apply(extract_t2_year)
    df['t1_year'] = df['filepath'].apply(extract_t1_year)

    green_cols_t1 = ['dw_t1_prop_trees', 'dw_t1_prop_grass', 'dw_t1_prop_shrub_and_scrub']
    df['green_prop_t1'] = df[green_cols_t1].sum(axis=1)
    green_cols_t2 = ['dw_t2_prop_trees', 'dw_t2_prop_grass', 'dw_t2_prop_shrub_and_scrub']
    df['green_prop_t2'] = df[green_cols_t2].sum(axis=1)
    df['green_prop_diff'] = df['green_prop_t2'] - df['green_prop_t1']
    dw_class_names = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    for c in dw_class_names:
        df[f'dw_prop_diff_{c}'] = df[f'dw_t2_prop_{c}'] - df[f'dw_t1_prop_{c}']
    plot_geographical_distributions(df, output_dir)
    plot_correlation_heatmaps(df, output_dir)
    plot_distributions(df, output_dir)
    plot_relationships(df, output_dir)
    plot_semantic_proportions(df, output_dir)
    plot_change_interactions(df, output_dir)
    plot_incremental_zero_diffs(df, output_dir)
    print_data_summary(df)
    plot_temporal_distributions(df, output_dir)
    plot_year_distribution_split(df, 't1_year', 'samples_per_year_t1', 't1', output_dir)
    plot_year_distribution_split(df, 't2_year', 'samples_per_year_t2', 't2', output_dir)
    print("All visualizations saved.")

def analyze_metrics(csv_path, output_report_path):
    """
    Performs statistical analysis on the metrics CSV and writes a report.
    """
    print(f"Loading metrics from {csv_path} for analysis...")
    df = pd.read_csv(csv_path)
    green_cols_t1 = ['dw_t1_prop_trees', 'dw_t1_prop_grass', 'dw_t1_prop_shrub_and_scrub']
    df['green_prop_t1'] = df[green_cols_t1].sum(axis=1)
    green_cols_t2 = ['dw_t2_prop_trees', 'dw_t2_prop_grass', 'dw_t2_prop_shrub_and_scrub']
    df['green_prop_t2'] = df[green_cols_t2].sum(axis=1)
    df['green_prop_diff'] = df['green_prop_t2'] - df['green_prop_t1']
    dw_class_names = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    for c in dw_class_names:
        df[f'dw_prop_diff_{c}'] = df[f'dw_t2_prop_{c}'] - df[f'dw_t1_prop_{c}']
    df['log_meta_population'] = np.log(df['meta_population'] + 1)
    with open(output_report_path, 'w') as f:
        f.write("Statistical Analysis Report\n")
        f.write("===========================\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data source: {os.path.basename(csv_path)}\n\n")
        f.write("----------------------------------------------------\n")
        f.write("Section 1: Interaction between NDVI and Temperature Changes\n")
        f.write("----------------------------------------------------\n\n")
        f.write("Test: Chi-squared Test of Independence\n")
        f.write("Purpose: To determine if there is a significant association between the category of NDVI change and the category of temperature change.\n")
        f.write("Null Hypothesis (H0): The two variables are independent.\n\n")
        ndvi_quantiles = df['ndvi_diff_mean'].quantile([0.25, 0.75])
        temp_quantiles = df['temp_diff_mean'].quantile([0.25, 0.75])
        def categorize_change(x, quantiles, prefix):
            if x < quantiles.iloc[0]: return f'{prefix} High Neg'
            elif x > quantiles.iloc[1]: return f'{prefix} High Pos'
            else: return f'{prefix} Low'
        df['ndvi_change_cat'] = df['ndvi_diff_mean'].apply(lambda x: categorize_change(x, ndvi_quantiles, 'NDVI'))
        df['temp_change_cat'] = df['temp_diff_mean'].apply(lambda x: categorize_change(x, temp_quantiles, 'Temp'))
        contingency_table = pd.crosstab(df['ndvi_change_cat'], df['temp_change_cat'])
        cat_order = ['High Neg', 'Low', 'High Pos']
        ndvi_order = [f'NDVI {c}' for c in cat_order]
        temp_order = [f'Temp {c}' for c in cat_order]
        contingency_table = contingency_table.reindex(index=ndvi_order, columns=temp_order)
        f.write("Contingency Table:\n")
        f.write(contingency_table.to_string())
        f.write("\n\n")
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        f.write("Results:\n")
        f.write(f"- Chi-squared statistic: {chi2:.4f}\n")
        f.write(f"- p-value: {p:.4g}\n")
        f.write(f"- Degrees of freedom: {dof}\n\n")
        f.write("Interpretation:\n")
        if p < 0.05:
            f.write(f"Since the p-value (p={p:.4g}) is less than 0.05, we reject the null hypothesis. ")
            f.write("This suggests there is a statistically significant association between the categories of NDVI change and temperature change.\n\n")
        else:
            f.write(f"Since the p-value (p={p:.4g}) is not less than 0.05, we fail to reject the null hypothesis. ")
            f.write("There is not enough evidence to suggest a significant association.\n\n")
        f.write("----------------------------------------------------\n")
        f.write("Section 2: Correlation Analysis of Environmental and Urban Changes\n")
        f.write("----------------------------------------------------\n\n")
        f.write("Test: Pearson Correlation Coefficient\n")
        f.write("Purpose: To measure the linear relationship between pairs of continuous variables.\n")
        f.write("Null Hypothesis (H0): There is no linear correlation between the two variables (r=0).\n\n")
        analyses = {
            "Change in Built Area vs. Mean Temperature Change": ('dw_prop_diff_built', 'temp_diff_mean'),
            "Change in Built Area vs. Mean NDVI Change": ('dw_prop_diff_built', 'ndvi_diff_mean'),
            "Change in Green Area vs. Mean Temperature Change": ('green_prop_diff', 'temp_diff_mean'),
            "Change in Green Area vs. Mean NDVI Change": ('green_prop_diff', 'ndvi_diff_mean'),
            "Log Population vs. Mean Temperature Change": ('log_meta_population', 'temp_diff_mean'),
            "Log Population vs. Mean NDVI Change": ('log_meta_population', 'ndvi_diff_mean'),
            "Time Delta vs. Total Temperature Change (L2)": ('meta_delta_time_years', 'delta_temp_l2_norm'),
            "Time Delta vs. Total NDVI Change (L2)": ('meta_delta_time_years', 'delta_ndvi_l2_norm'),
        }
        for title, (var1, var2) in analyses.items():
            clean_df = df[[var1, var2]].dropna()
            r, p_corr = pearsonr(clean_df[var1], clean_df[var2])
            f.write(f"---\nAnalysis: {title}\n")
            f.write(f"- Variables: {var1}, {var2}\n")
            f.write(f"- Pearson's r: {r:.4f}\n")
            f.write(f"- p-value: {p_corr:.4g}\n")
            f.write("- Interpretation: ")
            if p_corr < 0.05:
                strength = "very weak"
                if abs(r) > 0.7: strength = "strong"
                elif abs(r) > 0.4: strength = "moderate"
                elif abs(r) > 0.2: strength = "weak"
                direction = "positive" if r > 0 else "negative"
                f.write(f"There is a {strength}, {direction}, and statistically significant linear relationship (r={r:.2f}, p={p_corr:.4g}). ")
                f.write(f"As '{var1}' increases, '{var2}' tends to {'increase' if r > 0 else 'decrease'}.\n\n")
            else:
                f.write(f"The relationship is not statistically significant (p={p_corr:.4g}). We cannot conclude there is a linear relationship.\n\n")
        f.write("End of Report.\n")
    print(f"Statistical report saved to {output_report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize, extract metrics from, or analyze processed dataset .npz files.")
    subparsers = parser.add_subparsers(dest='command', required=True)
    parser_vis = subparsers.add_parser('visualize', help='Visualize a single .npz file.')
    parser_vis.add_argument("npz_file", type=str, help="Path to the .npz file to visualize.")
    parser_vis.add_argument('--normalized', action='store_true', help="Show normalized data as the model sees it.")
    parser_vis.add_argument('--unnormalized', action='store_true', help="Show un-normalized data for human interpretation.")
    parser_extract = subparsers.add_parser('extract', help='Extract metrics from all .npz files in a directory.')
    parser_extract.add_argument("input_dir", type=str, help="Root directory of the processed dataset containing split subfolders (e.g., train, val, test).")
    parser_extract.add_argument("output_csv", type=str, help="Path to save the output CSV file.")
    parser_vis_csv = subparsers.add_parser('visualize_csv', help='Generate visualizations from a metrics CSV file.')
    parser_vis_csv.add_argument("csv_path", type=str, help="Path to the metrics CSV file.")
    parser_vis_csv.add_argument("output_dir", type=str, help="Directory to save the visualization plots.")
    parser_analyze_csv = subparsers.add_parser('analyze_csv', help='Generate a statistical analysis report from a metrics CSV file.')
    parser_analyze_csv.add_argument("csv_path", type=str, help="Path to the metrics CSV file.")
    parser_analyze_csv.add_argument("output_report_path", type=str, help="Path to save the output analysis report file.")
    args = parser.parse_args()
    if args.command == 'visualize':
        if not args.normalized and not args.unnormalized:
            visualize_npz(args.npz_file)
            visualize_npz_normalized(args.npz_file)
        else:
            if args.unnormalized:
                visualize_npz(args.npz_file)
            if args.normalized:
                visualize_npz_normalized(args.npz_file)
        plt.show()
    elif args.command == 'extract':
        all_metrics = []
        split_dirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
        if not split_dirs:
            print(f"No split subdirectories found in {args.input_dir}. Looking for .npz files directly.")
            split_dirs = ['.']
        metrics_path = os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'normalization_metrics.json')
        if not os.path.exists(metrics_path):
            print(f"Error: Normalization metrics not found at {metrics_path}")
            exit()
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        for split in split_dirs:
            split_name = split if split != '.' else 'unknown'
            split_path = os.path.join(args.input_dir, split)
            npz_files = glob.glob(os.path.join(split_path, '*.npz'))
            if not npz_files:
                if split != '.':
                    print(f"No .npz files found in split: {split}")
                continue
            print(f"Processing {len(npz_files)} files from split: {split_name}")
            for f in tqdm(npz_files, desc=f"Extracting metrics for {split_name} split"):
                try:
                    sample_metrics = extract_metrics(f, metrics)
                    sample_metrics['split'] = split_name
                    all_metrics.append(sample_metrics)
                except Exception as e:
                    print(f"Failed to process {f}: {e}")
        if not all_metrics:
            print("No metrics were extracted from any split.")
            exit()
        df = pd.DataFrame(all_metrics)
        cols = df.columns.tolist()
        if 'split' in cols:
            cols.insert(1, cols.pop(cols.index('split')))
            df = df[cols]
        df.to_csv(args.output_csv, index=False)
        print(f"Metrics for all splits saved to {args.output_csv}")
    elif args.command == 'visualize_csv':
        visualize_metrics(args.csv_path, args.output_dir)
    elif args.command == 'analyze_csv':
        analyze_metrics(args.csv_path, args.output_report_path)

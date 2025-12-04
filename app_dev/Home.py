from loguru import logger
import streamlit as st
import torch
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from urban_planner.config import CONFIG
from src.dataset import collate_fn
from src.utils.plot_utils import get_styled_figure_ax, style_legend
from src.utils.visualization import HEX_COLORS, DW_CLASSES
from app_dev.app_src.utils import (
    load_model,
    load_dataset,
    load_normalization_metrics,
    plot_zoomed_views,
    get_unnormalized_data,
)
from app_dev.app_src.model_diagram import instantiate_model_diagram

st.set_page_config(layout="wide")

def home():
    """Main function for the single model visualization page."""
    CONFIG.device='cpu'
    st.title("Single Model Visualization")

    # --- Sidebar for Configuration ---
    st.sidebar.header("Configuration")
    checkpoint_path = st.sidebar.text_input("Path to Checkpoint", value="models/lgs-3c-futurefull-metaemb_trial_0_best.pth")
    
    model, checkpoint, additional_embeddings, metadata_only_embeddings = load_model(checkpoint_path)
    metadata_input_length = checkpoint.get('metadata_input_length', 4)
    if not model:
        st.stop()

    dataset = load_dataset()
    if not dataset:
        st.stop()
        
    metrics = load_normalization_metrics()

    # --- Sidebar for Data Selection ---
    st.sidebar.header("Data Sample Selection")
    selection_mode = st.sidebar.radio("Selection Mode", ["Index", "Cycle", "Filename"], index=1) #default = Cycle

    if 'sample_idx' not in st.session_state:
        st.session_state.sample_idx = 0

    if selection_mode == "Index":
        st.session_state.sample_idx = st.sidebar.number_input("Sample Index", min_value=0, max_value=len(dataset)-1, value=st.session_state.sample_idx, step=1)
    elif selection_mode == "Cycle":
        col1, col2 = st.sidebar.columns(2)
        if col1.button("⬅️ Previous"):
            st.session_state.sample_idx = (st.session_state.sample_idx - 1) % len(dataset)
        if col2.button("Next ➡️"):
            st.session_state.sample_idx = (st.session_state.sample_idx + 1) % len(dataset)
        st.sidebar.write(f"Current Index: {st.session_state.sample_idx}")
    else: # Filename
        filenames = [os.path.basename(p) for p in dataset.file_list]
        selected_file = st.sidebar.selectbox("Select by Filename", filenames, index=st.session_state.sample_idx)
        st.session_state.sample_idx = filenames.index(selected_file)

    # --- Data Loading and Prediction ---
    # try:
    sample = dataset[st.session_state.sample_idx]
    # Use collate_fn to get the correct batch shape and device placement
    batch = collate_fn([sample])
    input_tensor, meta_tensor, temp_series_tensor, _, t1_dates, t2_dates, target_tensor = batch

    with torch.no_grad():
        if metadata_input_length == 8:
            meta_tensor = torch.cat([meta_tensor, t1_dates, t2_dates], dim=1)
        output_tensor = model(input_tensor, temp_series_tensor, meta_tensor)
        logger.debug(f"Model output shape: {output_tensor.shape}")
        logger.debug(f"Output stats: min={output_tensor.min().item()}, max={output_tensor.max().item()}, mean={output_tensor.mean().item()}")

    # Move to CPU and convert to numpy for visualization
    input_np = input_tensor[0].cpu().numpy()
    meta_np = meta_tensor[0].cpu().numpy()
    temp_series_np = temp_series_tensor[0].cpu().numpy()
    target_np = target_tensor[0].cpu().numpy()
    output_np = output_tensor[0].cpu().numpy()
    
    # Un-normalize data
    output_channels_min = [output_np[i].min() for i in range(output_np.shape[0])]
    output_channels_max = [output_np[i].max() for i in range(output_np.shape[0])]
    logger.warning(meta_np)
    inputs_viz, meta_viz, temp_series_viz, targets_viz, outputs_viz = get_unnormalized_data(
        input_np, meta_np, temp_series_np, target_np, output_np, metrics
    )
    logger.warning(meta_viz)
    output_normalized_channels_min = [outputs_viz[i].min() for i in range(outputs_viz.shape[0])]
    output_normalized_channels_max = [outputs_viz[i].max() for i in range(outputs_viz.shape[0])]

    # except Exception as e:
    #     st.error(f"Failed to process sample {st.session_state.sample_idx}: {e}")
    #     st.stop()

    # --- Main Panel for Visualization ---

    model_type = checkpoint.get('model_type', 'unet')
    st.header(f'Model visualisation: `{model_type.upper()}` architecture')
    instantiate_model_diagram(
        model_type=model_type,
        metadata_length=checkpoint.get('metadata_input_length', 4)
    )

    st.header("Model Inputs")
    st.write(f"Showing sample `{os.path.basename(dataset.file_list[st.session_state.sample_idx])}`")

    if st.button("Save Input & Target Plots as PDF"):
        save_pdfs = True
    else:
        save_pdfs = False

    cols = st.columns(len(inputs_viz))
    for i, (title, img) in enumerate(inputs_viz.items()):
        with cols[i]:
            st.subheader(title)
            fig, ax = get_styled_figure_ax(aspect='none')

            if "DW" in title:
                ax.imshow(img)
                # ax.set_title(f"{title} (Categorical)")
                for i, color in enumerate(HEX_COLORS):
                    ax.plot([], [], color=color, label=DW_CLASSES[i].capitalize(), marker='s', linestyle='', markersize=8)
                style_legend(ax, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)
            elif "RGB" in title:
                ax.imshow(img)
                # ax.set_title(title)
            else:
                if 'NDVI' in title:
                    color_map = sns.color_palette("crest", as_cmap=True)
                    cbar_label = 'NDVI Value'
                else:
                    color_map = sns.color_palette("rocket_r", as_cmap=True)
                    cbar_label = 'Temperature (°C)'
                im = ax.imshow(img, cmap=color_map)
                # ax.set_title(title)
                cbar = fig.colorbar(im, ax=ax, shrink=0.75)
                cbar.set_label(cbar_label)
            
            ax.axis('off')
            st.pyplot(fig)

            if save_pdfs:
                output_dir = "reports/tests/app/data"
                os.makedirs(output_dir, exist_ok=True)
                filename = f"{title.replace(' ', '_').replace('/', '-')}_{st.session_state.sample_idx}.pdf"
                filepath = os.path.join(output_dir, filename)
                fig.set_facecolor('white')
                fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
                st.success(f"Saved {filepath}")

    st.header("Metadata and Embeddings")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Embedding Configuration")
        st.info(f"**Uses both embeddings:** `{additional_embeddings}`")
        st.info(f"**Uses metadata-only embedding:** `{metadata_only_embeddings}`")
        
        st.subheader("Sample Metadata")
        meta_keys = ['lat', 'lon', 'population', 'delta_time_years']
        t1_date_str = f"{int(t1_dates[0, 0].item())}-{int(t1_dates[0, 1].item()):02d}"
        t2_date_str = f"{int(t2_dates[0, 0].item())}-{int(t2_dates[0, 1].item()):02d}"
        
        meta_text = f"**T1 Date:** {t1_date_str}\n**T2 Date:** {t2_date_str}\n"
        for i, key in enumerate(meta_keys):
            meta_text += f"**{key.capitalize()}:** {meta_viz[i]:.4f}\n\n"
        st.markdown(meta_text)

    with col2:
        st.subheader("Historical Temperature Series")
        
        fig_ts, ax_ts = get_styled_figure_ax(aspect='auto', figsize=(15, 5))
        ax_ts.plot(temp_series_viz, linewidth=2)
        ax_ts.set_xlabel("Time Step")
        ax_ts.set_ylabel("Temperature Anomaly (°C)")
        # ax_ts.set_title("Historical Temperature Series") 
        st.pyplot(fig_ts)
        st.caption("Un-normalized temperature series used as input for the temporal encoder.")

        if st.button("Save Temperature Series PDF"):
            output_dir = "reports/tests/app/data"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"Temperature_Series_{st.session_state.sample_idx}.pdf"
            filepath = os.path.join(output_dir, filename)
            fig_ts.set_facecolor('white')
            fig_ts.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
            st.success(f"Saved {filepath}")
    
    st.header("Predictions vs. Ground Truth")
    target_channels = [c.replace('after_', '') for c in CONFIG.dataset.target_channels]

    for i, name in enumerate(target_channels):
        st.subheader(f"Target: {name}")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h5 style='text-align: center;'>Ground Truth</h5>", unsafe_allow_html=True)
            fig, ax = get_styled_figure_ax(aspect='none')
            if 'ndvi' in name.lower():
                cmap = sns.color_palette("crest", as_cmap=True)
                cbar_label = 'NDVI Value'
            elif 'temp' in name.lower():
                cmap = sns.color_palette("rocket_r", as_cmap=True)
                cbar_label = 'Temperature (°C)'
            im = ax.imshow(targets_viz[i], cmap=cmap)
            cbar = fig.colorbar(im, ax=ax, shrink=0.75)
            cbar.set_label(cbar_label)
            ax.axis('off')
            style_legend(ax)
            st.pyplot(fig)

            if save_pdfs:
                output_dir = "reports/tests/app/data"
                os.makedirs(output_dir, exist_ok=True)
                filename = f"Target_{name}_{st.session_state.sample_idx}.pdf"
                filepath = os.path.join(output_dir, filename)
                fig.set_facecolor('white')
                fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
                st.success(f"Saved {filepath}")

        with col2:
            st.markdown("<h5 style='text-align: center;'>Prediction</h5>", unsafe_allow_html=True)
            fig, ax = get_styled_figure_ax(aspect='none')
            im = ax.imshow(outputs_viz[i], cmap='viridis')
            fig.colorbar(im, ax=ax)
            ax.axis('off')
            style_legend(ax)
            st.pyplot(fig)
        
        # Zoomed views
        with st.expander(f"Show Zoomed Quadrant Views for '{name}'"):
            fig_zoomed = plot_zoomed_views(targets_viz[i], outputs_viz[i], name)
            st.write(f"Prediction stats for {name}: min={output_channels_min[i]:.4f}, max={output_channels_max[i]:.4f}")
            st.write(f"Un-normalized Prediction {name} stats: min={output_normalized_channels_min[i]:.4f}, max={output_normalized_channels_max[i]:.4f}")
            st.pyplot(fig_zoomed)

home()

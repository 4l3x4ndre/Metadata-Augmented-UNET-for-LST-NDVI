import streamlit as st
import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from urban_planner.config import CONFIG
from src.dataset import collate_fn
from app_dev.app_src.utils import (
    load_model,
    load_dataset,
    load_normalization_metrics,
    get_unnormalized_data,
    plot_zoomed_views_with_error,
)

def model_comparison():
    """Page for comparing multiple models."""
    st.title("Model Comparison")
    CONFIG.device = 'cpu'

    # --- Sidebar ---
    st.sidebar.header("Configuration")

    # Model Selection
    model_files = glob.glob("models/**/*.pth", recursive=True)
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        model_files,
        default=model_files[:2] if len(model_files) >=2 else model_files
    )

    if not selected_models:
        st.warning("Please select at least one model.")
        st.stop()

    # Load Models
    models = {}
    for path in selected_models:
        model_name = os.path.basename(path)
        model, checkpoint, _, _ = load_model(path)
        if model:
            models[model_name] = (model, checkpoint.get('metadata_input_length', 4))

    # --- Data Selection ---
    dataset = load_dataset()
    if not dataset:
        st.stop()
    metrics = load_normalization_metrics()

    st.sidebar.header("Data Sample Selection")
    selection_mode = st.sidebar.radio("Selection Mode", ["Index", "Cycle", "Filename"], index=1)

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
    try:
        sample = dataset[st.session_state.sample_idx]
        batch = collate_fn([sample])
        input_tensor, meta_tensor, temp_series_tensor, _, t1_dates, t2_dates, target_tensor = batch

        outputs = {}
        with torch.no_grad():
            for name, (model, metadata_input_length) in models.items():
                if metadata_input_length == 8:
                    meta_tensor = torch.cat([meta_tensor, t1_dates, t2_dates], dim=1)
                output_tensor = model(input_tensor, temp_series_tensor, meta_tensor)
                outputs[name] = output_tensor[0].cpu().numpy()

        input_np = input_tensor[0].cpu().numpy()
        meta_np = meta_tensor[0].cpu().numpy()
        temp_series_np = temp_series_tensor[0].cpu().numpy()
        target_np = target_tensor[0].cpu().numpy()

        first_model_output = next(iter(outputs.values()))
        inputs_viz, meta_viz, temp_series_viz, targets_viz, _ = get_unnormalized_data(
            input_np, meta_np, temp_series_np, target_np, first_model_output, metrics
        )

        outputs_viz = {}
        for name, output_np in outputs.items():
             _, _, _, _, unnorm_output = get_unnormalized_data(
                input_np, meta_np, temp_series_np, target_np, output_np, metrics
            )
             outputs_viz[name] = unnorm_output

    except Exception as e:
        st.error(f"Failed to process sample {st.session_state.sample_idx}: {e}")
        logger.exception(e)
        st.stop()

    # --- Visualization ---
    st.header("Model Inputs")
    st.write(f"Showing sample `{os.path.basename(dataset.file_list[st.session_state.sample_idx])}`")
    cols = st.columns(len(inputs_viz))
    for i, (title, img) in enumerate(inputs_viz.items()):
        with cols[i]:
            st.subheader(title)
            if "DW" in title:
                st.image(img, caption=f"{title} (Categorical)", width='stretch')
            elif "RGB" in title:
                st.image(img, caption=title, width='stretch')
            else:
                fig, ax = plt.subplots()
                im = ax.imshow(img, cmap='viridis')
                ax.set_title(title)
                ax.axis('off')
                fig.colorbar(im, ax=ax)
                st.pyplot(fig)

    st.header("Predictions vs. Ground Truth")
    target_channels = [c.replace('after_', '') for c in CONFIG.dataset.target_channels]

    for i, name in enumerate(target_channels):
        st.subheader(f"Target: {name}")

        if not outputs_viz:
            st.warning("No model outputs to display.")
            continue

        model_tabs = st.tabs(list(outputs_viz.keys()))

        for j, model_name in enumerate(outputs_viz.keys()):
            with model_tabs[j]:
                cols = st.columns(3) # GT, Pred, Error
                
                gt = targets_viz[i]
                pred = outputs_viz[model_name][i]
                error = pred - gt
                err_max_abs = np.max(np.abs(error)) if np.max(np.abs(error)) > 0 else 1
                
                vmin = np.min([gt.min(), pred.min()])
                vmax = np.max([gt.max(), pred.max()])

                with cols[0]:
                    st.markdown("<h5 style='text-align: center;'>Ground Truth</h5>", unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    im = ax.imshow(gt, cmap='viridis', vmin=vmin, vmax=vmax)
                    plt.colorbar(im, ax=ax)
                    ax.axis('off')
                    st.pyplot(fig)
                
                with cols[1]:
                    st.markdown(f"<h5 style='text-align: center;'>Prediction</h5>", unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    im = ax.imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
                    plt.colorbar(im, ax=ax)
                    ax.axis('off')
                    st.pyplot(fig)

                with cols[2]:
                    st.markdown(f"<h5 style='text-align: center;'>Error Map</h5>", unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    im = ax.imshow(error, cmap='coolwarm', vmin=-err_max_abs, vmax=err_max_abs)
                    plt.colorbar(im, ax=ax)
                    ax.axis('off')
                    st.pyplot(fig)
        
        # Zoomed views
        with st.expander(f"Show Zoomed Quadrant Views for '{name}'"):
            if not outputs_viz:
                st.warning("No model outputs to display.")
            else:
                zoom_model_tabs = st.tabs(list(outputs_viz.keys()))
                for j, model_name in enumerate(outputs_viz.keys()):
                    with zoom_model_tabs[j]:
                        fig_zoomed = plot_zoomed_views_with_error(targets_viz[i], outputs_viz[model_name][i], name)
                        st.pyplot(fig_zoomed)

model_comparison()
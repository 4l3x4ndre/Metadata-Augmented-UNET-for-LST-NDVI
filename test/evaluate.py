from loguru import logger
import wandb
import os
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd
from typer import Typer
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from scipy.ndimage import laplace
import matplotlib.pyplot as plt
import json
import matplotlib.patches as mpatches

from src.dataset import create_dataloader
from urban_planner.config import CONFIG
from src.model import UrbanPredictor
from src.utils.visualization import dw_to_rgb, get_dw_legend_patches, DW_CLASSES

app = Typer()

def get_unnormalized_data(targets, outputs, metrics):
    """Un-normalizes temperature data for visualization and metric calculation."""
    if not metrics:
        logger.warning("Normalization metrics not found. Using raw data.")
        return targets, outputs

    unnorm_targets = np.zeros_like(targets)
    unnorm_outputs = np.zeros_like(outputs)
    target_channel_names = CONFIG.dataset.target_channels

    for i, ch in enumerate(target_channel_names):
        if 'temp' in ch.lower():
            unnorm_targets[:, i] = targets[:, i] * metrics['temp_std'] + metrics['temp_mean']
            unnorm_outputs[:, i] = outputs[:, i] * metrics['temp_std'] + metrics['temp_mean']
        else:  # NDVI [-1, 1] range channel is not normalized
            unnorm_targets[:, i] = targets[:, i]
            unnorm_outputs[:, i] = outputs[:, i]
            
    return unnorm_targets, unnorm_outputs


@app.command()
def main(
    checkpoint_path: str,
    device: str = '',
    wandblog: bool = True,
    study_name: str = "test",
    n_visualize: int = 10,
    jobid: str = ""
):
    # --- Directory Setup ---
    os.makedirs("reports/tests/visualizations", exist_ok=True)

    # --- Torch config ---
    if device:
        if device.lower() == 'gpu' and torch.cuda.is_available():
            CONFIG.device = "cuda:0"
        else:
            CONFIG.device = device
            
    torch.manual_seed(CONFIG.seed)
    logger.info(f"🧠 Using device {CONFIG.device}")

    # --- Get Training Cities ---
    train_dir = os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'train')
    if os.path.isdir(train_dir):
        train_files = os.listdir(train_dir)
        train_cities = set()
        for filename in train_files:
            if filename.endswith('.npz'):
                parts = filename.split('_')
                city = " ".join(parts[:-8])
                train_cities.add(city)
        logger.info(f"Found {len(train_cities)} unique cities in the training set.")
    else:
        logger.warning(f"Training directory not found at {train_dir}. Cannot determine known/unknown cities.")
        train_cities = set()

    # --- Load Checkpoint and Config ---
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG.device)
    
    hyperparams = checkpoint.get('hyperparameters', {})
    cfg = CONFIG.training
    cfg.batch_size = hyperparams.get('batch_size', 16)
    
    model_type = checkpoint.get('model_type', 'unet')
    
    # Resolve embedding flags (Backwards compatibility)
    if 'temporal_embeddings' in hyperparams:
        temporal_embeddings = hyperparams['temporal_embeddings']
        metadata_embeddings = hyperparams['metadata_embeddings']
    else:
        # Legacy fallback
        default_emb = True
        checkpoint_study = checkpoint.get('study_name', '')
        # Check if 'noemb' is in study_name (arg) or checkpoint study name
        if 'noemb' in study_name or 'noemb' in checkpoint_study:
            default_emb = False
            
        additional_embeddings = checkpoint.get('additional_embeddings', default_emb)
        metadata_only_embeddings = checkpoint.get('metadata_only_embeddings', False)
        if additional_embeddings:
            temporal_embeddings = True
            metadata_embeddings = True
        elif metadata_only_embeddings:
            temporal_embeddings = False
            metadata_embeddings = True
        else:
            temporal_embeddings = False
            metadata_embeddings = False

    trial_id = checkpoint.get('trial_id', 'unknown')

    logger.info(f"Loaded model from trial {trial_id} of study '{checkpoint.get('study_name', 'N/A')}'")
    logger.info(f"Model Config: temporal_embeddings={temporal_embeddings}, metadata_embeddings={metadata_embeddings}")

    if temporal_embeddings and metadata_embeddings:
        tag_emb = "emb"
    elif temporal_embeddings:
        tag_emb = "tempemb"
    elif metadata_embeddings:
        tag_emb = "metaemb"
    else:
        tag_emb = "noemb"

    # --- WandB Initialization ---
    if wandblog:
        wandb_name = f"eval_{study_name}_trial_{trial_id}_{jobid}"
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            config=OmegaConf.to_container(CONFIG, resolve=True),
            name=wandb_name,
            group=study_name,
            reinit=True,
            tags=["evaluation", "test", study_name, model_type, tag_emb,
                  f"target_{'_'.join(CONFIG.dataset.target_channels)}"],
        )
        wandb.config.update({"checkpoint_path": checkpoint_path})

    # --- Load Data and Model ---
    test_loader = create_dataloader(
        split='test',
        dataset_type=CONFIG.dataset.dataset_type,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    metadata_input_length = checkpoint.get('metadata_input_length', 4)
    model = UrbanPredictor(
        model_type=model_type,
        spatial_channels=CONFIG.dataset.nb_input_channels,
        seq_len=CONFIG.dataset.temporal_length,
        temporal_dim=hyperparams.get('temporal_dim', 16),
        meta_features=metadata_input_length,
        meta_dim=hyperparams.get('meta_dim', 8),
        lstm_dim=hyperparams.get('lstm_hidden', 32),
        out_channels=len(CONFIG.dataset.target_channels),
        temporal_embeddings=temporal_embeddings,
        metadata_embeddings=metadata_embeddings
    ).to(CONFIG.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Load Normalization Metrics ---
    metrics_path = os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'normalization_metrics.json')
    if not os.path.exists(metrics_path):
        metrics = None
    else:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    # --- Evaluation Loop ---
    results = []
    created_visuals = 0
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_stack, metadata, temp_series, _, t1_dates, t2_dates, targets = batch
            if metadata_input_length == 8:
                metadata = torch.cat([metadata, t1_dates, t2_dates], dim=1)
            outputs = model(input_stack, temp_series, metadata)

            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Debug NaN or unique values
            if np.isnan(outputs_np).any():
                logger.error(f"NaN values found in outputs at sample index {sample_idx}")
            if np.isnan(targets_np).any():
                logger.error(f"NaN values found in targets at sample index {sample_idx}")
            if np.unique(outputs_np).size == 1:
                logger.warning(f"Outputs have a single unique value at sample index {sample_idx}")
            if np.unique(targets_np).size == 1:
                logger.warning(f"Targets have a single unique value at sample index {sample_idx}")
            
            # Un-normalize for interpretable metrics
            targets_unnorm, outputs_unnorm = get_unnormalized_data(targets_np, outputs_np, metrics)

            for _arr,_name in [(targets_unnorm, "Targets"), (outputs_unnorm, "Outputs")]:
                if np.isnan(_arr).any():
                    logger.error(f"NaN values found in { _name } at sample index {sample_idx}")
                if np.unique(_arr).size == 1:
                    logger.warning(f"{ _name } have a single unique value at sample index {sample_idx}")

            for i in range(outputs_np.shape[0]):
                # dw_map = np.argmax(input_stack[i, :9].cpu().numpy(), axis=0)
                dw_map = np.argmax(
                    np.stack(
                        [input_stack[i, _c].cpu().numpy()*_c for _c in range(9)] # channel x 0 for the first one is fine as argmax
                    ), 
                    axis=0
                )

                sample_info = test_loader.dataset.get_metadata_from_idx(sample_idx)
                is_known_city = sample_info['city'] in train_cities
                
                for ch_idx, ch_name in enumerate(CONFIG.dataset.target_channels):
                    pred = outputs_unnorm[i, ch_idx]
                    gt = targets_unnorm[i, ch_idx]

                    for _arr,_name in [(pred, "Prediction"), (gt, "Ground Truth")]:
                        if np.isnan(_arr).any():
                            logger.error(f"NaN values found in { _name } for channel {ch_name} at sample index {sample_idx}")
                        if np.unique(_arr).size == 1:
                            logger.warning(f"{ _name } have a single unique value for channel {ch_name} at sample index {sample_idx}")
                    
                    t1_year = int(t1_dates[i, 0].item())
                    t1_month = int(t1_dates[i, 1].item())
                    t2_year = int(t2_dates[i, 0].item())
                    t2_month = int(t2_dates[i, 1].item())
                    time_delta = t2_year - t1_year

                    # Overall sample metrics
                    sample_mae = np.mean(np.abs(pred - gt))
                    sample_rmse = np.sqrt(np.mean((pred - gt) ** 2))
                    sample_laplacian_var_pred = np.var(laplace(pred))
                    sample_laplacian_var_gt = np.var(laplace(gt))

                    results.append({
                        "sample_idx": sample_idx, "channel": ch_name, "dw_class": "overall",
                        "mae": sample_mae, "rmse": sample_rmse,
                        "laplacian_var_pred": sample_laplacian_var_pred,
                        "laplacian_var_gt": sample_laplacian_var_gt,
                        "is_known_city": is_known_city,
                        "t1_year": t1_year,
                        "t1_month": t1_month,
                        "t2_year": t2_year,
                        "t2_month": t2_month,
                        "time_delta": time_delta,
                        **sample_info
                    })

                    # Per-class metrics
                    for dw_code, dw_name in DW_CLASSES.items():
                        mask = (dw_map == dw_code)
                        if np.any(mask):
                            mae = np.mean(np.abs(pred[mask] - gt[mask]))
                            rmse = np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2))
                            results.append({
                                "sample_idx": sample_idx, "channel": ch_name, "dw_class": dw_name,
                                "mae": mae, "rmse": rmse,
                                "laplacian_var_pred": None, "laplacian_var_gt": None,
                                "is_known_city": is_known_city,
                                "t1_year": t1_year,
                                "t1_month": t1_month,
                                "t2_year": t2_year,
                                "t2_month": t2_month,
                                "time_delta": time_delta,
                                **sample_info
                            })
                
                # Create visualizations for the first n_visualize samples
                if created_visuals < n_visualize:
                    plot_evaluation_results(
                        input_stack[i].cpu().numpy(),
                        targets_unnorm[i],
                        outputs_unnorm[i],
                        pd.DataFrame([r for r in results if r['sample_idx'] == sample_idx]),
                        study_name,
                        trial_id,
                        sample_idx,
                        wandblog,
                        metrics,
                        sample_info
                    )
                    created_visuals += 1
                
                sample_idx += 1

    # --- Aggregate and Save Results ---
    df = pd.DataFrame(results)
    report_path = f"reports/tests/{study_name}_{model_type}_{tag_emb}_{trial_id}_job{jobid}_evaluation.csv"
    df.to_csv(report_path, index=False)
    logger.success(f"Full evaluation report saved to {report_path}")

    # --- Save Information File ---
    info_path = f"reports/tests/{study_name}_{model_type}_{tag_emb}_{trial_id}_job{jobid}_info.csv"
    info_data = {
        "evaluation_csv_path": report_path,
        "model_embedding_type": tag_emb,
        "study_name": study_name,
        "trial_id": trial_id,
        "model_architecture": model_type
    }
    pd.DataFrame([info_data]).to_csv(info_path, index=False)
    logger.success(f"Evaluation info saved to {info_path}")

    # --- Log to Console and WandB ---
    summary = df.groupby(['is_known_city', 't1_year', 'channel', 'dw_class', 'city', 'lat', 'lon'])[['mae', 'rmse', 'laplacian_var_pred', 'laplacian_var_gt']].mean().reset_index()
    
    logger.info("--- Evaluation Summary ---")

    if not summary[summary['is_known_city'] == True].empty:
        logger.info("--- Known Cities (seen in training) ---")
        print(summary[summary['is_known_city'] == True].to_string())

    if not summary[summary['is_known_city'] == False].empty:
        logger.info("--- Unknown Cities (not seen in training) ---")
        print(summary[summary['is_known_city'] == False].to_string())

    if wandblog:
        # Log summary table for known cities
        known_summary = summary[summary['is_known_city'] == True]
        overall_summary_known = known_summary[known_summary['dw_class'] == 'overall']
        if not overall_summary_known.empty:
            wandb.log({"summary/overall_metrics_known": wandb.Table(dataframe=overall_summary_known)})
        
        per_class_summary_known = known_summary[known_summary['dw_class'] != 'overall']
        if not per_class_summary_known.empty:
            wandb.log({"summary/per_class_metrics_known": wandb.Table(dataframe=per_class_summary_known)})

        # Log summary table for unknown cities
        unknown_summary = summary[summary['is_known_city'] == False]
        overall_summary_unknown = unknown_summary[unknown_summary['dw_class'] == 'overall']
        if not overall_summary_unknown.empty:
            wandb.log({"summary/overall_metrics_unknown": wandb.Table(dataframe=overall_summary_unknown)})

        per_class_summary_unknown = unknown_summary[unknown_summary['dw_class'] != 'overall']
        if not per_class_summary_unknown.empty:
            wandb.log({"summary/per_class_metrics_unknown": wandb.Table(dataframe=per_class_summary_unknown)})
        
        # Log aggregated scalar values for easier plotting in wandb
        for _, row in summary.iterrows():
            known_str = "known" if row['is_known_city'] else "unknown"
            metric_prefix = f"metrics/{known_str}/{row['channel']}/{row['dw_class']}"
            wandb.log({
                f"{metric_prefix}/mae": row['mae'],
                f"{metric_prefix}/rmse": row['rmse'],
            })
            if row['dw_class'] == 'overall':
                 wandb.log({
                    f"{metric_prefix}/laplacian_var_pred": row['laplacian_var_pred'],
                    f"{metric_prefix}/laplacian_var_gt": row['laplacian_var_gt'],
                })

        run.finish()

def plot_evaluation_results(input_np, gt_unnorm, pred_unnorm, metrics_df, study_name, trial_id, sample_idx, wandblog, metrics, sample_info):
    """Generates and saves a detailed visualization for a single sample."""
    
    # --- Inputs ---
    dw_t1_map = np.argmax(input_np[:9], axis=0)
    dw_t1_rgb = dw_to_rgb(dw_t1_map, return_numpy=True)

    if metrics and 'rgb_mean' in metrics and 'rgb_std' in metrics:
        rgb_normalized = input_np[9:12]
        rgb_t1 = (rgb_normalized * np.array(metrics['rgb_std'])[:, np.newaxis, np.newaxis] + np.array(metrics['rgb_mean'])[:, np.newaxis, np.newaxis]) * 255.0
        rgb_t1 = np.clip(rgb_t1.transpose(1, 2, 0), 0, 255).astype(np.uint8)
    else:
        rgb_t1 = np.clip(input_np[9:12].transpose(1, 2, 0), 0, 1)
    
    # --- Plotting ---
    city = sample_info["city"]
    lat, lon = sample_info['lat'], sample_info['lon']
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(f"Evaluation - {city} ({lat}, {lon})\nSample {sample_idx} (Trial {trial_id})", fontsize=20)
    
    gs = fig.add_gridspec(3, len(CONFIG.dataset.target_channels) * 2)

    # --- Input Plots ---
    ax_dw = fig.add_subplot(gs[0, 0])
    ax_dw.imshow(dw_t1_rgb)
    ax_dw.set_title("Input DW (t1)")
    ax_dw.axis('off')
    ax_dw.legend(handles=get_dw_legend_patches(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_rgb.imshow(rgb_t1)
    ax_rgb.set_title("Input RGB (t1)")
    ax_rgb.axis('off')

    # --- GT, Pred, and Error Plots per Channel ---
    for i, ch_name in enumerate(CONFIG.dataset.target_channels):
        gt = gt_unnorm[i]
        pred = pred_unnorm[i]
        error = pred - gt
        
        vmin, vmax = np.min([gt.min(), pred.min()]), np.max([gt.max(), pred.max()])
        err_max_abs = np.max(np.abs(error))

        # GT
        ax_gt = fig.add_subplot(gs[1, i*2])
        im = ax_gt.imshow(gt, cmap='viridis', vmin=vmin, vmax=vmax)
        ax_gt.set_title(f"GT: {ch_name}")
        plt.colorbar(im, ax=ax_gt, orientation='horizontal', pad=0.05)
        ax_gt.axis('off')

        # Prediction
        ax_pred = fig.add_subplot(gs[1, i*2+1])
        im = ax_pred.imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"Pred: {ch_name}")
        plt.colorbar(im, ax=ax_pred, orientation='horizontal', pad=0.05)
        ax_pred.axis('off')

        # Error Map
        ax_err = fig.add_subplot(gs[2, i*2])
        im = ax_err.imshow(error, cmap='coolwarm', vmin=-err_max_abs, vmax=err_max_abs)
        ax_err.set_title("Error (Pred - GT)")
        plt.colorbar(im, ax=ax_err, orientation='horizontal', pad=0.05)
        ax_err.axis('off')

        # Per-class MAE bar chart
        ax_bar = fig.add_subplot(gs[2, i*2+1])
        class_metrics = metrics_df[(metrics_df['channel'] == ch_name) & (metrics_df['dw_class'] != 'overall')]
        class_metrics = class_metrics.set_index('dw_class')
        class_metrics['mae'].plot(kind='bar', ax=ax_bar)
        ax_bar.set_title(f"MAE per DW Class")
        ax_bar.set_ylabel("MAE")
        ax_bar.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    img_filename = f"{study_name}_trial_{trial_id}_sample_{city}_{sample_idx}.png"
    img_path = os.path.join("reports/tests/visualizations", img_filename)
    plt.savefig(img_path, dpi=150)
    plt.close(fig)
    
    if wandblog:
        wandb.log({f"visualizations/sample_{sample_idx}": wandb.Image(img_path)})

if __name__ == "__main__":
    app()

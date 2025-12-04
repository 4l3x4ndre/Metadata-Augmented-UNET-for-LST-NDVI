from loguru import logger
import wandb
import os
from torch.optim import SGD, Adam, AdamW
import torch
from omegaconf import OmegaConf
import optuna
from typing import Callable
from typer import Typer

from src.dataset import create_dataloader, RandomFlip
from urban_planner.config import CONFIG
from src.model import UrbanPredictor
from src.utils.metrics import RunningLoss
from src.utils.losses import compute_loss_mse, compute_loss_mse_gradient, compute_loss_l1_grad_ssim, compute_all_loss
from src.utils.visualize_predictions import plot_predictions_vs_targets

app = Typer()

def validate(
    model:torch.nn.Module, 
    loader:torch.utils.data.DataLoader,
    criterion:Callable
):
    """
    Computes the loss on the validation set.
    """
    model.eval()
    total_loss = 0
    l_dict = {}
    num_samples = 0
    with torch.no_grad():
        for input_stack, metadata, temp_series, temp_series_lengths, t1_dates, t2_dates, targets in loader:
            metadata_full = torch.cat([metadata, t1_dates, t2_dates], dim=1)
            outputs = model(input_stack, temp_series, metadata_full)
            try:
                # We weight the batch loss by the number of samples in it
                # batch_loss,_ = compute_loss(outputs, targets, criterion=criterion)
                loss_object = criterion(outputs, targets)
                batch_loss = loss_object['total']
                if batch_loss is not None:
                    total_loss += batch_loss.item() * len(input_stack)
                    num_samples += len(input_stack)
                l_dict_batch = compute_all_loss(outputs, targets)
                for k, v in l_dict_batch.items():
                    if k not in l_dict:
                        l_dict[k] = 0.0
                    l_dict[k] += v.detach().cpu().item() * len(input_stack)
            except ValueError as e:
                logger.warning(f"Skipping batch in validation due to error: {e}")
                continue

    model.train() # Set model back to training mode
    
    # Return average loss
    if num_samples == 0:
        logger.warning("Validation loader was empty or all batches failed.")
        return float('inf'), {}
        
    return total_loss / num_samples, {k: v / num_samples for k, v in l_dict.items()}

@app.command()
def main(
        device:str='',
        wandblog:bool=True, 
        n_trials:int=50, 
        force_study_name:bool=False,
        temporal_embeddings:bool=True,
        metadata_embeddings:bool=True,
        study_name:str="urban-predictor",
        model_type:str="unet++",
        jobid:str="",
):
    """
    Params:
    additional_embeddings: If True, use temperature serie AND metadata embeddings.
    """
    assert model_type in ['unet', 'unet++'], "model_type must be 'unet' or 'unet++'"
    if not force_study_name:
        if temporal_embeddings and metadata_embeddings:
            study_name += "-emb"
        elif temporal_embeddings:
            study_name += "-tempemb"
        elif metadata_embeddings:
            study_name += "-metaemb"
        else:
            study_name += "-noemb"

    # --- Torch config ---
    if device != '':
        cuda_available = torch.cuda.is_available()
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        logger.info(f"CUDA available: {cuda_available}")
        logger.info(f"Number of GPUs: {num_gpus}")
        logger.info(f"GPU names: {gpu_names}")

        if cuda_available:
            CONFIG.device = "cuda:0" if device.lower() == 'gpu' else 'cpu'
            _device_name = torch.cuda.get_device_name(0) if device.lower() == 'gpu' else 'CPU'
        else:
            CONFIG.device = 'cpu'
            _device_name = 'CPU'
            logger.warning("CUDA is not available. Using CPU.")
    else:
        _device_name = torch.cuda.get_device_name(0) if 'cuda' in CONFIG.device else 'CPU'
    # --------------------
    logger.info(f"🧠 Using device {CONFIG.device} (device name: {_device_name})")
    torch.manual_seed(CONFIG.seed)

    if wandblog:
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)

    os.makedirs(CONFIG.MODELS_DIR, exist_ok=True)
    os.makedirs("reports/training_optuna", exist_ok=True)
    os.makedirs("reports/training/visualizations", exist_ok=True)


    def objective(
        trial: optuna.trial.Trial,
    ) -> float:

        if wandblog:
            targets_str = ""
            for i, tc in enumerate(CONFIG.dataset.target_channels):
                targets_str += tc.split('_')[-1]
                if i != len(CONFIG.dataset.target_channels) - 1:
                    targets_str += "-"
            tag_emb = "noemb"
            if temporal_embeddings and metadata_embeddings:
                targets_str += "-emb"
                tag_emb = "emb"
            elif temporal_embeddings:
                targets_str += "-tempemb"
                tag_emb = "tempemb"
            elif metadata_embeddings:
                targets_str += "-metaemb"
                tag_emb = "metaemb"
            if '++' in model_type:
                targets_str += "++"
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT"), 
                config=OmegaConf.to_container(CONFIG, resolve=True),
                group=study_name,
                name=f"trial-{trial.number}-{targets_str}-{jobid}",
                reinit=True,
                tags=[study_name, model_type, tag_emb,
                      f"loss_{CONFIG.training.loss}",
                      f"target_{'_'.join(CONFIG.dataset.target_channels)}",
                      "metadata_full"
                      ]
            )

        cfg = CONFIG.training
        hyperparams = {
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "weight_decay": cfg.weight_decay,
            "temporal_dim": cfg.temporal_dim,
            "meta_dim": cfg.meta_dim,
            "lstm_hidden": cfg.lstm_hidden,
            "model_type": model_type,
            "target_channels": ",".join(CONFIG.dataset.target_channels),
            "input_channels": ",".join(CONFIG.dataset.input_channels),
            "temporal_embeddings": temporal_embeddings,
            "metadata_embeddings": metadata_embeddings,
        }
        if wandblog:
            wandb.config.update(hyperparams)

        train_loader = create_dataloader(
            split='train',
            dataset_type=CONFIG.dataset.dataset_type,
            transform=RandomFlip(),
            batch_size=cfg.batch_size,
            shuffle=True,
            recompute_splits_indices=False,
            seed=1,
            num_workers=0
        )

        val_loader = create_dataloader(
            split='val',
            dataset_type=CONFIG.dataset.dataset_type,
            transform=None,
            batch_size=cfg.batch_size,
            shuffle=False,
            recompute_splits_indices=False,
            seed=1,
            num_workers=0
        )

        model = UrbanPredictor(
            model_type=model_type,
            spatial_channels=CONFIG.dataset.nb_input_channels,
            seq_len=CONFIG.dataset.temporal_length,
            temporal_dim=cfg.temporal_dim,
            meta_features=CONFIG.dataset.nb_metadata_features,
            meta_dim=cfg.meta_dim,
            lstm_dim=cfg.lstm_hidden,
            out_channels=len(CONFIG.dataset.target_channels),
            deep_supervision=False,
            temporal_embeddings=temporal_embeddings,
            metadata_embeddings=metadata_embeddings,
        ).to(CONFIG.device)
        model.train()
         
        if cfg.optimizer == 'SGD':
            optimizer = SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
        elif cfg.optimizer == 'Adam':
            optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'AdamW':
            optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented.")

        if cfg.loss == 'mse':
            criterion = compute_loss_mse
        elif cfg.loss == 'mse-gradient':
            criterion = compute_loss_mse_gradient
        elif cfg.loss == 'l1-gradient-ssim':
            criterion = compute_loss_l1_grad_ssim
        else:
            raise NotImplementedError(f"Loss {cfg.loss} not implemented.")

        # --- Training & Validation Loop ---
        step = 0
        best_val_loss = float('inf')
        ema_loss = RunningLoss(mode='ema', ema_alpha=0.98)
        sma_loss = RunningLoss(mode='sma', window_size=50)
        cum_loss = RunningLoss(mode='cumulative')
        
        logger.info(f"Starting training for {cfg.epochs} epochs...")
        for epoch in range(cfg.epochs):
            model.train()
            epoch_train_loss = 0
            num_train_samples = 0
            cum_loss.reset()
            ema_loss.reset()
            sma_loss.reset()
            
            for input_stack, metadata, temp_series, temp_series_lengths, t1_dates, t2_dates, targets in train_loader:
                metadata_full = torch.cat([metadata, t1_dates, t2_dates], dim=1)
                outputs = model(input_stack, temp_series, metadata_full)

                losses = criterion(outputs, targets)
                # 'total' is the key for the main loss value
                batch_loss = losses.get('total', None) 

                if batch_loss is not None:
                    batch_loss.backward()
                    if cfg.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    bloss_item = batch_loss.detach().cpu().item()
                    epoch_train_loss += bloss_item * len(input_stack)
                    num_train_samples += len(input_stack)

                    ema_val = ema_loss.update(bloss_item)
                    sma_val = sma_loss.update(bloss_item)
                    cum_val = cum_loss.update(bloss_item, n=len(input_stack)) 

                    if wandblog and step % CONFIG.logging.frequency_log == 0:
                        l_dict = {
                            "train/batch_loss": bloss_item,
                            "train/ema_loss": ema_val,
                            "train/sma_loss": sma_val,
                            "train/cum_loss": cum_val,
                            "train/step": step,
                            "train/epoch": epoch,
                        }
                        for key, value in losses.items():
                            l_dict[f"train/loss_{key}"] = value.detach().cpu().item()
                        wandb.log(l_dict, step=step)

                    if step % CONFIG.logging.frequency_plt == 0:
                        plot_predictions_vs_targets(input_stack, metadata, temp_series, t1_dates, t2_dates, outputs, targets, study_name, trial.number, step, bloss_item)

                
                step += 1
            
            # --- Epoch End: Validation and Checkpointing ---
            avg_epoch_train_loss = epoch_train_loss / num_train_samples if num_train_samples > 0 else float('inf')
            val_loss, val_loss_dict = validate(model, val_loader, criterion)
            
            logger.info(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {avg_epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if wandblog:
                log_dict = {
                    "val/loss": val_loss,
                    "val/epoch": epoch,
                    "train/epoch_loss": avg_epoch_train_loss,
                }
                for key, value in val_loss_dict.items():
                    if not key in log_dict:
                        log_dict[f"val/loss_{key}"] = value
                wandb.log(log_dict, step=step)

            # Save the model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                    "hyperparameters": hyperparams,
                    "model_type": model_type,
                    "study_name": study_name,
                    "trial_id": trial.number,
                    'metadata_input_length': CONFIG.dataset.nb_metadata_features
                }
                model_name = f"{study_name}_trial_{trial.number}_best_job{jobid}.pth"
                model_filename = os.path.join(CONFIG.MODELS_DIR, model_name)
                torch.save(checkpoint, model_filename)
                logger.success(f"New best model saved with val_loss: {best_val_loss:.4f}")

            # --- Optuna Pruning ---
            trial.report(val_loss, epoch)
            if trial.should_prune():
                if wandblog: wandb.finish()
                raise optuna.exceptions.TrialPruned()

        if wandblog:
            run.finish()
            
        return best_val_loss

    # --- Run Optuna Study ---
    storage_name = f"sqlite:///reports/training_optuna/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        storage=storage_name,
        load_if_exists=True
    )

    # Re-evaluate failed trials
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.FAIL:
            study.enqueue_trial(trial.params)
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Study finished. Best trial: {study.best_trial.number}")
    logger.info(f"  Value (min val_loss): {study.best_trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")



if __name__ == "__main__":
    app()

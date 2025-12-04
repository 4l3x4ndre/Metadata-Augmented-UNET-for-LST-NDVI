import torch
from src.model import UrbanPredictor
import os
import logging
from urban_planner.config import CONFIG

logger = logging.getLogger(__name__)

import torch
from src.model import UrbanPredictor
import os
import logging

logger = logging.getLogger(__name__)

def load_model(model_path, device='cpu'):
    logger.info(f"Loading model from {model_path}")
    
    # Load checkpoint first to inspect config
    checkpoint = torch.load(model_path, map_location=device)
    
    # Remove unused keys to save disk space
    if isinstance(checkpoint, dict):
        keys_to_remove = ['optimizer_state_dict', 'scheduler_state_dict', 'optimizer', 'loss', 'epoch', 'step']
        dirty = False
        for key in keys_to_remove:
            if key in checkpoint:
                del checkpoint[key]
                dirty = True
        
        if dirty:
            logger.info(f"Cleaning model checkpoint: removing unused keys from {model_path}")
            try:
                torch.save(checkpoint, model_path)
            except Exception as e:
                logger.error(f"Failed to save cleaned model: {e}")

    hyperparams = checkpoint.get('hyperparameters', {})
    model_type = checkpoint.get('model_type', 'unet')
    
    # Logic adapted from test/evaluate.py to determine embeddings
    if 'temporal_embeddings' in hyperparams:
        temporal_embeddings = hyperparams['temporal_embeddings']
        metadata_embeddings = hyperparams['metadata_embeddings']
    else:
        # Legacy fallback
        default_emb = True
        checkpoint_study = checkpoint.get('study_name', '')
        
        if 'noemb' in checkpoint_study:
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
    
    logger.info(f"Determined Config: Type={model_type}, TempEmb={temporal_embeddings}, MetaEmb={metadata_embeddings}")

    # Hyperparams (Default based on code/paper)
    spatial_channels = 23 # 9+3+1+1+9
    seq_len = 10 
    temporal_dim = hyperparams.get('temporal_dim', 64)
    meta_features = checkpoint.get('metadata_input_length', 4)
    meta_dim = hyperparams.get('meta_dim', 64)
    lstm_dim = hyperparams.get('lstm_hidden', 96)
    out_channels = 2 # NDVI, Temp
    
    model = UrbanPredictor(
        model_type=model_type,
        spatial_channels=spatial_channels,
        seq_len=seq_len,
        temporal_dim=temporal_dim,
        meta_features=meta_features,
        meta_dim=meta_dim,
        lstm_dim=lstm_dim,
        out_channels=out_channels,
        temporal_embeddings=temporal_embeddings,
        metadata_embeddings=metadata_embeddings
    )
    
    # Load weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def run_inference(model, input_tensor, meta_tensor, temp_series_tensor, device='cpu'):
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        meta_tensor = meta_tensor.to(device)
        temp_series_tensor = temp_series_tensor.to(device)
        
        output = model(input_tensor, temp_series_tensor, meta_tensor)
        return output.cpu().numpy()

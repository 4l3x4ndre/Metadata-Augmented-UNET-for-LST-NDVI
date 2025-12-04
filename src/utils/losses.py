import torch 
import torch.nn.functional as F
from piq import ssim

def gradient_loss(pred, target) -> dict[str, torch.Tensor]:
    """
    The gradient loss measures how the spatial derivatives differ between predicted and target maps.
    Goal: "Are the slopes, edges, and transitions between neighboring pixels similar?"
    If the model predicts a map that’s spatially smooth where it should be rough 
    (or vice versa), the gradient difference will be large.
    """
    # difference between each pixel and the one directly below it (y direction)
    # and the one directly to the right of it (x direction)
    dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    
    dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    # For every pixel, compute how different the predicted and target gradients are in both directions.
    # Average all those differences into one scalar loss.
    dy_loss = torch.mean(torch.abs(dy_pred - dy_target))
    dx_loss = torch.mean(torch.abs(dx_pred - dx_target))

    return {'gradient':dy_loss + dx_loss}

def compute_loss_mse(outputs, targets) -> dict[str, torch.Tensor]:
    """
    Combined loss: MSE + gradient loss
    outputs, targets: (B, C, H, W)
    """
    # MSE loss
    mse_loss = F.mse_loss(outputs, targets)
    total_loss = mse_loss 

    return {
        'total': total_loss,
        'mse': mse_loss,
    }

def compute_loss_mse_gradient(outputs, targets, lambda_grad=0.1) -> dict[str, torch.Tensor]:
    """
    Combined loss: MSE + gradient loss
    outputs, targets: (B, C, H, W)
    """
    # MSE loss
    mse_loss = compute_loss_mse(outputs, targets)['mse']
    gradient_loss_value = gradient_loss(outputs, targets)['gradient']

    # Weighted combination
    total_loss = mse_loss + lambda_grad * gradient_loss_value

    return  {
        'total': total_loss,
        'mse': mse_loss,
        'gradient': gradient_loss_value
    }

def compute_loss_l1_grad_ssim(outputs, targets, lambda_grad=0.1, lambda_ssim=0.5) -> dict[str, torch.Tensor]:
    """
    Combined loss: L1 + gradient + SSIM
    outputs, targets: (B, C, H, W)
    lambda_grad: weight for gradient loss
    lambda_ssim: weight for SSIM loss
    """
    # Pixelwise loss
    pixel_loss:torch.Tensor = F.l1_loss(outputs, targets)

    # Gradient loss
    grad_loss:torch.Tensor = gradient_loss(outputs, targets)['gradient']

    # clamp temperature channel:
    targets_scaled = torch.stack([
        (targets[:, 0] + 1.0) / 2.0,                  # NDVI
        torch.clamp((targets[:, 1]), 0.0, 1.0)  # Temperature
    ], dim=1)

    # Clamp the temperature channel to [0, 1]
    outputs_scaled = torch.stack(
            [
                (outputs[:, 0] + 1.0) / 2.0,                  # NDVI
                torch.clamp((outputs[:, 1]), 0.0, 1.0)  # Temperature
            ], dim=1
    )

    # SSIM loss per channel, averaged over channels
    # ssim expects (B, C, H, W)
    ssim_vals = ssim(outputs_scaled, targets_scaled, data_range=1.0, reduction='none')  # returns per-image ssim
    ssim_loss:torch.Tensor = 1 - torch.Tensor(ssim_vals).mean()  # average over batch

    # Total loss
    total_loss:torch.Tensor = pixel_loss + lambda_grad * grad_loss + lambda_ssim * ssim_loss

    return {
        'total': total_loss,
        'pixel': pixel_loss,
        'gradient': grad_loss,
        'ssim': ssim_loss,
    }

def compute_all_loss(outputs, targets, lambda_grad=0.1, lambda_ssim=0.5):
    """
    Output a dict of all loss components.
    """
    losses = {}

    # MSE + gradient
    mse_grad_dict = compute_loss_mse_gradient(outputs, targets, lambda_grad=lambda_grad)
    losses.update(mse_grad_dict)

    # L1 + gradient + SSIM
    l1_grad_ssim_dict = compute_loss_l1_grad_ssim(outputs, targets, lambda_grad=lambda_grad, lambda_ssim=lambda_ssim)
    losses.update(l1_grad_ssim_dict)

    return losses

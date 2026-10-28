import numpy as np
import torch
from typing import Dict, Union

def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """
    Masked Mean Absolute Error for non-negative physical flow/speed levels.
    Standard PeMS benchmark protocol: ignores zero/missing detector values.
    Uses tolerance (labels > 1e-4) to robustly handle floating-point precision after inverse z-score transformation.
    Note: For differences/residuals that can take negative values, use symmetric_masked_diff_mae.
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    elif null_val == 0.0:
        mask = labels > 1e-4
    else:
        mask = torch.abs(labels - null_val) > 1e-4
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def symmetric_masked_diff_mae(pred_diff: torch.Tensor, y_diff: torch.Tensor, raw_y: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Masked MAE for step-to-step variations (differences can be positive or negative).
    Supervises both rising (y_diff > 0) and falling (y_diff < 0) traffic slopes,
    only masking out steps where adjacent raw sensors were unrecorded (raw_y <= 1e-4).
    """
    valid_mask = (raw_y[:, 1:] > 1e-4) & (raw_y[:, :-1] > 1e-4)
    loss = torch.abs(pred_diff - y_diff) * valid_mask.float()
    return loss.sum() / valid_mask.float().sum().clamp(min=1.0)

def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked Root Mean Squared Error with robust floating-point zero tolerance."""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    elif null_val == 0.0:
        mask = labels > 1e-4
    else:
        mask = torch.abs(labels - null_val) > 1e-4
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))

def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """
    Masked Mean Absolute Percentage Error (percentage scale, 0-100%).
    Thresholds at labels > 5.0 (or > 1e-4) to prevent division by near-zero velocities/flows.
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels > 1.0  # Standard PeMS benchmark threshold to avoid zero division
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / (labels + 1e-5))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100.0

def compute_all_metrics(
    preds: Union[torch.Tensor, np.ndarray], 
    labels: Union[torch.Tensor, np.ndarray], 
    null_val: float = 0.0
) -> Dict[str, float]:
    """Computes MAE, RMSE, and MAPE metrics."""
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    mae = masked_mae(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

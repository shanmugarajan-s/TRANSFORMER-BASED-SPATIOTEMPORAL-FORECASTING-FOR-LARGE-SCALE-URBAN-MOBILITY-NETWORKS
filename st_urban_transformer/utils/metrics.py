import numpy as np
import torch
from typing import Dict, Union

def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked Mean Absolute Error (ignores missing/null values)."""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked Root Mean Squared Error."""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))

def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked Mean Absolute Percentage Error (percentage scale, 0-100%)."""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels > 1e-4  # Avoid zero division
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

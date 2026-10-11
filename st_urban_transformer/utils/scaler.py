import numpy as np
import torch
from typing import Union

class StandardScaler:
    """Standard Z-Score Scaler for Spatiotemporal Traffic Tensors."""
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(data, torch.Tensor):
            self.mean = torch.mean(data).item()
            self.std = torch.std(data).item()
        else:
            self.mean = float(np.mean(data))
            self.std = float(np.std(data))
        
        if self.std == 0:
            self.std = 1.0

    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return (data * self.std) + self.mean

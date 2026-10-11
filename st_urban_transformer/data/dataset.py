import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any
from utils.scaler import StandardScaler
from config import DataConfig

class SpatiotemporalDataset(Dataset):
    """
    Sliding window dataset for Spatiotemporal Urban Mobility Forecasting.
    Yields:
      - x: (P, N, C) past observations
      - x_time: (P, D_time) past temporal context (time-of-day, day-of-week)
      - y: (H, N, C) target future observations
      - y_time: (H, D_time) future temporal context
    """
    def __init__(
        self,
        traffic: np.ndarray,
        timestamps: np.ndarray,
        seq_len: int = 12,
        pred_len: int = 12
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.traffic = torch.tensor(traffic, dtype=torch.float32)
        self.timestamps = torch.tensor(timestamps, dtype=torch.long)
        
        self.num_samples = len(self.traffic) - (self.seq_len + self.pred_len) + 1
        if self.num_samples <= 0:
            raise ValueError(f"Insufficient timesteps ({len(self.traffic)}) for seq_len={seq_len} + pred_len={pred_len}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.traffic[s_begin:s_end]
        seq_x_time = self.timestamps[s_begin:s_end]
        
        seq_y = self.traffic[r_begin:r_end]
        seq_y_time = self.timestamps[r_begin:r_end]

        return seq_x, seq_x_time, seq_y, seq_y_time

def create_dataloaders(
    traffic: np.ndarray,
    timestamps: np.ndarray,
    cfg: DataConfig
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Partitions spatiotemporal sequences into train/val/test splits chronologically,
    fits standard scaling on train set only, and builds PyTorch DataLoaders.
    """
    num_timesteps = len(traffic)
    train_end = int(num_timesteps * cfg.train_ratio)
    val_end = int(num_timesteps * (cfg.train_ratio + cfg.val_ratio))

    train_data = traffic[:train_end]
    val_data = traffic[train_end:val_end]
    test_data = traffic[val_end:]

    train_time = timestamps[:train_end]
    val_time = timestamps[train_end:val_end]
    test_time = timestamps[val_end:]

    # Fit scaler solely on training split
    scaler = StandardScaler()
    scaler.fit(train_data)

    norm_train = scaler.transform(train_data)
    norm_val = scaler.transform(val_data)
    norm_test = scaler.transform(test_data)

    train_set = SpatiotemporalDataset(norm_train, train_time, cfg.seq_len, cfg.pred_len)
    val_set = SpatiotemporalDataset(norm_val, val_time, cfg.seq_len, cfg.pred_len)
    test_set = SpatiotemporalDataset(norm_test, test_time, cfg.seq_len, cfg.pred_len)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, scaler

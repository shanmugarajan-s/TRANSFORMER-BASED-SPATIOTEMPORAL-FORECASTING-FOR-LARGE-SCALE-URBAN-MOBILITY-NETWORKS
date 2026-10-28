import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataConfig:
    num_nodes: int = 100          # Number of sensor/road nodes (e.g., 307 for PeMS04, 207 for METR-LA)
    seq_len: int = 12             # Historical input horizon (12 steps = 1 hour at 5-min intervals)
    pred_len: int = 12            # Target prediction horizon (12 steps = 1 hour ahead)
    in_dim: int = 1               # Number of input traffic features (e.g., flow, speed)
    out_dim: int = 1              # Output prediction dimension
    num_timesteps: int = 288 * 30 # Synthetic default: 30 days of 5-min intervals
    train_ratio: float = 0.6      # Standard PeMS benchmark protocol (60/20/20 split)
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    batch_size: int = 32
    num_workers: int = 0          # Windows safe default

@dataclass
class ModelConfig:
    d_model: int = 64             # Hidden representation dimension
    n_heads: int = 4              # Attention heads
    d_ff: int = 256               # Feed-forward hidden dimension
    num_layers: int = 3           # Number of stacked ST-Transformer blocks
    dropout: float = 0.1          # Dropout rate
    node_emb_dim: int = 16        # Dimension for learnable spatial node embeddings
    use_adaptive_adj: bool = True # Enable learned spatial affinity matrix
    use_spatial_prior: bool = True # Fuse physical distance graph with adaptive attention
    attention_type: str = "hybrid" # Options: "linear" (O(N)), "hybrid" (Linear + Graph Diff), "dense" (O(N^2))
    use_delay_modeling: bool = True # Traffic congestion propagation delay modeling

@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7             # Early stopping patience
    grad_clip: float = 5.0        # Gradient clipping norm
    save_dir: str = "checkpoints"
    device: str = "cuda"          # Fallback to cpu handled in training script
    seed: int = 42

@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

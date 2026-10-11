import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .layers import SpatioTemporalEmbedding, STTransformerBlock, PropagationDelayModeling
from config import ModelConfig, DataConfig

class STTransformer(nn.Module):
    """
    Spatio-Temporal Transformer for Large-Scale Urban Mobility Networks.
    Features:
    - Decoupled temporal and sub-quadratic spatial attention (Linear / Hybrid O(N))
    - Learnable dynamic node embeddings for adaptive spatial affinities
    - Physical congestion wave propagation delay modeling across horizons
    - Direct multi-horizon forecasting head
    """
    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        static_adj: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_nodes = data_cfg.num_nodes
        self.seq_len = data_cfg.seq_len
        self.pred_len = data_cfg.pred_len
        self.in_dim = data_cfg.in_dim
        self.out_dim = data_cfg.out_dim
        
        self.d_model = model_cfg.d_model
        self.use_adaptive_adj = model_cfg.use_adaptive_adj
        self.use_spatial_prior = model_cfg.use_spatial_prior
        self.attention_type = model_cfg.attention_type
        self.use_delay_modeling = model_cfg.use_delay_modeling

        # 1. Spatiotemporal Embeddings
        self.embedding = SpatioTemporalEmbedding(
            num_nodes=self.num_nodes,
            in_dim=self.in_dim,
            d_model=self.d_model,
            num_slots=288,
            num_days=7,
            dropout=model_cfg.dropout
        )

        # 2. Adaptive Graph Construction (Source and Destination node matrices)
        if self.use_adaptive_adj:
            self.node_vec_src = nn.Parameter(torch.randn(self.num_nodes, model_cfg.node_emb_dim))
            self.node_vec_dst = nn.Parameter(torch.randn(self.num_nodes, model_cfg.node_emb_dim))
            nn.init.xavier_uniform_(self.node_vec_src)
            nn.init.xavier_uniform_(self.node_vec_dst)

        # Static Adjacency Prior
        if static_adj is not None and self.use_spatial_prior:
            self.register_buffer("static_adj", static_adj)
        else:
            self.static_adj = None

        # 3. Propagation Delay Modeling
        if self.use_delay_modeling:
            self.delay_module = PropagationDelayModeling(d_model=self.d_model, num_horizons=self.pred_len)

        # 4. Stacked Factorized ST-Transformer Blocks
        self.blocks = nn.ModuleList([
            STTransformerBlock(
                d_model=model_cfg.d_model,
                n_heads=model_cfg.n_heads,
                d_ff=model_cfg.d_ff,
                attention_type=model_cfg.attention_type,
                dropout=model_cfg.dropout
            )
            for _ in range(model_cfg.num_layers)
        ])

        # 5. Multi-Horizon Prediction Head
        self.temporal_proj = nn.Linear(self.seq_len, self.pred_len)
        self.channel_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.out_dim)
        )

    def compute_spatial_bias(self) -> Optional[torch.Tensor]:
        """Calculates combined spatial graph bias from static topology and dynamic embeddings."""
        bias = None
        if self.use_adaptive_adj:
            # Learned dynamic correlation matrix
            adaptive_adj = F.softmax(F.relu(torch.mm(self.node_vec_src, self.node_vec_dst.t())), dim=-1)
            bias = adaptive_adj

        if self.static_adj is not None and self.use_spatial_prior:
            if bias is not None:
                bias = 0.5 * bias + 0.5 * self.static_adj
            else:
                bias = self.static_adj

        return bias

    def forward(self, x: torch.Tensor, x_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, N, in_dim) - Historical observations
            x_time: (B, seq_len, 2) - Time features (time-of-day, day-of-week)
        Returns:
            out: (B, pred_len, N, out_dim) - Future forecasts
        """
        # Embed input features
        h = self.embedding(x, x_time) # (B, seq_len, N, d_model)

        # Calculate spatial topological / adaptive bias
        spatial_bias = self.compute_spatial_bias()

        # Pass through stacked Spatio-Temporal Transformer blocks
        for block in self.blocks:
            h = block(h, spatial_bias=spatial_bias)

        # Multi-Horizon Prediction Head
        # (B, seq_len, N, d_model) -> transpose time to project: (B, d_model, N, seq_len)
        h_t = h.permute(0, 3, 2, 1)
        h_proj = self.temporal_proj(h_t) # (B, d_model, N, pred_len)
        
        # Reshape to (B, pred_len, N, d_model)
        h_out = h_proj.permute(0, 3, 2, 1)

        # Apply Vectorized Physical Horizon Propagation Delay Modeling
        if self.use_delay_modeling and self.static_adj is not None:
            # attenuations: (pred_len, N, N)
            attenuations = self.delay_module(self.static_adj)
            # Parallel diffusion across all nodes and horizons:
            h_diffused = torch.einsum('hmn, bhnd -> bhmd', attenuations, h_out)
            h_out = h_out + 0.1 * h_diffused

        # Output feature projection: (B, pred_len, N, out_dim)
        out = self.channel_proj(h_out)
        return out

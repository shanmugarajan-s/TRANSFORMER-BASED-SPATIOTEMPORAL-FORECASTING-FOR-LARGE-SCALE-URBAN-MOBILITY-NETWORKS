import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence order."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length, :] # (1, length, d_model)

class SpatioTemporalEmbedding(nn.Module):
    """
    Unified Spatiotemporal Embedding:
    - Linear feature projection: C -> d_model
    - Temporal position: Sinusoidal sequence order + discrete cyclic embeddings:
      * Time-of-Day embedding (288 slots per 24 hours, 5-min intervals)
      * Day-of-Week embedding (7 days per week)
    - Spatial node identity: Learnable node dictionary (N, d_model)
    """
    def __init__(
        self, 
        num_nodes: int, 
        in_dim: int, 
        d_model: int, 
        num_slots: int = 288, 
        num_days: int = 7, 
        dropout: float = 0.1,
        time_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.feature_proj = nn.Linear(in_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        
        # Periodic temporal embeddings: captures cyclic 5-min diurnal traffic and weekly differences
        self.tod_emb = nn.Embedding(num_slots, d_model)
        self.dow_emb = nn.Embedding(num_days, d_model)
        
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, d_model))
        nn.init.xavier_uniform_(self.node_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_time: torch.Tensor) -> torch.Tensor:
        # x: (B, P, N, C)
        # x_time: (B, P, 2) where [..., 0] is slot in [0..287] and [..., 1] is day in [0..6]
        B, P, N, _ = x.shape

        # 1. Feature projection: (B, P, N, d_model)
        feat_emb = self.feature_proj(x)

        # 2. Temporal embedding: (B, P, 1, d_model)
        order_emb = self.pos_encoder(P).unsqueeze(2) # (1, P, 1, d_model)
        
        # Ensure integer indices for discrete embeddings
        slot_idx = x_time[..., 0].long().clamp(0, 287)
        dow_idx = x_time[..., 1].long().clamp(0, 6)
        tod_emb = self.tod_emb(slot_idx).unsqueeze(2) # (B, P, 1, d_model)
        dow_emb = self.dow_emb(dow_idx).unsqueeze(2) # (B, P, 1, d_model)
        
        temporal_emb = order_emb + tod_emb + dow_emb

        # 3. Spatial node embedding: (1, 1, N, d_model)
        spatial_emb = self.node_embedding.unsqueeze(0).unsqueeze(0)

        # Fusion
        out = feat_emb + temporal_emb + spatial_emb
        return self.dropout(out)

class TemporalAttention(nn.Module):
    """Multi-Head Self-Attention along the temporal sequence dimension."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, N, D)
        B, P, N, D = x.shape
        # Permute to (B * N, P, D) to process all nodes in parallel
        x_perm = x.permute(0, 2, 1, 3).contiguous().view(B * N, P, D)
        
        # Pre-LN & Attention
        norm_x = self.norm(x_perm)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        out = x_perm + self.dropout(attn_out)
        
        # Reshape back to (B, P, N, D)
        return out.view(B, N, P, D).permute(0, 2, 1, 3).contiguous()

class DenseSpatialAttention(nn.Module):
    """
    Standard Full Spatial Self-Attention (O(N^2) Complexity).
    Suitable for medium networks (N < 1,000).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, spatial_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, P, N, D)
        B, P, N, D = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(B * P, N, D)

        q = self.q_proj(x_flat).view(B * P, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(B * P, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(B * P, N, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if spatial_bias is not None:
            if spatial_bias.dim() == 2:
                scores = scores + spatial_bias.unsqueeze(0).unsqueeze(0)
            elif spatial_bias.dim() == 4:
                scores = scores + spatial_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B * P, N, D)
        out = self.dropout(self.out_proj(context)).view(B, P, N, D)

        return x + out

class LinearSpatialAttention(nn.Module):
    """
    Sub-Quadratic Linear Spatial Attention for Massive Urban Mobility Networks (O(N) Complexity).
    - Replaces softmax matrix product with kernelized associative map: phi(Q)(phi(K)^T V).
    - Scales smoothly to N >= 10,000 nodes without OOM.
    - If enable_graph_diffusion=True (hybrid mode), integrates physical road graph diffusion shortcut.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, enable_graph_diffusion: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Graph diffusion message passing shortcut (enabled in 'hybrid', disabled in 'linear')
        self.enable_graph_diffusion = enable_graph_diffusion
        if self.enable_graph_diffusion:
            self.graph_proj = nn.Linear(d_model, d_model)
            
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Non-linear positive kernel map phi(x) = ELU(x) + 1.0"""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, spatial_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, P, N, D)
        B, P, N, D = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(B * P, N, D)

        # 1. Project Q, K, V
        q = self.feature_map(self.q_proj(x_flat).view(B * P, N, self.n_heads, self.head_dim).transpose(1, 2)) # (B*P, H, N, D_h)
        k = self.feature_map(self.k_proj(x_flat).view(B * P, N, self.n_heads, self.head_dim).transpose(1, 2))
        v = self.v_proj(x_flat).view(B * P, N, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. Linear O(N) Associative Attention:
        # Compute KV summary: (B*P, H, D_h, D_h)
        kv = torch.matmul(k.transpose(-2, -1), v)

        # Compute normalizer Z: (B*P, H, N, 1)
        k_sum = k.sum(dim=-2, keepdim=True) # (B*P, H, 1, D_h)
        z = torch.matmul(q, k_sum.transpose(-2, -1)) # (B*P, H, N, 1)

        # Compute global attention output: (B*P, H, N, D_h)
        context = torch.matmul(q, kv) / (z + 1e-6)
        context = context.transpose(1, 2).contiguous().view(B * P, N, D)
        attn_out = self.dropout(self.out_proj(context))

        # 3. Local Road Graph Diffusion shortcut (O(|E|) complexity, only in hybrid mode)
        if self.enable_graph_diffusion and spatial_bias is not None and spatial_bias.dim() == 2:
            # spatial_bias is the (N, N) normalized road adjacency
            g_out = torch.matmul(spatial_bias.unsqueeze(0), x_flat)
            g_out = self.graph_proj(g_out)
            attn_out = attn_out + self.dropout(g_out)

        out = attn_out.view(B, P, N, D)
        return x + out

class SpatialAttention(nn.Module):
    """
    Unified Spatial Attention Layer supporting Dense, Pure Linear, and Hybrid options.
    """
    def __init__(self, d_model: int, n_heads: int, attention_type: str = "hybrid", dropout: float = 0.1):
        super().__init__()
        self.attention_type = attention_type.lower()
        if self.attention_type == "linear":
            # Pure Linear O(N) attention WITHOUT graph diffusion shortcut
            self.impl = LinearSpatialAttention(d_model, n_heads, dropout=dropout, enable_graph_diffusion=False)
        elif self.attention_type == "hybrid":
            # Linear O(N) attention WITH physical road graph diffusion shortcut
            self.impl = LinearSpatialAttention(d_model, n_heads, dropout=dropout, enable_graph_diffusion=True)
        else:
            # Dense Softmax O(N^2) attention
            self.impl = DenseSpatialAttention(d_model, n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, spatial_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.impl(x, spatial_bias)

class PropagationDelayModeling(nn.Module):
    """
    Localized Physical Congestion Wave Propagation Delay Module.
    Models traffic shockwave attenuation across road network horizons simultaneously in O(1) tensor ops.
    Crucially maintains strict topological locality:
    - Non-connected road pairs (adj == 0) receive zero weight.
    - Connected edges diffuse with horizon-dependent distance/velocity decay.
    - Each horizon matrix is row-stochastic (row sum = 1), preventing any O(N) feature magnitude explosion.
    """
    def __init__(self, d_model: int, num_horizons: int = 12):
        super().__init__()
        self.num_horizons = num_horizons
        # Learnable free-flow wave velocity parameter (hops / 5-min slot)
        self.wave_velocity = nn.Parameter(torch.tensor([1.5]))
        self.decay_scale = nn.Parameter(torch.tensor([0.5]))

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj: (N, N) static road adjacency matrix (values in [0, 1])
        Returns:
            attenuations: (num_horizons, N, N) row-normalized local diffusion matrices
        """
        N = adj.shape[0]
        device = adj.device
        
        # Strict edge mask: only direct topological edges can propagate physical waves
        edge_mask = (adj > 0).float()
        
        # Effective graph distance (inversely related to adjacency affinity)
        effective_dist = torch.where(edge_mask > 0, 1.0 / (adj + 1e-4), torch.zeros_like(adj))
        
        horizons = torch.arange(1, self.num_horizons + 1, device=device, dtype=torch.float32).view(self.num_horizons, 1, 1)
        velocity = torch.clamp(torch.abs(self.wave_velocity), min=0.1)
        decay_scale = torch.clamp(torch.abs(self.decay_scale), min=0.01)
        
        # Attenuation along physical edges across horizons
        # At larger horizons h, the propagation distance radius expands: (h * velocity)
        edge_attenuation = edge_mask.unsqueeze(0) * torch.exp(
            - (effective_dist.unsqueeze(0) * decay_scale) / (horizons * velocity + 1e-3)
        )
        
        # Include self-loop (each node retains its local state memory)
        self_loop = torch.eye(N, device=device).unsqueeze(0)
        attenuations = edge_attenuation + self_loop
        
        # Row-normalize to guarantee row-stochastic diffusion: sum_j W_ij = 1.0
        # Prevents any unbounded summation scaling with N
        row_sum = attenuations.sum(dim=-1, keepdim=True) + 1e-6
        attenuations = attenuations / row_sum
        
        return attenuations

class STTransformerBlock(nn.Module):
    """Factorized Spatio-Temporal Transformer Block."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, attention_type: str = "hybrid", dropout: float = 0.1):
        super().__init__()
        self.temporal_attn = TemporalAttention(d_model, n_heads, dropout)
        self.spatial_attn = SpatialAttention(d_model, n_heads, attention_type=attention_type, dropout=dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, spatial_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Temporal Self-Attention
        x = self.temporal_attn(x)
        # 2. Sub-Quadratic Spatial Attention
        x = self.spatial_attn(x, spatial_bias=spatial_bias)
        # 3. Feed-Forward Network
        x = x + self.ffn(self.norm(x))
        return x

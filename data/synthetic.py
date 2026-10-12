import numpy as np
from typing import Tuple
from utils.graph_utils import generate_synthetic_adjacency

def generate_synthetic_traffic(
    num_timesteps: int = 288 * 30, # 30 days of 5-minute slots (288 slots/day)
    num_nodes: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesizes realistic urban traffic flow dynamics:
    - Diurnal rhythms (morning and evening peak hours)
    - Weekly patterns (weekday vs weekend differences)
    - Spatial network propagation along an underlying graph topology
    - Random local congestion shocks
    
    Returns:
        traffic: (T, N, 1) float32 array of traffic flow volumes
        timestamps: (T, 2) float32 array containing [time_of_day_norm, day_of_week_norm]
        adj: (N, N) float32 adjacency matrix
    """
    np.random.seed(seed)
    adj = generate_synthetic_adjacency(num_nodes=num_nodes, seed=seed)
    
    # Time indices
    t = np.arange(num_timesteps)
    slots_per_day = 288
    days_per_week = 7
    
    time_of_day = (t % slots_per_day).astype(np.int64)
    day_of_week = ((t // slots_per_day) % days_per_week).astype(np.int64)
    timestamps = np.stack([time_of_day, day_of_week], axis=-1)

    # 1. Base Diurnal Cycle: Morning peak (~8:00 AM, slot 96) & Evening peak (~18:00 PM, slot 216)
    slot_idx = t % slots_per_day
    morning_peak = np.exp(-((slot_idx - 96) ** 2) / (2 * 18 ** 2))
    evening_peak = np.exp(-((slot_idx - 216) ** 2) / (2 * 22 ** 2))
    daily_base = 20.0 + 50.0 * morning_peak + 60.0 * evening_peak  # Base flow [20, 130]

    # 2. Weekend dampening
    is_weekend = ((t // slots_per_day) % 7 >= 5).astype(np.float32)
    weekend_factor = 1.0 - 0.25 * is_weekend
    temporal_profile = daily_base * weekend_factor  # Shape (T,)

    # 3. Node-specific capacity variations
    node_scale = np.random.uniform(0.6, 1.4, size=(1, num_nodes))
    node_phase_shift = np.random.normal(0, 3.0, size=(1, num_nodes)) # Phase shift in slots
    
    # Broadcast across nodes with phase shifts
    traffic = np.zeros((num_timesteps, num_nodes), dtype=np.float32)
    for n in range(num_nodes):
        shifted_slot = (slot_idx + node_phase_shift[0, n]) % slots_per_day
        m_peak = np.exp(-((shifted_slot - 96) ** 2) / (2 * 18 ** 2))
        e_peak = np.exp(-((shifted_slot - 216) ** 2) / (2 * 22 ** 2))
        node_daily = (20.0 + 50.0 * m_peak + 60.0 * e_peak) * weekend_factor
        traffic[:, n] = node_daily * node_scale[0, n]

    # 4. Spatial Diffusion smoothing (correlated neighbor propagation)
    # Traffic propagates slightly to topological neighbors
    norm_adj = adj / (adj.sum(axis=0, keepdims=True) + 1e-5)
    diffused_traffic = 0.8 * traffic + 0.2 * np.matmul(traffic, norm_adj)
    
    # 5. Stochastic noise and localized congestion events
    noise = np.random.normal(0, 3.0, size=(num_timesteps, num_nodes))
    # Random congestion incidents (drastic drops in flow/speed lasting 1-2 hours)
    num_incidents = num_timesteps // (288 * 2)
    for _ in range(num_incidents):
        inc_t = np.random.randint(0, num_timesteps - 24)
        inc_n = np.random.randint(0, num_nodes)
        diffused_traffic[inc_t : inc_t + 24, inc_n] *= np.random.uniform(0.3, 0.6)

    final_traffic = np.clip(diffused_traffic + noise, a_min=0.0, a_max=None)
    final_traffic = np.expand_dims(final_traffic, axis=-1).astype(np.float32) # (T, N, 1)

    return final_traffic, timestamps, adj

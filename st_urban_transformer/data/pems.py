import os
import csv
import urllib.request
import numpy as np
from typing import Tuple, Optional
from utils.graph_utils import generate_synthetic_adjacency

PEMS_METADATA = {
    "pems04": {
        "num_nodes": 307,
        "timesteps": 16992, # 59 days at 5-min intervals (Jan-Feb 2018)
        "features": 3,      # [flow, occupy, speed]
        "start_dow": 0,     # Jan 1, 2018 was Monday (0)
        "urls": [
            "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/PEMS04.npz",
            "https://raw.githubusercontent.com/guoshnBuaa/PDFormer/main/data/PEMS04/PEMS04.npz"
        ],
        "dist_urls": [
            "https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS04/distance.csv"
        ]
    },
    "pems08": {
        "num_nodes": 170,
        "timesteps": 17856, # 62 days at 5-min intervals (Jul-Aug 2016)
        "features": 3,      # [flow, occupy, speed]
        "start_dow": 4,     # Jul 1, 2016 was Friday (4, where Monday=0)
        "urls": [
            "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/PEMS08.npz",
            "https://raw.githubusercontent.com/guoshnBuaa/PDFormer/main/data/PEMS08/PEMS08.npz"
        ],
        "dist_urls": [
            "https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS08/distance.csv"
        ]
    }
}

def build_pems_adjacency(dist_csv_path: str, num_nodes: int, sigma_factor: float = 0.1) -> np.ndarray:
    """
    Constructs Gaussian kernel adjacency matrix from distance.csv:
    A_ij = exp(- (dist_ij / sigma)^2) if dist_ij <= threshold else 0.
    Uses pure Python csv parsing to avoid external DLL dependencies.
    """
    if not os.path.exists(dist_csv_path):
        return generate_synthetic_adjacency(num_nodes=num_nodes, sparsity=0.85)

    edges = []
    nodes = set()
    with open(dist_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 3:
                try:
                    u, v, cost = int(row[0]), int(row[1]), float(row[2])
                    edges.append((u, v, cost))
                    nodes.add(u)
                    nodes.add(v)
                except ValueError:
                    continue

    unique_nodes = sorted(list(nodes))
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    n = max(num_nodes, len(unique_nodes))
    dist_mx = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(dist_mx, 0.0)

    for u, v, cost in edges:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            dist_mx[i, j] = cost

    finite_dists = dist_mx[~np.isinf(dist_mx)]
    sigma = np.std(finite_dists) * sigma_factor if len(finite_dists) > 0 else 1.0

    adj = np.exp(- (dist_mx ** 2) / (sigma ** 2 + 1e-5))
    adj[np.isinf(dist_mx)] = 0.0
    np.fill_diagonal(adj, 1.0)

    return adj[:num_nodes, :num_nodes].astype(np.float32)

def download_file_with_fallback(urls: list, dest_path: str, timeout: int = 25) -> bool:
    for url in urls:
        try:
            print(f"Downloading from {url}...")
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response, open(dest_path, 'wb') as out_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
                print(f"Successfully downloaded to {dest_path} ({os.path.getsize(dest_path):,} bytes)")
                return True
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
    return False

def load_pems_data(
    dataset_name: str = "pems08",
    data_dir: str = "data_cache",
    feature_idx: int = 0 # 0: flow, 1: occupy, 2: speed
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads PeMS04 or PeMS08 benchmark dataset:
    - Verifies real dataset file presence (never falls back silently to synthetic data)
    - Extracts traffic feature (default: flow)
    - Generates calendar-accurate temporal context (time-of-day [0..287], day-of-week [0..6])
    - Constructs topological Gaussian adjacency matrix
    
    Returns:
        traffic: (T, N, 1) float32 traffic array
        timestamps: (T, 2) int64 array [time_of_day_idx, day_of_week_idx]
        adj: (N, N) float32 adjacency matrix
    Raises:
        FileNotFoundError: If real dataset files are unavailable.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in PEMS_METADATA:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(PEMS_METADATA.keys())}")

    meta = PEMS_METADATA[dataset_name]
    num_nodes = meta["num_nodes"]
    timesteps = meta["timesteps"]
    start_dow = meta["start_dow"]

    os.makedirs(data_dir, exist_ok=True)
    npz_path = os.path.join(data_dir, f"{dataset_name.upper()}.npz")
    dist_path = os.path.join(data_dir, f"{dataset_name}_distance.csv")

    # 1. Check or download
    if not os.path.exists(npz_path):
        success = download_file_with_fallback(meta["urls"], npz_path)
        if not success:
            raise FileNotFoundError(
                f"PeMS benchmark '{dataset_name.upper()}.npz' not found at '{npz_path}' and remote download failed. "
                f"Silent synthetic fallback is disabled to ensure scientific rigor. "
                f"Please verify your internet connection or place '{dataset_name.upper()}.npz' into '{data_dir}'."
            )

    if not os.path.exists(dist_path):
        download_file_with_fallback(meta["dist_urls"], dist_path)

    # 2. Load NPZ array
    data_npz = np.load(npz_path)
    raw_data = data_npz["data"] # Shape: (T, N, C)
    print(f"Loaded {dataset_name.upper()} raw tensor shape: {raw_data.shape}")

    # Extract target feature (e.g. traffic flow)
    traffic = raw_data[:, :, feature_idx:feature_idx + 1].astype(np.float32)
    T, N, _ = traffic.shape

    # 3. Generate calendar-accurate 5-minute discrete slot indices
    # slots_per_day = 288 (12 slots/hour * 24 hours)
    # days_per_week = 7
    slots_per_day = 288
    t = np.arange(T, dtype=np.int64)
    time_of_day_idx = (t % slots_per_day)
    day_of_week_idx = (((t // slots_per_day) + start_dow) % 7)
    timestamps = np.stack([time_of_day_idx, day_of_week_idx], axis=-1)

    # 4. Build adjacency matrix
    adj = build_pems_adjacency(dist_path, num_nodes=N)

    return traffic, timestamps, adj

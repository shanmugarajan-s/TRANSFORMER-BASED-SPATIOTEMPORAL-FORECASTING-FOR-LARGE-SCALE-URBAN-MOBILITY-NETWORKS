import numpy as np
import torch

def calculate_random_walk_matrix(adj_matrix: np.ndarray) -> np.ndarray:
    """Calculates row-normalized random walk transition matrix D^(-1) A."""
    d = np.array(adj_matrix.sum(1), dtype=np.float32)
    d_inv = np.zeros_like(d)
    np.divide(1.0, d, out=d_inv, where=d > 0)
    d_mat = np.diag(d_inv)
    return d_mat.dot(adj_matrix)

def calculate_normalized_laplacian(adj_matrix: np.ndarray) -> np.ndarray:
    """Calculates symmetric normalized Laplacian L = I - D^(-1/2) A D^(-1/2)."""
    d = np.array(adj_matrix.sum(1), dtype=np.float32)
    d_inv_sqrt = np.zeros_like(d)
    np.divide(1.0, np.sqrt(d), out=d_inv_sqrt, where=d > 0)
    d_mat = np.diag(d_inv_sqrt)
    return np.eye(adj_matrix.shape[0], dtype=np.float32) - d_mat.dot(adj_matrix).dot(d_mat)

def generate_synthetic_adjacency(num_nodes: int, sparsity: float = 0.85, seed: int = 42) -> np.ndarray:
    """
    Generates a realistic distance-based graph adjacency matrix 
    using 2D Euclidean sensor coordinates with a Gaussian threshold kernel.
    """
    np.random.seed(seed)
    # Simulated 2D coordinates for urban sensors in a city area
    coords = np.random.uniform(0, 10, size=(num_nodes, 2))
    
    # Pairwise Euclidean distance
    dist = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1))
    
    # Gaussian kernel: exp(-dist^2 / sigma^2)
    sigma = np.std(dist)
    weights = np.exp(- (dist ** 2) / (sigma ** 2 + 1e-5))
    
    # Apply sparsity threshold
    threshold = np.quantile(weights, sparsity)
    adj = np.where(weights >= threshold, weights, 0.0)
    
    # Ensure self-loops
    np.fill_diagonal(adj, 1.0)
    return adj.astype(np.float32)

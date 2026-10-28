import os
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import PipelineConfig
from data import generate_synthetic_traffic, create_dataloaders, load_pems_data
from models import STTransformer
from utils import compute_all_metrics, masked_mae, symmetric_masked_diff_mae

def set_seed(seed: int = 42):
    """Ensures full determinism across Python random, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train Spatiotemporal Urban Mobility Transformer")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "pems04", "pems08"],
                        help="Dataset: 'synthetic', 'pems04' (307 sensors), or 'pems08' (170 sensors)")
    parser.add_argument("--num_nodes", type=int, default=100, help="Number of graph nodes (for synthetic)")
    parser.add_argument("--seq_len", type=int, default=12, help="Input historical sequence length")
    parser.add_argument("--pred_len", type=int, default=12, help="Forecast prediction horizon")
    parser.add_argument("--d_model", type=int, default=64, help="Model hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of ST Transformer layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--attention_type", type=str, default="hybrid", choices=["hybrid", "linear", "dense"],
                        help="Spatial Attention: 'linear' (O(N)), 'hybrid' (Linear + Graph Diff), 'dense' (O(N^2))")
    # Ablation toggles
    parser.add_argument("--no_adaptive_adj", action="store_true", help="Ablation: disable learned adaptive adjacency")
    parser.add_argument("--no_spatial_prior", action="store_true", help="Ablation: disable static road distance prior")
    parser.add_argument("--no_delay_modeling", action="store_true", help="Ablation: disable physical wave delay module")
    
    # Practical execution & reporting options
    parser.add_argument("--diff_weight", type=float, default=0.3,
                        help="Temporal wave variation loss weight to capture natural ups and downs")
    parser.add_argument("--batches_per_epoch", type=int, default=0, help="Max batches per epoch (0 = all dataset batches)")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and directly evaluate checkpoint on test set")
    parser.add_argument("--results_out", type=str, default="", help="Path to write evaluation metrics to JSON via json.dump")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, scaler, device, grad_clip, max_batches: int = 0, diff_weight: float = 0.0):
    model.train()
    total_loss = 0.0
    count = 0
    for x, x_time, y, y_time in dataloader:
        if max_batches > 0 and count >= max_batches:
            break
        x = x.to(device)
        x_time = x_time.to(device)
        y_time = y_time.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x, x_time, y_time)

        # Invert predictions and targets back to original scale for computing metric loss
        preds_inv = scaler.inverse_transform(preds)
        y_inv = scaler.inverse_transform(y)

        # 1. Base absolute level error
        loss = masked_mae(preds_inv, y_inv)

        # 2. Symmetric temporal wave variation loss: captures step-to-step rise/fall dynamics
        # Uses symmetric validity mask: only supervises where BOTH adjacent steps have valid sensor readings (> 1e-4)
        # Prevents asymmetric supervision (where negative differences y_diff < 0 were previously dropped by masked_mae)
        if diff_weight > 0.0:
            pred_diff = preds_inv[:, 1:] - preds_inv[:, :-1]
            y_diff = y_inv[:, 1:] - y_inv[:, :-1]
            loss_diff = symmetric_masked_diff_mae(pred_diff, y_diff, y_inv)
            loss = loss + diff_weight * loss_diff

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)

@torch.no_grad()
def evaluate_epoch(model, dataloader, scaler, device, max_batches: int = 0):
    model.eval()
    all_preds = []
    all_targets = []
    count = 0

    for x, x_time, y, y_time in dataloader:
        if max_batches > 0 and count >= max_batches:
            break
        x = x.to(device)
        x_time = x_time.to(device)
        y_time = y_time.to(device)

        preds = model(x, x_time, y_time)
        preds_inv = scaler.inverse_transform(preds)
        y_inv = scaler.inverse_transform(y.to(device))

        all_preds.append(preds_inv.cpu())
        all_targets.append(y_inv.cpu())
        count += 1

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_all_metrics(all_preds, all_targets)
    return metrics, all_preds, all_targets

def main(args=None):
    if args is None:
        args = parse_args()
    set_seed(args.seed)

    cfg = PipelineConfig()
    cfg.train.seed = args.seed
    cfg.data.num_nodes = args.num_nodes
    cfg.data.seq_len = args.seq_len
    cfg.data.pred_len = args.pred_len
    cfg.data.batch_size = args.batch_size
    cfg.model.d_model = args.d_model
    cfg.model.num_layers = args.num_layers
    cfg.model.attention_type = args.attention_type
    cfg.model.use_adaptive_adj = not args.no_adaptive_adj
    cfg.model.use_spatial_prior = not args.no_spatial_prior
    cfg.model.use_delay_modeling = not args.no_delay_modeling
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.device = args.device
    cfg.train.save_dir = args.save_dir

    device = torch.device(cfg.train.device)
    print(f"\n============================================================")
    print(f"=== Training ST-Transformer on {device} (Seed: {cfg.train.seed}) ===")
    print(f"Nodes: {cfg.data.num_nodes} | History: {cfg.data.seq_len} | Horizon: {cfg.data.pred_len} | Attention: {cfg.model.attention_type.upper()}")
    print(f"Ablation Flags: Adaptive={cfg.model.use_adaptive_adj} | StaticPrior={cfg.model.use_spatial_prior} | DelayMod={cfg.model.use_delay_modeling}")
    print(f"============================================================")

    os.makedirs(cfg.train.save_dir, exist_ok=True)

    # 1. Acquire traffic dataset
    if args.dataset in ["pems04", "pems08"]:
        print(f"Loading {args.dataset.upper()} benchmark dataset (Split: {cfg.data.train_ratio:.1f}/{cfg.data.val_ratio:.1f}/{cfg.data.test_ratio:.1f})...")
        traffic, timestamps, adj = load_pems_data(dataset_name=args.dataset)
        cfg.data.num_nodes = traffic.shape[1]
    else:
        print("Generating synthetic spatiotemporal traffic mobility dataset...")
        traffic, timestamps, adj = generate_synthetic_traffic(
            num_timesteps=cfg.data.num_timesteps,
            num_nodes=cfg.data.num_nodes,
            seed=cfg.train.seed
        )

    # 2. Build DataLoaders and Scaler
    train_loader, val_loader, test_loader, scaler = create_dataloaders(traffic, timestamps, cfg.data)
    print(f"Samples - Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # 3. Initialize Model
    adj_tensor = torch.tensor(adj, dtype=torch.float32, device=device)
    model = STTransformer(cfg.data, cfg.model, static_adj=adj_tensor).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params:,}")

    # Build unique checkpoint name
    ablation_tags = []
    if cfg.model.attention_type != "hybrid":
        ablation_tags.append(cfg.model.attention_type)
    if not cfg.model.use_adaptive_adj:
        ablation_tags.append("no_adaptive")
    if not cfg.model.use_spatial_prior:
        ablation_tags.append("no_spatial_prior")
    if not cfg.model.use_delay_modeling:
        ablation_tags.append("no_delay")

    tag_str = "_" + "_".join(ablation_tags) if ablation_tags else ""
    ckpt_name = f"best_st_transformer_seed{cfg.train.seed}{tag_str}.pt" if (cfg.train.seed != 42 or tag_str) else "best_st_transformer.pt"
    best_ckpt_path = os.path.join(cfg.train.save_dir, ckpt_name)

    # 4. Training Loop (if not eval_only)
    if not args.eval_only:
        optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-5)

        best_val_mae = float("inf")
        patience_counter = 0

        for epoch in range(1, cfg.train.epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, cfg.train.grad_clip, max_batches=args.batches_per_epoch, diff_weight=args.diff_weight)
            val_metrics, _, _ = evaluate_epoch(model, val_loader, scaler, device, max_batches=args.batches_per_epoch)
            scheduler.step()
            epoch_time = time.time() - t0

            val_mae = val_metrics["MAE"]
            print(f"Epoch {epoch:02d}/{cfg.train.epochs} [{epoch_time:.1f}s] - "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f} | Val RMSE: {val_metrics['RMSE']:.4f} | Val MAPE: {val_metrics['MAPE']:.2f}%", flush=True)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "scaler_mean": scaler.mean,
                    "scaler_std": scaler.std,
                    "config": cfg
                }, best_ckpt_path)
                print(f"  -> Saved best checkpoint: {best_ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.train.patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

    # 5. Final Evaluation on Full Test Set
    print("\n=== Standard Literature Benchmark Protocol Evaluation on Test Set ===")
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        epoch_str = f"epoch {checkpoint.get('epoch', 'N/A')}"
        val_str = f"{checkpoint['val_metrics']['MAE']:.4f}" if "val_metrics" in checkpoint else "N/A"
        print(f"Loaded best checkpoint ({best_ckpt_path}) from {epoch_str} with Val MAE {val_str}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at '{best_ckpt_path}'. Training did not produce a valid checkpoint.")

    test_metrics, test_preds, test_targets = evaluate_epoch(model, test_loader, scaler, device)

    # Horizon-specific metrics (Horizon 3 = 15m, Horizon 6 = 30m, Horizon 12 = 60m)
    h3 = compute_all_metrics(test_preds[:, 2], test_targets[:, 2])
    h6 = compute_all_metrics(test_preds[:, 5], test_targets[:, 5])
    h12 = compute_all_metrics(test_preds[:, 11], test_targets[:, 11])

    print("\n------------------------------------------------------------")
    print(f"Horizon 3  (15 min): MAE = {h3['MAE']:.2f} | RMSE = {h3['RMSE']:.2f} | MAPE = {h3['MAPE']:.2f}%")
    print(f"Horizon 6  (30 min): MAE = {h6['MAE']:.2f} | RMSE = {h6['RMSE']:.2f} | MAPE = {h6['MAPE']:.2f}%")
    print(f"Horizon 12 (60 min): MAE = {h12['MAE']:.2f} | RMSE = {h12['RMSE']:.2f} | MAPE = {h12['MAPE']:.2f}%")
    print("------------------------------------------------------------")
    print(f"Overall Average (1-12): MAE = {test_metrics['MAE']:.2f} | RMSE = {test_metrics['RMSE']:.2f} | MAPE = {test_metrics['MAPE']:.2f}%")
    print("------------------------------------------------------------\n")

    result_data = {
        "seed": cfg.train.seed,
        "attention_type": cfg.model.attention_type,
        "use_adaptive_adj": cfg.model.use_adaptive_adj,
        "use_spatial_prior": cfg.model.use_spatial_prior,
        "use_delay_modeling": cfg.model.use_delay_modeling,
        "horizon_3_15min": {"mae": round(float(h3["MAE"]), 2), "rmse": round(float(h3["RMSE"]), 2), "mape": round(float(h3["MAPE"]), 2)},
        "horizon_6_30min": {"mae": round(float(h6["MAE"]), 2), "rmse": round(float(h6["RMSE"]), 2), "mape": round(float(h6["MAPE"]), 2)},
        "horizon_12_60min": {"mae": round(float(h12["MAE"]), 2), "rmse": round(float(h12["RMSE"]), 2), "mape": round(float(h12["MAPE"]), 2)},
        "average_1_12": {"mae": round(float(test_metrics["MAE"]), 2), "rmse": round(float(test_metrics["RMSE"]), 2), "mape": round(float(test_metrics["MAPE"]), 2)}
    }

    if args.results_out:
        with open(args.results_out, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"Exported run metrics via json.dump to: {args.results_out}")

    return result_data

if __name__ == "__main__":
    main()

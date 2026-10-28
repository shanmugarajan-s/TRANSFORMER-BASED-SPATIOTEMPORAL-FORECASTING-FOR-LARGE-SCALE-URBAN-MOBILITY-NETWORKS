import os
import json
import argparse
import torch

from utils import compute_all_metrics
from utils.scaler import StandardScaler
from data import generate_synthetic_traffic, create_dataloaders, load_pems_data
from models import STTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ST-Transformer on the TEST split")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_st_transformer.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--dataset", type=str, default="pems08", choices=["synthetic", "pems04", "pems08"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to save test metrics as JSON, e.g. results/seed42.json")
    return parser.parse_args()


def _r(m):
    """Round a metrics dict to 2 decimals."""
    return {"mae": round(m["MAE"], 2), "rmse": round(m["RMSE"], 2), "mape": round(m["MAPE"], 2)}


@torch.no_grad()
def evaluate(checkpoint_path: str, dataset_name: str, device_str: str, out_path=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device(device_str)
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    scaler = StandardScaler(mean=ckpt["scaler_mean"], std=ckpt["scaler_std"])

    # 1. Load the SAME dataset the checkpoint was trained on
    if dataset_name in ["pems04", "pems08"]:
        traffic, timestamps, adj = load_pems_data(dataset_name=dataset_name)
        cfg.data.num_nodes = traffic.shape[1]
    else:
        traffic, timestamps, adj = generate_synthetic_traffic(
            num_timesteps=cfg.data.num_timesteps,
            num_nodes=cfg.data.num_nodes,
            seed=cfg.train.seed,
        )

    # Safety check: verify that this dataset scale is consistent with the checkpoint
    # (within reasonable 5% margin between train-split and full dataset)
    train_end = int(len(traffic) * cfg.data.train_ratio)
    data_mean = float(traffic[:train_end, :, 0].mean())
    data_std = float(traffic[:train_end, :, 0].std())
    if abs(data_mean - scaler.mean) > 0.05 * max(1.0, abs(scaler.mean)) or \
       abs(data_std - scaler.std) > 0.05 * max(1.0, abs(scaler.std)):
        raise ValueError(
            "Scaler mismatch: checkpoint was trained on a different dataset or split "
            f"(checkpoint mean/std = {scaler.mean:.3f}/{scaler.std:.3f}, "
            f"this data = {data_mean:.3f}/{data_std:.3f})."
        )

    _, _, test_loader, _ = create_dataloaders(traffic, timestamps, cfg.data, scaler=scaler)

    # 2. Rebuild the exact architecture stored in the checkpoint config
    adj_tensor = torch.tensor(adj, dtype=torch.float32, device=device)
    model = STTransformer(cfg.data, cfg.model, static_adj=adj_tensor).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Predict on the TEST split only
    all_preds, all_targets = [], []
    for x, x_time, y, y_time in test_loader:
        x = x.to(device)
        x_time = x_time.to(device)
        y_time = y_time.to(device)
        preds = model(x, x_time, y_time)
        all_preds.append(scaler.inverse_transform(preds).cpu())
        all_targets.append(scaler.inverse_transform(y.to(device)).cpu())

    preds = torch.cat(all_preds, dim=0)      # (S, H, N, C)
    targets = torch.cat(all_targets, dim=0)  # (S, H, N, C)

    overall = compute_all_metrics(preds, targets)
    print("\n" + "=" * 50)
    print("      TEST SET RESULTS (average over steps 1-12)")
    print("=" * 50)
    print(f"  MAE  : {overall['MAE']:.4f}")
    print(f"  RMSE : {overall['RMSE']:.4f}")
    print(f"  MAPE : {overall['MAPE']:.2f} %")
    print("=" * 50)

    print("\n--- Horizon-wise Performance (TEST) ---")
    print(f"{'Step':<6} | {'Ahead':<8} | {'MAE':<8} | {'RMSE':<8} | {'MAPE (%)':<8}")
    print("-" * 52)
    horizon_results = {}
    for h in [1, 3, 6, 12]:
        if h <= cfg.data.pred_len:
            m = compute_all_metrics(preds[:, h - 1], targets[:, h - 1])
            print(f"{h:<6} | {h * 5:>3} min  | {m['MAE']:<8.4f} | {m['RMSE']:<8.4f} | {m['MAPE']:<8.2f}")
            horizon_results[f"horizon_{h}_{h * 5}min"] = _r(m)

    print("\nFor reference, validation metrics stored at the best epoch "
          f"(epoch {ckpt.get('epoch')}): {ckpt.get('val_metrics')}")
    print("Report TEST numbers above, not the validation numbers.")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        result = {
            "checkpoint": checkpoint_path,
            "dataset": dataset_name,
            "split": "test",
            "seed": cfg.train.seed,
            "attention_type": cfg.model.attention_type,
            "use_adaptive_adj": cfg.model.use_adaptive_adj,
            "use_spatial_prior": cfg.model.use_spatial_prior,
            "use_delay_modeling": cfg.model.use_delay_modeling,
            "best_epoch": ckpt.get("epoch"),
            "val_metrics_at_best_epoch": ckpt.get("val_metrics"),
            **horizon_results,
            "average_1_12": _r(overall),
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved test metrics to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.checkpoint, args.dataset, args.device, args.out)

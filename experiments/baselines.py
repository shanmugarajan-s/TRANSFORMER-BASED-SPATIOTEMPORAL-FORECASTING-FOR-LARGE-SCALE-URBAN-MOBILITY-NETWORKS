import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.pems import load_pems_data

def evaluate_empirical_baselines(dataset_name="pems08", train_ratio=0.6, val_ratio=0.2, out_path=None):
    """
    Computes exact non-parametric baselines (Historical Average & Persistence)
    on the exact TEST split of the PeMS dataset.
    """
    print(f"Loading {dataset_name.upper()} dataset for empirical baseline evaluation...")
    traffic, timestamps, adj = load_pems_data(dataset_name=dataset_name)

    num_timesteps = len(traffic)
    train_end = int(num_timesteps * train_ratio)
    val_end = int(num_timesteps * (train_ratio + val_ratio))

    train_traffic = traffic[:train_end, :, 0] # (T_train, N)
    train_tod = timestamps[:train_end, 0]     # (T_train,)

    # 1. Historical Average (HA) Table per 5-min slot of the day (0..287)
    ha_table = np.zeros((288, traffic.shape[1]), dtype=np.float32)
    for slot in range(288):
        mask = (train_tod == slot)
        if np.any(mask):
            ha_table[slot] = np.mean(train_traffic[mask], axis=0)
        else:
            ha_table[slot] = np.mean(train_traffic, axis=0)

    # 2. Extract Test Split sequences
    test_traffic = traffic[val_end:, :, 0]
    test_tod = timestamps[val_end:, 0]
    num_test_samples = len(test_traffic) - 24

    ha_preds, pers_preds, ground_truth = [], [], []

    for i in range(num_test_samples):
        y_true = test_traffic[i + 12 : i + 24] # (12, N)
        last_val = test_traffic[i + 11]       # (N,)

        y_ha = np.zeros_like(y_true)
        for h in range(12):
            fut_slot = int(test_tod[i + 12 + h])
            y_ha[h] = ha_table[fut_slot]

        y_pers = np.tile(last_val, (12, 1))

        ha_preds.append(y_ha)
        pers_preds.append(y_pers)
        ground_truth.append(y_true)

    ha_preds = np.array(ha_preds)       # (S, 12, N)
    pers_preds = np.array(pers_preds)   # (S, 12, N)
    ground_truth = np.array(ground_truth) # (S, 12, N)

    mask = ground_truth > 0

    def get_metrics(preds):
        h3 = float(np.mean(np.abs(preds[:, 2][mask[:, 2]] - ground_truth[:, 2][mask[:, 2]])))
        h6 = float(np.mean(np.abs(preds[:, 5][mask[:, 5]] - ground_truth[:, 5][mask[:, 5]])))
        h12 = float(np.mean(np.abs(preds[:, 11][mask[:, 11]] - ground_truth[:, 11][mask[:, 11]])))
        overall = float(np.mean(np.abs(preds[mask] - ground_truth[mask])))
        return round(h3, 2), round(h6, 2), round(h12, 2), round(overall, 2)

    ha_h3, ha_h6, ha_h12, ha_avg = get_metrics(ha_preds)
    pers_h3, pers_h6, pers_h12, pers_avg = get_metrics(pers_preds)

    print("\n" + "=" * 60)
    print(f"  EMPIRICAL BASELINE RESULTS ON {dataset_name.upper()} TEST SET")
    print("=" * 60)
    print(f"{'Model':<25} | {'15m MAE':<9} | {'30m MAE':<9} | {'60m MAE':<9} | {'Avg MAE':<9}")
    print("-" * 60)
    print(f"{'Historical Average (HA)':<25} | {ha_h3:<9} | {ha_h6:<9} | {ha_h12:<9} | {ha_avg:<9}")
    print(f"{'Persistence (Last Step)':<25} | {pers_h3:<9} | {pers_h6:<9} | {pers_h12:<9} | {pers_avg:<9}")
    print("=" * 60)

    results = [
        {
            "model": "Historical Average (HA)",
            "paradigm": "Non-parametric Baseline",
            "mae_15m": ha_h3,
            "mae_30m": ha_h6,
            "mae_60m": ha_h12,
            "mae_avg": ha_avg,
            "complexity": "O(1)",
            "source": "empirical_test_evaluation"
        },
        {
            "model": "Persistence (Last Step)",
            "paradigm": "Non-parametric Baseline",
            "mae_15m": pers_h3,
            "mae_30m": pers_h6,
            "mae_60m": pers_h12,
            "mae_avg": pers_avg,
            "complexity": "O(1)",
            "source": "empirical_test_evaluation"
        }
    ]

    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved empirical baselines to: {out_path}")

    return results

if __name__ == "__main__":
    evaluate_empirical_baselines()

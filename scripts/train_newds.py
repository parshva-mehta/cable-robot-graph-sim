"""Baseline training on the converted new_platform real data (all-real, no sim mix).

The stock train_real_data.py schedule assumes a sim/real mixture (mix_ratio ~1.0,
load_sim=True) that requires MuJoCo `mjc_*` trajectories in the data_root. This dataset
is real-only (converted via scripts/convert_new_platform.py), so mix_ratio is pinned to 0.

Run from the repo root:  python scripts/train_newds.py
"""
import json
import shutil
from pathlib import Path

from nn_training.real_tensegrity_gnn_training_engine import *
from utilities.misc_utils import setup_logger

torch.autograd.set_detect_anomaly(False)
torch._dynamo.config.cache_size_limit = 512
torch.backends.cuda.matmul.allow_tf32 = True


def train():
    config_file_path = "nn_training/configs/3_bar_real_newds_config.json"
    with open(config_file_path, "r") as j:
        config_file = json.load(j)

    Path(config_file["output_path"]).mkdir(exist_ok=True, parents=True)
    logger = setup_logger(config_file["output_path"])

    config_file["batch_size_per_update"] = 64
    config_file["dt"] = 0.01

    # all-real curriculum: grow the prediction horizon, shrink the LR
    target_dts     = [0.04, 0.08, 0.16]
    epochs         = [300,  300,  200]
    learning_rates = [1e-4, 1e-5, 1e-6]
    batch_sizes    = [64,   64,   64]
    eval_steps     = [2,    5,    5]
    dt_deltas      = [0.10, 0.10, 0.10]   # wide dt window: irregular real steps still pair up
    vel_min_dts    = [0.05, 0.05, 0.05]

    params = list(zip(target_dts, epochs, learning_rates, batch_sizes,
                      eval_steps, dt_deltas, vel_min_dts))
    for run_idx, (n, e, lr, bs, es, dtd, vmd) in enumerate(params, start=1):
        config_file["target_dt"] = n
        config_file["optimizer_params"]["lr"] = lr
        config_file["load_sim"] = False
        config_file["batch_size_per_step"] = bs
        config_file["mix_ratio"] = 0.0        # all real
        config_file["dt_delta"] = dtd
        config_file["eval_step_size"] = es
        config_file["vel_min_dt"] = vmd

        print(f"\n===== run {run_idx}/{len(params)}  target_dt={n} epochs={e} lr={lr} =====",
              flush=True)
        trainer = RealTensegrityMultiSimMultiStepMotorGNNTrainingEngine(config_file, logger)
        trainer.to("cuda:0" if torch.cuda.is_available() else "cpu")
        print("train/val samples:", len(trainer.train_dataset), len(trainer.val_dataset),
              flush=True)
        trainer.run(e)

        out = Path(config_file["output_path"])
        for tag in ("best_loss_model", "best_rollout_model", "best_n_step_rollout_model"):
            src = out / f"{tag}.pt"
            if src.exists():
                shutil.copy(src, out / f"{n}_steps_{tag}.pt")


if __name__ == "__main__":
    train()

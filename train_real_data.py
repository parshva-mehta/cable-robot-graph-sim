import shutil

from nn_training.real_tensegrity_gnn_training_engine import *
from utilities.misc_utils import setup_logger

torch.autograd.set_detect_anomaly(False)
torch._dynamo.config.cache_size_limit = 512
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=64)


def train():
    config_file_path = "nn_training/configs/3_bar_real_train_config.json"
    with open(config_file_path, 'r') as j:
        config_file = json.load(j)

    Path(config_file['output_path']).mkdir(exist_ok=True)
    logger = setup_logger(config_file['output_path'])

    config_file['batch_size_per_update'] = 256
    config_file['dt'] = 0.01

    target_dts = [0.04, 0.08, 0.16, 0.2, 0.24]
    epochs = [100, 120, 80, 50, 40, 30]
    learning_rates = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    batch_sizes = [256, 256, 256, 128, 64, 64]
    load_sim = [False, True, True, True, True, True]
    mix_ratios = [1., 1., 0.95, 0.95, 0.92, 0.9]
    eval_steps = [2, 20, 10, 5, 5, 3]
    dt_deltas = [0.0, 0.0, 0.03, 0.03, 0.04, 0.05]
    vel_min_dts = [0.0, 0.0, 0.05, 0.05, 0.05, 0.05]

    params = list(zip(
        target_dts, epochs, learning_rates,
        load_sim, batch_sizes, eval_steps,
        mix_ratios, dt_deltas, vel_min_dts
    ))
    for n, e, lr, load, batch_size, eval_step, mix_r, dt_d, vm_dt in params[:]:
        config_file['target_dt'] = n
        config_file['optimizer_params']['lr'] = lr
        config_file['load_sim'] = load
        config_file['batch_size_per_step'] = batch_size
        config_file['mix_ratio'] = mix_r
        config_file['dt_delta'] = dt_d
        config_file['eval_step_size'] = eval_step
        config_file['vel_min_dt'] = vm_dt

        trainer = RealTensegrityMultiSimMultiStepMotorGNNTrainingEngine(
            config_file,
            logger
        )

        trainer.to('cuda:0')
        trainer.run(e)

        output_dir = Path(config_file['output_path'])
        try:
            shutil.copy(output_dir / "best_loss_model.pt", output_dir / f"{n}_steps_best_loss_model.pt")
            shutil.copy(output_dir / "best_rollout_model.pt", output_dir / f"{n}_steps_best_rollout_model.pt")
            shutil.copy(output_dir / "best_n_step_rollout_model.pt",
                        output_dir / f"{n}_steps_best_n_step_rollout_model.pt")
        except:
            print("No best_rollout_model")


if __name__ == '__main__':
    train()

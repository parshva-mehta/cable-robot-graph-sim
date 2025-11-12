import random
import shutil

from nn_training.tensegrity_gnn_training_engine import *
from utilities.misc_utils import setup_logger


def train():
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 512
    np.set_printoptions(precision=64)
    config_file_path = "nn_training/configs/cotrain_new_3_bar_train_config.json"
    # config_file_path = "nn_training/configs/multi_sim_new_3_bar_train_config.json"
    # config_file_path = "nn_training/configs/test.json"
    # config_file_path = "nn_training/configs/3_bar_train_config.json"
    with open(config_file_path, 'r') as j:
        cfg = json.load(j)
    Path(cfg['output_path']).mkdir(parents=True, exist_ok=True)

    logger = setup_logger(cfg['output_path'])
    save_code = True
    # idxs.add(datasets[0])
    # logger.info(f"Eval on dataset {datasets[0]}")

    num_steps = [4, 4, 8, 8, 16]
    epochs = [100, 50, 120, 60, 30, 30, 15]
    learning_rates = [1e-5, 1e-6, 1e-6, 1e-7, 1e-8, 1e-9, 1e-9]
    batch_sizes = [256, 512, 256, 512, 256, 256, 256, 256]
    load_sim = [False, True, True, True, True]
    eval_steps = [25, 25, 20, 20, 10, 5, 3]

    params = list(zip(num_steps, epochs, learning_rates, load_sim, batch_sizes, eval_steps))
    for n, e, lr, load, batch_size, eval_step in params[:]:
        cfg['num_steps_fwd'] = n
        cfg['optimizer_params']['lr'] = lr
        cfg['load_sim'] = load
        cfg['save_code'] = save_code
        cfg['batch_size_per_step'] = batch_size
        cfg['batch_size_per_update'] = batch_size
        cfg['eval_stepsize'] = eval_step

        save_code = False
        trainer = TensegrityMultiSimMultiStepMotorGNNTrainingEngine(cfg, logger)

        trainer.to('cuda:0')
        trainer.run(e)

        output_dir = Path(cfg['output_path'])
        try:
            shutil.copy(output_dir / "best_loss_model.pt", output_dir / f"{n}_steps_best_loss_model.pt")
            shutil.copy(output_dir / "best_rollout_model.pt", output_dir / f"{n}_steps_best_rollout_model.pt")
            shutil.copy(output_dir / "best_n_step_rollout_model.pt",
                        output_dir / f"{n}_steps_best_n_step_rollout_model.pt")
        except:
            print("No best_rollout_model")

    del logger


if __name__ == '__main__':
    train()

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
        config_file = json.load(j)

    out = config_file['output_path']
    Path(out).mkdir(parents=True, exist_ok=True)
    config_file['use_gt_act_lens'] = True

    idxs = set()
    for i in range(5):
        config_file['load_sim_path'] = config_file['output_path'] + 'best_rollout_model.pt'

        Path(config_file['output_path']).parent.mkdir(parents=True, exist_ok=True)

        # config_file['val_data_paths'] = []
        #     f"dataset_{i}/mjc_mppi_run_47",
        #     f"dataset_{i}/mjc_mppi_run_31",
        #     f"dataset_{i}/mjc_mppi_run_39",
        #     f"dataset_{i}/pdrop_v2",
        #     f"dataset_{i}/pthrow_v2",
        #     f"dataset_{i}/random_ctrls_v2",
        # ]
        #
        # config_file['train_data_paths'] = []
        #     f"dataset_{i}/mjc_mppi_run_4",
        #     f"dataset_{i}/mjc_mppi_run_9",
        #     f"dataset_{i}/mjc_mppi_run_15",
        #     f"dataset_{i}/mjc_mppi_run_18",
        #     f"dataset_{i}/mjc_mppi_run_22",
        #     f"dataset_{i}/mjc_mppi_run_26",
        #     f"dataset_{i}/pdrop_v0",
        #     f"dataset_{i}/pdrop_v1",
        #     f"dataset_{i}/pthrow_v0",
        #     f"dataset_{i}/pthrow_v1",
        #     f"dataset_{i}/random_ctrls_v0",
        #     f"dataset_{i}/random_ctrls_v2",
        # ]

        # datasets = np.random.choice([k for k in range(9) if k not in idxs], 1, replace=False).tolist()
        # for d in datasets:
        #     config_file['train_data_paths'].extend([
        #         f"mjc_6d_new_platform_real_dl_v1/dataset_{d}/mjc_ccw_1",
        #         f"mjc_6d_new_platform_real_dl_v1/dataset_{d}/mjc_ccw_2",
        #         f"mjc_6d_new_platform_real_dl_v1/dataset_{d}/mjc_cw_1",
        #         f"mjc_6d_new_platform_real_dl_v1/dataset_{d}/mjc_cw_2",
        #         f"mjc_6d_new_platform_real_dl_v1/dataset_{d}/mjc_roll_2",
        #         f"mjc_6d_new_platform_real_dl_v1/dataset_{d}/mjc_roll_3",
        #     ])
        #     config_file['val_data_paths'].extend([
        #         f"mjc_6d_new_platform_real_dl_v1/dataset_{d}/mjc_ccw_3",
        #         f"mjc_6d_new_platform_real_dl_v1/dataset_{d}/mjc_cw_3",
        #         f"mjc_6d_new_platform_real_dl_v1/dataset_{d}/mjc_roll_4",
        #     ])

        Path(config_file['output_path']).mkdir(parents=True, exist_ok=True)

        logger = setup_logger(config_file['output_path'])
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
            config_file['num_steps_fwd'] = n
            config_file['optimizer_params']['lr'] = lr
            config_file['load_sim'] = load
            config_file['save_code'] = save_code
            config_file['batch_size_per_step'] = batch_size
            config_file['batch_size_per_update'] = batch_size
            config_file['eval_stepsize'] = eval_step

            save_code = False

            trainer = TensegrityMultiSimMultiStepGNNTrainingEngine(
                config_file, torch.nn.MSELoss(), 0.01, logger
            )

            # trainer.to('cuda:0')
            trainer.run(e)

            output_dir = Path(config_file['output_path'])
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

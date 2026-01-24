import os
from pathlib import Path

import torch

from model_predictive_control.runner import run_mppi
from utilities.misc_utils import save_curr_code

if __name__ == '__main__':
    # tmp()
    save_flag = True
    torch.backends.cuda.matmul.allow_tf32 = True
    for i in range(9):
        cfg = {
            "model_path": "../../tensegrity/data_sets/tensegrity_real_datasets/"
                          "new_platform_models/multi_4step_recur_motor_newds/best_n_step_rollout_model.pt",
            "output": "../../tensegrity/data_sets/tensegrity_real_datasets/"
                      f"new_platform_models/multi_4step_recur_motor_newds/mppi_ds_iter1_7/free_run_v{i}/",
            "xml_path": 'mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml',
            "vis_xml_path": 'mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml',
            'env_type': '3bar',
            'env_kwargs': {
                'attach_type': 'real_attach'
            },
            'act_interval': 0.5,
            'ctrl_interval': 0.5,
            'sensor_interval': 0.01,
            'horizon': 2.0,
            'max_time': 60,
            'threshold': 0.5,
            'nsamples': 200,
            'planner_type': 'gnn',
            'visualize': True,
            'device': 'cuda',
            'save_data': True,
            'start': [0.0, 0.0, 1.0],
            'goal': [-10, 0, 1.0],
            'boundaries': [-35., -35., 35., 35.],
            'obstacles': [],
            # 'obstacles': [[-3.5, -3.5, -2.5, 1.5],  # [xmin, ymin, xmax, ymax]
            #               [-15.5, -13.5, -14.5, -8.5]],
        }
        output = Path(cfg['output'])
        output.parent.mkdir(exist_ok=True)
        output.mkdir(exist_ok=True)

        if save_flag:
            code_dir_name = "tensegrity_physics_engine"
            curr_code_dir = os.getcwd()
            code_output = output.parent / code_dir_name
            save_curr_code(curr_code_dir, code_output)
            save_flag = False

        with torch.no_grad():
            mppi = TensegrityMPPIRunner(cfg)
            mppi.run_goal()

            combine_videos(Path(cfg['output'], 'vids/'), Path(cfg['output'], 'vids/combined.mp4'))
            shutil.rmtree(Path(cfg['output'], 'frames/'))

            del mppi


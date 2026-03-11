# CableRobotGraphSim

Code for the paper: 

## Install conda

## Create virtual env

```
conda create --name cable_robot_gnn python=3.10
conda activate cable_robot_gnn
pip install -r requirements.txt
```

## Download dataset

## Prepare data

```angular2html
cd ../
mkdir tensegrity
cd tensegrity
mkdir models data_sets
cd data_sets
mv {DOWNLOAD_DIR}/ ./
unzip 
```

## Train model with simulation data only

``python3 train_sim_data.py``

## Train model with simulation and real data

``python3 train_real_data.py``

## Eval model

``python3 eval.py``

## Modifying config files for your own experiments

There are two config json files relevant for simulation and training:
1. `nn_training/configs/*`    <--- json files that control training related parameters/settings
2. `simulators/configs/*`   <--- json files that specify simulation and robot parameters and configuration


## If this work was useful for your research, please cite:

```angular2html
@article{chen2026cablerobotgraphsim,
  title   = {CableRobotGraphSim: A Graph Neural Network for
             Modeling Partially Observable Cable-Driven
             Robot Dynamics},
  author  = {Chen, Nelson and Johnson III, William R. and
             Kramer-Bottiglio, Rebecca and Bekris, Kostas and
             Aanjaneya, Mridul},
  journal = {arXiv preprint arXiv:2602.21331},
  year    = {2026}
}
```


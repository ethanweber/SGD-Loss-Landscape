"""Contains functions for different experiments to run.
"""

from utils import run_cmd, run_cmds


def sgd_vs_gd_experiment(dataset_name,
                         config_name_sgd,
                         config_name_gd):
    """Train GD vs. SGD.
    """

    # dataset
    cmd = (
        " python generate_dataset.py"
        " --n 1000"
        " --d 256"
        " --dataset_name {}"
        " --percent_splits \"0.6, 0.2, 0.2\""
    ).format(dataset_name)
    run_cmd(cmd)

    # plot dataset
    cmd = " python plot_dataset.py --dataset_name {}".format(dataset_name)
    run_cmd(cmd)

    # make the model configs
    # ----- SGD -----
    cmd = (
        " python make_model_configs.py"
        " --num-layers 1"
        " --batch-size 1"
        " --epochs 1000"
        " --lr 0.01"
        " --dataset-name {}"
        " --config-name {}"
    ).format(dataset_name, config_name_sgd)
    run_cmd(cmd)
    # ----- GD -----
    cmd = (
        " python make_model_configs.py"
        " --num-layers 1"
        " --batch-size -1"
        " --epochs 10000"
        " --lr 0.01"
        " --dataset-name {}"
        " --config-name {}"
    ).format(dataset_name, config_name_gd)
    run_cmd(cmd)

    # ---- RUN THE NETWORKS ----

    # run the network
    cmd1 = "python run_network.py --config-name {}".format(config_name_sgd)
    cmd2 = "python run_network.py --config-name {}".format(config_name_gd)
    run_cmds([cmd1, cmd2])

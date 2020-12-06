"""Contains functions for different experiments to run.
"""

import subprocess


def dataset_size(dataset_name, config_name):
    """Train on different amounts of the dataset.
    """

    # dataset
    cmd = (
        " python generate_dataset.py"
        " --n 1000"
        " --d 2"
        " --dataset_name {}"
        " --percent_splits \"0.996, 0.002, 0.002\""
    ).format(dataset_name)
    print("\n\nRunning cmd : {} \n".format(cmd))
    returned_value = subprocess.call(cmd, shell=True)

    # plot dataset
    cmd = " python plot_dataset.py --dataset_name {}".format(dataset_name)
    print("\n\nRunning cmd : {} \n".format(cmd))
    returned_value = subprocess.call(cmd, shell=True)

    # make model config
    cmd = (
        " python make_model_configs.py"
        " --num-layers 2"
        " --batch-size 1"
        " --epochs 100"
        " --lr 0.001"
        " --dataset-name {}"
        " --config-name {}"
    ).format(dataset_name, config_name)
    print("\n\nRunning cmd : {} \n".format(cmd))
    returned_value = subprocess.call(cmd, shell=True)

    # run the network
    cmd = "python run_network.py --config-name {}".format(config_name)
    print("\n\nRunning cmd : {} \n".format(cmd))
    returned_value = subprocess.call(cmd, shell=True)

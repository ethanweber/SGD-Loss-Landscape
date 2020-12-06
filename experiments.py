"""Contains functions for different experiments to run.
"""

import subprocess
import sys


def dataset_size(dataset_name, config_name):
    """Train on different amounts of the dataset.
    """

    # dataset
    cmd = (
        " python generate_dataset.py"
        " --n 1000"
        " --d 256"
        " --dataset_name {}"
        " --percent_splits \"0.6, 0.2, 0.2\""
        # " --percent_splits \"0.1, 0.2, 0.7\""
    ).format(dataset_name)
    print("\n\nRunning cmd : {} \n".format(cmd))
    returned_value = subprocess.call(cmd, shell=True)
    if returned_value != 0.0:
        print("Exiting early")
        sys.exit()

    # plot dataset
    cmd = " python plot_dataset.py --dataset_name {}".format(dataset_name)
    print("\n\nRunning cmd : {} \n".format(cmd))
    returned_value = subprocess.call(cmd, shell=True)
    if returned_value != 0.0:
        print("Exiting early")
        sys.exit()

    # make model config
    # cmd = (
    #     " python make_model_configs.py"
    #     " --num-layers 2"
    #     " --batch-size 1"
    #     " --epochs 1000"
    #     " --lr 0.01"
    #     " --dataset-name {}"
    #     " --config-name {}"
    # ).format(dataset_name, config_name)
    cmd = (
        " python make_model_configs.py"
        " --num-layers 2"
        " --batch-size -1"
        " --epochs 10000"
        " --lr 0.1"
        " --dataset-name {}"
        " --config-name {}"
    ).format(dataset_name, config_name)
    print("\n\nRunning cmd : {} \n".format(cmd))
    returned_value = subprocess.call(cmd, shell=True)
    if returned_value != 0.0:
        print("Exiting early")
        sys.exit()

    # run the network
    cmd = "python run_network.py --config-name {}".format(config_name)
    print("\n\nRunning cmd : {} \n".format(cmd))
    returned_value = subprocess.call(cmd, shell=True)
    if returned_value != 0.0:
        print("Exiting early")
        sys.exit()

"""Dataset size experiment.
"""

from experiments import *

# sgd vs. gd
dataset_name = "ethan_test_dataset"
config_name_sgd = "ethan_config_name_sgd"
config_name_gd = "ethan_config_name_gd"
sgd_vs_gd_experiment(
    dataset_name,
    config_name_sgd,
    config_name_gd)

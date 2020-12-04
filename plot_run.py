"""Generate synthetic datasets.
"""

import argparse
import pprint
from sklearn.datasets import make_regression
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils import (
    make_dir_for_filename,
    load_from_json,
    write_to_json
)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--config-name', type=str, required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    pprint.pprint(args)

    config = load_from_json(os.path.join("configs", args.config_name + ".json"))
    assert config["config_name"] == args.config_name


    # load data to plot
    filename = os.path.join("runs", args.config_name, "train_val.json")
    train_val = load_from_json(filename)
    # pprint.pprint(train_val)

    epochs = sorted(train_val.keys())
    train_loss = []
    val_loss = []
    for epoch in epochs:
        train_loss.append(train_val[epoch]["train"])
        val_loss.append(train_val[epoch]["val"])

    # plot the progress
    fig = plt.figure()
    plt.title("train val progress")
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    # save plot
    filename = os.path.join("plots", config["dataset_name"], args.config_name, "train_loss.png")
    make_dir_for_filename(filename)
    plt.legend()
    plt.savefig(filename)

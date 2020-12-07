"""
Plot the loss from multiple dataset runs.
"""

import argparse
import pprint
from sklearn.datasets import make_regression
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
import seaborn as sns
import pandas as pd

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

    config_name = args.config_name 
    losses = [] 

    config = load_from_json(os.path.join("configs", config_name + ".json"))
    assert config["config_name"] == config_name

    # load data to plot
    filename = os.path.join("runs", config_name, "train_val.json")
    train_val = load_from_json(filename)

    epochs = sorted(train_val.keys())
    train_loss = []
    val_loss = []
    for epoch in epochs:
        # train_loss.append(train_val[epoch]["train"])
        # val_loss.append(train_val[epoch]["val"])
        train_loss = train_val[epoch]["train"]
        val_loss = train_val[epoch]["val"]
        epoch = int(epoch)

        losses.append({"epoch": epoch, "loss": train_loss, "split": "train", "config": config_name})
        losses.append({"epoch": epoch, "loss": val_loss, "split": "validation", "config": config_name})

    df = pd.DataFrame(losses)
    print(df)

    # plot the training losses 
    fig = sns.lineplot(data=df, x="epoch", y="loss", hue="split") 
    
    # save plot
    # make_dir_for_filename("joint_plot.png")
    fig.get_figure().savefig(f"plots/{config_name}.svg")

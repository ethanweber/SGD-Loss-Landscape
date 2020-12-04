"""Generate synthetic datasets.
"""

import argparse
import pprint
from sklearn.datasets import make_regression
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils import make_dir_for_filename

parser = argparse.ArgumentParser(description="")
parser.add_argument('--n', type=int, default=1000, help="number of data points")
parser.add_argument('--d', type=int, default=100, help="feature dimension")
parser.add_argument('--dataset_name', type=str, default="example_dataset", help="dataset name")
parser.add_argument('--percent_splits', type=list, default=[0.6, 0.2, 0.2], help="percent splits")


if __name__ == "__main__":
    args = parser.parse_args()
    
    print("Generating dataset with args:")
    pprint.pprint(args)

    dataset = make_regression(n_samples=args.n, n_features=args.d)
    # TODO(ethan): add correlation and Gaussian noise params
    X, Y = dataset

    dataset_path = os.path.join("datasets", args.dataset_name)
    
    # make sure folder exists
    assert os.path.exists(dataset_path)

    # read a split and plot
    X = np.load(os.path.join(dataset_path, "trainX.npy"))
    Y = np.load(os.path.join(dataset_path, "trainY.npy"))

    # run PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    # plot the PCA results w/ labels
    fig = plt.figure()
    plt.title("train distribution")
    colors = {-1: "red", 1: "green"}
    for label in sorted(colors.keys(), reverse=True):
        idx = np.where(Y==label)
        plt.scatter(
            pca_result[:,0][idx], 
            pca_result[:,1][idx], 
            label=f"y = {label} ({colors[label]})", 
            color=colors[label], 
            alpha=0.1
        )
    filename = os.path.join("plots", args.dataset_name, "train_dist.png")
    make_dir_for_filename(filename)
    plt.legend()
    plt.savefig(filename)

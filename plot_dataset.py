"""Generate synthetic datasets.
"""

import argparse
import pprint
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils import make_dir_for_filename

parser = argparse.ArgumentParser(description="")
parser.add_argument('--n', type=int, default=1000, help="number of data points")
parser.add_argument('--d', type=int, default=100, help="feature dimension")
parser.add_argument('--dataset_name', type=str, default="example_dataset", help="dataset name")

if __name__ == "__main__":
    args = parser.parse_args()
    pprint.pprint(args)

    dataset_path = os.path.join("datasets", args.dataset_name)

    # make sure folder exists
    assert os.path.exists(dataset_path)

    def plot_split(split):

        # read a split and plot
        X = np.load(os.path.join(dataset_path, f"{split}X.npy"))
        Y = np.load(os.path.join(dataset_path, f"{split}Y.npy"))

        # run PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)

        # plot the PCA results w/ labels
        fig = plt.figure()
        plt.title(f"{split} distribution")
        plt.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=Y[:, 0],
            cmap="hot",
            alpha=0.5
        )
        filename = os.path.join("plots", args.dataset_name, f"{split}_dist.png")
        make_dir_for_filename(filename)
        plt.colorbar()
        plt.savefig(filename)

    plot_split("train")
    plot_split("val")
    plot_split("test")

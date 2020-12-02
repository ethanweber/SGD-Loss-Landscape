"""Generate synthetic datasets.
"""

import argparse
import pprint
from sklearn.datasets import make_regression
import os
import numpy as np

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
    
    # make folder if it doesn't exist
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # get split indices
    split_indices = [0]
    end_perc = 0
    for perc in args.percent_splits:
        end_perc += perc
        split_indices.append(int(len(X) * end_perc)) 
    print("Splitting at indices:", split_indices)

    # save files
    np.save(os.path.join(dataset_path, "trainX.npy"), X[split_indices[0]:split_indices[1]])
    np.save(os.path.join(dataset_path, "trainY.npy"), X[split_indices[0]:split_indices[1]])

    np.save(os.path.join(dataset_path, "valX.npy"), X[split_indices[1]:split_indices[2]])
    np.save(os.path.join(dataset_path, "valY.npy"), X[split_indices[1]:split_indices[2]])

    np.save(os.path.join(dataset_path, "testX.npy"), X[split_indices[2]:split_indices[3]])
    np.save(os.path.join(dataset_path, "testY.npy"), X[split_indices[2]:split_indices[3]])

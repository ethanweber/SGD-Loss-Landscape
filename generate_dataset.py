"""Generate synthetic datasets.
"""

import argparse
import pprint
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import MinMaxScaler
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

    dataset = make_regression(n_samples=args.n, n_features=args.d, noise=10.0)
    # dataset = make_classification(n_samples=args.n, n_features=args.d, n_classes=2)
    # TODO(ethan): add correlation and Gaussian noise params
    X, Y = dataset
    X = np.resize(X, (X.shape[0], args.d)).astype(np.float32)
    Y = np.resize(Y, (Y.shape[0], 1)).astype(np.float32)

    # normalize dataset
    scaler = MinMaxScaler()
    scaler.fit(Y)
    X = scaler.transform(X)
    Y = (scaler.transform(Y) * 2.0) - 1.0
    print(X.shape)
    print(Y.shape)



    # TODO: decide if this is accurate
    # Y[Y == 0.0] = -1.0 # so we have labels {-1, 1}

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
    np.save(os.path.join(dataset_path, "trainY.npy"), Y[split_indices[0]:split_indices[1]])

    np.save(os.path.join(dataset_path, "valX.npy"), X[split_indices[1]:split_indices[2]])
    np.save(os.path.join(dataset_path, "valY.npy"), Y[split_indices[1]:split_indices[2]])

    np.save(os.path.join(dataset_path, "testX.npy"), X[split_indices[2]:split_indices[3]])
    np.save(os.path.join(dataset_path, "testY.npy"), Y[split_indices[2]:split_indices[3]])

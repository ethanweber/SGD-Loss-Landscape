import argparse
import pprint
from sklearn.datasets import make_regression

parser = argparse.ArgumentParser(description="")
parser.add_argument('--n', type=int, default=1000, help="number of data points")
parser.add_argument('--d', type=int, default=100, help="feature dimension")
parser.add_argument('--dataset_name', type=int, default="example_dataset", help="dataset name")


if __name__ == "__main__":
    args = parser.parse_args()
    
    print("Gengerating dataset with args:")
    pprint.pprint(args)

    dataset = make_regression(n_samples=args.n, n_features=args.d)
    X, y = dataset
    
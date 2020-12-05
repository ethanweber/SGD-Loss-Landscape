"""

Example:

"""

import torch
import argparse
import pprint
from sklearn.datasets import make_regression
import os
import numpy as np
import json
from modeling import get_model

parser = argparse.ArgumentParser(description="")
parser.add_argument('--num-layers', type=int, default=None, help="Number of layers to add in the model.", required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--dataset-name", type=str, help="Name of dataset.", required=True)
parser.add_argument("--config-name", type=str, help="Name of the config file.", required=True)
parser.add_argument("--lr", type=float, help="Learning rate", default=0.001)
parser.add_argument("--epochs", type=int, help="Number of epochs to train for.", default=1)
parser.add_argument("--num-gpus", type=int, help="Number of GPUs to use", default=1)
parser.add_argument("--optimizer", type=str, help="Optimizer", default="sgd")

# NOTE(ethan): this is important so the weights start at the same place!
torch.manual_seed(0)

def main(args):
    args = parser.parse_args()
    print("Generating model with with args:")
    pprint.pprint(args)

    training_data = np.load(os.path.join("datasets", args.dataset_name, "trainX.npy"))
    feature_dim = training_data.shape[1]

    bs = args.batch_size
    if bs == -1:
        bs = int(training_data.shape[0])

    model = get_model(feature_dim, args.num_layers)
    torch.save(model, os.path.join("models", args.config_name + ".pth"))
 
    model_definition = {
        "config_name": args.config_name,
        "dataset_name": args.dataset_name,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": bs,
        "epochs": args.epochs,
        "num_gpus": args.num_gpus
    }

    output_path = os.path.join("configs", args.config_name + ".json")
    with open(output_path, "w") as f:
        json.dump(model_definition, f)
    print(f"Wrote model config {args.config_name} to {output_path}!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

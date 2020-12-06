"""
Train the network.
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import argparse
import pprint
from sklearn.datasets import make_regression
import os
import numpy as np
import json
from tqdm import tqdm
from utils import (
    make_dir_for_filename,
    load_from_json,
    write_to_json,
    NumpyDataset
)
from modeling import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config-name", type=str, help="Name of the config file.", required=True)
    # parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"], help="Which action to perform.")
    parser.add_argument("--skip-train", action="store_true", help="Whether to train the network.")
    parser.add_argument("--skip-val", action="store_true", help="Whether to measure the validation set performance.")
    parser.add_argument("--skip-test", action="store_true", help="Whether to measure performance on the test set.")
    return parser.parse_args()


def main(args):
    print("Running network with args:")
    pprint.pprint(args)

    print("Loading network from config.")
    config = load_from_json(os.path.join("configs", args.config_name + ".json"))
    assert config["config_name"] == args.config_name
    print("\nConfig:")
    pprint.pprint(config)
    model = torch.load(os.path.join("models", args.config_name + ".pth"))
    print("\nModel:")
    pprint.pprint(model)

    # move to specified devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for param in model.parameters():  # sanity check
        assert param.requires_grad == True

    # TODO(ethan): choose optimer
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.0)
    # optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # setup datasets
    train_dataset = NumpyDataset("datasets", config["dataset_name"], "train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )
    val_dataset = NumpyDataset("datasets", config["dataset_name"], "val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    test_dataset = NumpyDataset("datasets", config["dataset_name"], "test")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    train_val = {}
    best_loss = float("inf")

    results_filename = os.path.join("runs", config["config_name"], "train_val.json")
    make_dir_for_filename(results_filename)

    print(train_dataset.X.shape)
    print(train_dataset.Y.shape)

    # train model
    for epoch in range(config["epochs"]):
        train_loss = 0
        num_points = 0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for idx, batch_data in pbar:
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # print(outputs)
            loss = torch.pow(outputs - labels, 2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            num_points += len(labels)

        train_val[epoch] = {}
        train_val[epoch]["train"] = train_loss / num_points
        print("Train loss: {}".format(train_loss / num_points))

        # add validation
        val_loss = 0
        num_points = 0
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for idx, batch_data in pbar:
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # print(outputs)
            loss = torch.pow(outputs - labels, 2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            val_loss += loss.item()
            num_points += len(labels)

        print("Val loss: {}".format(val_loss / num_points))
        train_val[epoch]["val"] = val_loss / num_points
        write_to_json(results_filename, train_val)


if __name__ == "__main__":
    args = parse_args()
    main(args)

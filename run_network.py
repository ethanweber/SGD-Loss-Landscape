"""Train the network.
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
    write_to_json
)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--config-name", type=str, help="Name of the config file.", required=True)
parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"], help="Which action to perform.")

if __name__ == "__main__":
    args = parser.parse_args()
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
    
    # TODO(ethan): choose optimer
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    criterion = nn.MSELoss()

    class NumpyDataset(torch.utils.data.Dataset):
        def __init__(self, dataset_prefix, dataset_name, mode):
            self.X = np.load(os.path.join(dataset_prefix, dataset_name, f"{mode}X.npy"))
            self.Y = np.load(os.path.join(dataset_prefix, dataset_name, f"{mode}Y.npy"))
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx].astype("float32"), self.Y[idx].astype("float32")
    
    dataset = NumpyDataset("datasets", config["dataset_name"], args.mode)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config["batch_size"],
        shuffle=args.mode == "train",
        num_workers=4
    )

    # move to specified devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_val = {}

    for epoch in range(config["epochs"]):
        # train mode
        model.train()

        running_loss = 0.0
        num_points = 0.0
        print("Epoch: {}".format(epoch))
        pbar = tqdm(enumerate(dataloader))
        for idx, batch_data in pbar:
            
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            num_points += len(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # # forward + backward + optimize
            outputs = model(inputs).view(-1) # since scalar output
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            pbar.set_description("Average loss: {:.3f}".format(running_loss / num_points))

        # populate the dictionary
        train_val[epoch] = {}
        train_val[epoch]["train"] = running_loss

        # -----------
        # test mode
        model.eval()

        valdataset = NumpyDataset("datasets", config["dataset_name"], "val")
        valdataloader = torch.utils.data.DataLoader(
            valdataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        print("Running on validation.")
        valloss = 0.0
        num_points = 0
        for idx, batch_data in tqdm(enumerate(valdataloader)):

            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            num_points += len(labels)

            # # forward + backward + optimize
            outputs = model(inputs).view(-1) # since scalar output
            loss = criterion(outputs, labels)
            valloss += loss.item()
        
        valaverageloss = valloss / num_points
        print(valaverageloss)
        train_val[epoch]["val"] = valaverageloss

    # save the results
    filename = os.path.join("runs", config["config_name"], "train_val.json")
    make_dir_for_filename(filename)
    write_to_json(filename, train_val)

    print('Finished training')

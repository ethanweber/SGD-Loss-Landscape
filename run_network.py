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

parser = argparse.ArgumentParser(description="")
parser.add_argument("--config_name", type=str, help="Name of the config file.", required=True)
parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"], help="Which action to perform.")

if __name__ == "__main__":
    args = parser.parse_args()
    
    print("Generating dataset with args:")
    pprint.pprint(args)

    print("Loading network from config.")
    with open(os.path.join("configs", args.config_name + ".json")) as json_file:
        config = json.load(json_file)
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
        num_workers=1
    )

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        print("Epoch: {}".format(epoch))
        for idx, batch_data in tqdm(enumerate(dataloader)):
            
            inputs, labels = batch_data

            # zero the parameter gradients
            optimizer.zero_grad()

            # # forward + backward + optimize
            outputs = model(inputs).view(-1) # since scalar output
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.item()
            # if idx % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
            # TODO: update tqdm bar

    print('Finished Training')

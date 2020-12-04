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
    write_to_json
)

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_prefix, dataset_name, mode):
        self.X = np.load(os.path.join(dataset_prefix, dataset_name, f"{mode}X.npy"))
        self.Y = np.load(os.path.join(dataset_prefix, dataset_name, f"{mode}Y.npy"))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx].astype("float32"), self.Y[idx].astype("float32")

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
    
    # TODO(ethan): choose optimer
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    criterion = nn.MSELoss(reduction="sum")
     
    train_dataset = NumpyDataset("datasets", config["dataset_name"], "train") 
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )

    # move to specified devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_val = {}
    best_loss = float("inf")

    results_filename = os.path.join("runs", config["config_name"], "train_val.json")
    make_dir_for_filename(results_filename)

    for epoch in range(config["epochs"]):
        if not args.skip_train:
            # train mode
            model.train()

            running_loss = 0.0
            num_points = 0.0
            print("Epoch: {}".format(epoch))
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
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

        if not args.skip_val:
            # test mode
            model.eval()

            val_dataset = NumpyDataset("datasets", config["dataset_name"], "val")
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4
            )

            print("Running on validation.")
            val_loss = 0.0
            num_points = 0
            for idx, batch_data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):

                inputs, labels = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                num_points += len(labels)

                # # forward + backward + optimize
                outputs = model(inputs).view(-1) # since scalar output
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            
            if val_loss < best_loss:
                # save the results
                write_to_json(results_filename, train_val)
                best_loss = val_loss
                print("Achieved the best loss at {best_loss}.")

            print("Validaton set loss:", val_loss)
            train_val[epoch]["val"] = val_loss 

    print('Finished training')

    if not args.skip_test:
        print('Starting test...')

        test_dataset = NumpyDataset("datasets", config["dataset_name"], "test") 
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4
        )
        print("Running on validation.")
        test_loss = 0.0
        num_points = 0
        for idx, batch_data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            num_points += len(labels)

            # # forward + backward + optimize
            outputs = model(inputs).view(-1) # since scalar output
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        
        print("Test set loss:", test_loss)


if __name__ == "__main__":
    args = parse_args()
    main(args)

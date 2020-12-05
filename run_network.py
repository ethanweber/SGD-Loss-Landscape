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
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.0)
     
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


    def get_error_on_dataset(dataset, model):
        """Returns the error for a dataset and model.
        """

        # test mode
        model.eval()

        criterion = nn.MSELoss(reduction="sum")

        total_error = 0.0
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        # TODO: assert batch_size is 1
        for idx, batch_data in enumerate(dataloader):
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            error = criterion(outputs, labels)
            total_error += error.item()
        return total_error / len(dataset)

    train_val[-1] = {}

    error = get_error_on_dataset(train_dataset, model)
    print("Initial training error: {:03f}".format(error))
    train_val[-1]["train"] = error

    val_dataset = NumpyDataset("datasets", config["dataset_name"], "val")
    error = get_error_on_dataset(val_dataset, model)
    print("Initial validation error: {:03f}".format(error))
    train_val[-1]["val"] = error


    for epoch in range(config["epochs"]):
        if not args.skip_train:
            # train mode
            model.train()

            criterion = nn.MSELoss(reduction="sum")

            train_loss = 0.0
            num_points = 0.0
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for idx, batch_data in pbar:
                
                inputs, labels = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                num_points += len(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                (loss / len(labels)).backward()
                optimizer.step()

                # print statistics
                train_loss += loss.item()
                pbar.set_description("Epoch: {}, Ave. train loss: {:.5f}".format(epoch, train_loss / num_points))

            # populate the dictionary
            error = get_error_on_dataset(train_dataset, model)
            train_val[epoch] = {}
            train_val[epoch]["train"] = error
            print("Training set loss:", error)


        if not args.skip_val:
            val_loss = get_error_on_dataset(val_dataset, model)
            train_val[epoch]["val"] = val_loss
            print("Validaton set loss:", val_loss)
            if val_loss < best_loss:
                # save the results
                best_loss = val_loss
                torch.save(model, os.path.join("runs", args.config_name, "best_weights.pth"))
                print(f"Achieved the best loss at {best_loss}.")
        
        write_to_json(results_filename, train_val)
        print("\n\n")

    print('Finished training')

    if not args.skip_test:
        print('Starting test...')

        test_dataset = NumpyDataset("datasets", config["dataset_name"], "test") 
        test_loss = get_error_on_dataset(test_dataset, model)
        filename = os.path.join("runs", config["config_name"], "test.json")
        make_dir_for_filename(filename)
        write_to_json(filename, {"test_loss": test_loss})
        print("Test set loss:", test_loss)


if __name__ == "__main__":
    args = parse_args()
    main(args)

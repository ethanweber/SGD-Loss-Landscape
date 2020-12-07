import os
import sys
sys.path.insert(0,'..')
import utils
import cifar10.model_loader
import pprint
import torch

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    return net

def load_regression(config_name, data_parallel=False):
    print("Loading network from config.")
    config = utils.load_from_json(os.path.join("../configs", config_name + ".json"))
    assert config["config_name"] == config_name
    print("\nConfig:")
    pprint.pprint(config)
    path = os.path.join("../runs/", config_name, "best_weights.pth")
    print(f"Loading weights from", path)
    model = torch.load(path)
    model = model.to("cuda")
    print("\nModel:")
    pprint.pprint(model)

    train_dataset = utils.NumpyDataset("../datasets", config["dataset_name"], "train") 
    test_dataset = utils.NumpyDataset("../datasets", config["dataset_name"], "test") 
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True,
        num_workers=4
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=True,
        num_workers=4
    )
    return model, train_dataloader, test_dataloader

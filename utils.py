import os
import json
import torch
import numpy as np

def make_dir_for_filename(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_from_json(filename: str):
    assert filename.endswith(".json")
    with open(filename, "r") as f:
        return json.load(f)

def write_to_json(filename: str, content: dict):
    assert filename.endswith(".json")
    with open(filename, "w") as f:
        json.dump(content, f)

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_prefix, dataset_name, mode):
        self.X = np.load(os.path.join(dataset_prefix, dataset_name, f"{mode}X.npy"))
        self.Y = np.load(os.path.join(dataset_prefix, dataset_name, f"{mode}Y.npy"))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx].astype("float32"), self.Y[idx].astype("float32")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

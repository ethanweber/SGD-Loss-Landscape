import torch
import argparse
import pprint
from sklearn.datasets import make_regression
import os
import numpy as np
import json

def get_model(feature_dim, num_layers):
    """Returns the model.
    """
    activation = torch.nn.Softplus()
    # activation = torch.nn.ReLU()
    layers = [torch.nn.Linear(feature_dim, 1024), activation]
    layers += [item for _ in range(num_layers) for item in [torch.nn.Linear(1024, 1024), activation]]
    layers.append(torch.nn.Linear(1024, 1))
    model = torch.nn.Sequential(*layers)
    return model

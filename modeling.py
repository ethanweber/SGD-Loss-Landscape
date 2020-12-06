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
    # activation = torch.nn.Softplus()
    activation = torch.nn.ELU()
    # activation = torch.nn.ReLU()
    # activation = torch.nn.LeakyReLU()
    layers = [torch.nn.Linear(feature_dim, 512), activation]
    layers += [item for _ in range(num_layers) for item in [torch.nn.Linear(512, 512), activation]]
    layers.append(torch.nn.Linear(512, 1))
    layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*layers)
    return model

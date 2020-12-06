#!/usr/bin/env python
# coding: utf-8

# #### Tests

# In[1]:


import numpy as np
from sklearn.linear_model import LinearRegression
import os
import json
import matplotlib.pyplot as plt

from utils import (
    make_dir_for_filename,
    load_from_json,
    write_to_json
)


# In[2]:


X = np.load(os.path.join("datasets/ethan_test_dataset", "trainX.npy"))
y = np.load(os.path.join("datasets/ethan_test_dataset", "trainY.npy"))


# In[3]:


reg = LinearRegression().fit(X, y)
reg.score(X, y)


# In[4]:


def get_data(config_name):
    filename = os.path.join("runs", config_name, "train_val.json")
    train_val = load_from_json(filename)
    
    epochs = sorted([int(x) for x in train_val.keys()])
#     print(epochs)
    train_loss = []
    val_loss = []
    for epoch in epochs:
        train_loss.append(train_val[str(epoch)]["train"])
        val_loss.append(train_val[str(epoch)]["val"])
    
    print(train_loss[-1])
    data = {
        "config_name": config_name,
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    return data


# In[5]:


def plot_data(data):
    fig = plt.figure(figsize=(10,10))
    plt.title("train val progress")
    plt.plot(data["epochs"], data["train_loss"], label="train")
    plt.plot(data["epochs"], data["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
#     plt.yscale("log")
    plt.savefig("plots/ethan_test_dataset/curves.png")

data = get_data(config_name="ethan_config_name")
plot_data(data)

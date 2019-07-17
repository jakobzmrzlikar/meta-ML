import json
import numpy as np
from keras.utils import to_categorical

def encode(path):
    with open(path, "r") as f:
        meta = json.load(f)

    with open("encoding.json", 'r') as e:
        encoding = json.load(e)

    dataset = meta["dataset"]
    conf = meta["model"]
    arch = conf["architecture"]
    fit = conf["hyperparameters"]["fit"]
    build = conf["hyperparameters"]["build"]
    results = meta["results"]

    data = np.zeros(10, dtype='int')
    for a, num in zip(arch["Activation"], arch["Dense"]):
        data[encoding["activation"][a]] += num
    
    data = np.append(data, [dataset["instances"], dataset["features"]])
    data = np.append(data, [fit["batch_size"], fit["epochs"]])
    data = np.append(data, to_categorical(encoding["loss"][build["loss"]], num_classes=14, dtype='int'))
    data = np.append(data, to_categorical(encoding["optimizer"][build["optimizer"]], num_classes=7, dtype='int'))
    data = np.append(data, round(100*results["acc"]))
    return data

import json
import numpy as np

def encode(path):
    with open(path, "r") as f:
        meta = json.load(f)

    with open("encoding.json", 'r') as e:
        encoding = json.load(e)

    dataset = meta["dataset"]
    conf = meta["model"]
    arch = conf["architecture"]
    hyperparams = conf["hyperparameters"]
    results = meta["results"]

    data = [0 for i in range(10)]
    for a, num in zip(arch["Activation"], arch["Dense"]):
        data[encoding["activation"][a]] += num

    data.append(dataset["instances"])
    data.append(dataset["features"])
    data.append(hyperparams["batch_size"])
    data.append(hyperparams["epochs"])
    data.append(encoding["loss"][hyperparams["loss"]])
    data.append(encoding["optimizer"][hyperparams["optimizer"]])
    data.append(round(100*results["acc"]))

    data = np.array(data, dtype='int')
    return data

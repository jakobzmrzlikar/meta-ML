import json
import numpy as np

loss = {
    "mean_squared_error": 0,
    "mean_absolute_error": 1,
    "mean_absolute_percentage_error": 2,
    "mean_squared_logarithmic_error": 3,
    "squared_hinge": 4,
    "hinge" : 5,
    "categorical_hinge": 6,
    "logcosh": 7,
    "categorical_crossentropy": 8,
    "sparse_categorical_crossentropy": 9,
    "binary_crossentropy": 10,
    "kullback_leibler_divergence": 11,
    "poisson": 12,
    "cosine_proximity": 13
}

optimizer = {
    "sgd": 0,
    "rmsprop": 1,
    "adagrad": 2,
    "adadelta": 3,
    "adam": 4,
    "adamax": 5,
    "nadam": 6
}

activation = {
    "softmax": 0,
    "elu": 1,
    "selu": 2,
    "softplus": 3,
    "softsign": 4,
    "relu": 5,
    "tanh": 6,
    "sigmoid": 7,
    "hard_sigmoid": 8,
    "linear": 9
}

def encode(path):
    with open(path, "r") as f:
        meta = json.load(f)

    dataset = meta["dataset"]
    conf = meta["model"]
    arch = conf["architecture"]
    hyperparams = conf["hyperparameters"]
    results = meta["results"]

    data = [0 for i in range(10)]
    for a, num in zip(arch["Activation"], arch["Dense"]):
        data[activation[a]] += num

    data.append(dataset["instances"])
    data.append(dataset["features"])
    data.append(hyperparams["batch_size"])
    data.append(hyperparams["epochs"])
    data.append(loss[hyperparams["loss"]])
    data.append(optimizer[hyperparams["optimizer"]])
    data.append(round(100*results["acc"]))

    data = np.array(data, dtype='int')
    return data

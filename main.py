import json
import csv
import numpy as np
import psutil
import keras
from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, Activation
from timeit import default_timer as timer

from meta_encoder import encode

def run(config):
    with open(config, 'r') as f:
        meta = json.load(f)

    dataset = meta["dataset"]
    conf = meta["model"]
    arch = conf["architecture"]
    hyperparams = conf["hyperparameters"]
    results = meta["results"]


    with open("data/" + dataset["id"] + "/train.csv") as d:
        reader = csv.reader(d, delimiter=',')
        data = list(reader)
        data = np.array(data, dtype="float")
        np.random.shuffle(data)

    x = data[:, :-1]
    y = keras.utils.to_categorical(data[:, -1])

    dataset["instances"] = data.shape[0]
    dataset["features"] = data.shape[1]-1


    if hyperparams["type"] == "Sequential":
        model = Sequential()
        for d, a in zip(arch["Dense"], arch["Activation"]):
            model.add(Dense(
            d,
            activation=a)) #TODO: First layer should have input size spec

    model.compile(
        optimizer=hyperparams["optimizer"],
        loss=hyperparams["loss"],
        metrics=hyperparams["metrics"]
    )

    start = timer()

    model.fit(
        x,
        y,
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        verbose=1,
    )
    end = timer()

    with open("data/" + dataset["id"] + "/test.csv") as d:
        reader = csv.reader(d, delimiter=',')
        data = list(reader)
        data = np.array(data).astype(float)

    x = data[:, :-1]
    y = keras.utils.to_categorical(data[:, -1])

    loss_and_metrics = model.evaluate(
        x,
        y,
        batch_size=hyperparams["batch_size"]
    )

    for name,value in zip(model.metrics_names, loss_and_metrics):
        results[name] = value
    results["time"] = end-start

    pprint(meta)

    with open(config, 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

if __name__ == "__main__":
    name = "pima_indian_diabetes"
    for i in range(840, 1793):
        print("---------------------------------------------------------------")
        print("Config {}/{}".format(i, 1793))
        config = "config/"+name+'_'+str(i)+".json"
        run(config)
        # data = np.load("meta/data.npy")
        # data = np.vstack((data, np.array(encode(config))))
        # np.save("meta/data.npy", data)

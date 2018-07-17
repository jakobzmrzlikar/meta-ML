import json
import numpy as np
import psutil
from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, Activation
from timeit import default_timer as timer


with open("meta/conf.json", 'r') as f:
    meta = json.load(f)

conf = meta["model"]
dataset = meta["dataset"]
arch = conf["architecture"]
train = conf["train"]
results = meta["results"]

data = np.genfromtxt("data/" + dataset["id"] + "/train.csv", delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7))
labels = np.genfromtxt("data/" + dataset["id"] + "/train.csv", delimiter=',', usecols=(-1))

if conf["type"] == "Sequential":
    model = Sequential()
    for d, a in zip(arch["Dense"], arch["Activation"]):
        model.add(Dense(
        d,
        activation=a,
        input_dim=8)) #TODO: First layer should have input size spec

model.compile(
    optimizer=conf["optimizer"],
    loss=conf["loss"],
    metrics=conf["metrics"]
)

start = timer()
cpu = psutil.cpu_freq()
model.fit(
    data,
    labels,
    epochs=train["epochs"],
    batch_size=train["batch_size"]
)
end = timer()

data = np.genfromtxt("data/" + dataset["id"] + "/test.csv", delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7)) #TODO this is very ugly temporary fix
labels = np.genfromtxt("data/" + dataset["id"] + "/test.csv", delimiter=',', usecols=(-1))
loss_and_metrics = model.evaluate(
    data,
    labels,
    batch_size=train["batch_size"]
)

for name,value in zip(model.metrics_names, loss_and_metrics):
    results[name] = value
results["time"] = end-start

pprint(meta)

with open("meta/conf.json", 'w') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

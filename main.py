import json
from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

with open("../meta/conf.json", 'r') as f:
    meta = json.load(f)

conf = meta["model"]
arch = conf["architecture"]
train = conf["train"]
results = meta["results"]

data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

if conf["type"] == "Sequential":
    model = Sequential()
    for d, a in zip(arch["Dense"], arch["Activation"]):
        model.add(Dense(
        d,
        activation=a,
        input_dim=100)) #TODO: First layer should have input size spec

model.compile(
    optimizer=conf["optimizer"],
    loss=conf["loss"],
    metrics=conf["metrics"]
)

model.fit(
    data,
    labels,
    epochs=train["epochs"],
    batch_size=train["batch_size"]
)

data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))
loss_and_metrics = model.evaluate(
    data,
    labels,
    batch_size=train["batch_size"]
)

for name,value in zip(model.metrics_names, loss_and_metrics):
    results[name] = value

pprint(meta)

with open("../meta/conf.json", 'w') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

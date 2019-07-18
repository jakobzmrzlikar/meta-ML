import json
import csv
import numpy as np
import psutil
import keras
from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import svm
from timeit import default_timer as timer


def build_model(model_type, num_features, arch, hyperparams):
    if model_type == "Sequential":
        model = Sequential()
        model.add(Dense(num_features,
                        activation='linear', input_shape=(num_features,)))

        for d, a in zip(arch["Dense"], arch["Activation"]):
            model.add(Dense(d, activation=a))

        model.compile(**hyperparams["compile"])

    elif model_type == 'SVC':
        model = svm.SVC(**hyperparams["compile"])

    return model


def test_model(model, model_type, hyperparams, datatset, test=None, preprocess_data=True, binary=False):
    if test is None:
        with open("data/" + datatset["id"] + "/test.csv", 'r') as d:
            reader = csv.reader(d, delimiter=',')
            test = list(reader)
            test = np.array(test, dtype="float")

    if preprocess_data:
        np.random.shuffle(test)
        x = test[:, :-1]
        if binary:
            y = test[:, -1]
        else:
            y = keras.utils.to_categorical(test[:, -1])
    else:
        x = test[0]
        y = test[1]

    if model_type == "Sequential":
        loss_and_metrics = model.evaluate(
            x,
            y,
            **hyperparams["evaluate"]
        )
    elif model_type == "SVC":
        loss_and_metrics = model.score(
            x,
            y, 
            **hyperparams["evaluate"]
        )
    return loss_and_metrics

def run(config, train=None, test=None, preprocess_data=True, binary=False):
    with open(config, 'r') as f:
        meta = json.load(f)

    dataset = meta["dataset"]
    conf = meta["model"]
    hyperparams = conf["hyperparameters"]
    results = meta["results"]

    if train is None:
        with open("data/" + dataset["id"] + "/train.csv", 'r') as d:
            reader = csv.reader(d, delimiter=',')
            train = list(reader)
            train = np.array(train, dtype="float")


    if preprocess_data:
        np.random.shuffle(train)
        x = train[:, :-1]
        if binary:
            y = train[:, -1]
        else:
            y = keras.utils.to_categorical(train[:, -1])
    else:
        x = train[0]
        y = train[1]

    dataset["instances"] = len(x)
    dataset["features"] = len(x[0])

    model = build_model(model_type=conf["type"], num_features=dataset['features'], arch=conf["architecture"], hyperparams=hyperparams)

    start = timer()

    history = model.fit(
        x,
        y,
        **hyperparams["fit"]
    )
    end = timer()
    results["time"] = end-start

    loss_and_metrics = test_model(model, conf["type"], hyperparams, dataset, test, preprocess_data, binary)
    for name,value in zip(model.metrics_names, loss_and_metrics):
        results[name] = value  

    pprint(meta)
    with open(config, 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

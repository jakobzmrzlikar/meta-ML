import json
import csv
import numpy as np
from keras.models import model_from_json

from core import run
from meta_encoder import encode
from meta_model import quick_train, cross_validation

if __name__ == "__main__":
    # Load the meta model architecture
    #
    # with open("meta/model.json", 'r') as f:
    #     meta = json.load(f)
    # model = model_from_json(meta)
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    # Load the metadataset
    data = np.load("meta/data.npy")

    name = input("Name of the dataset: ")

    # Load training and test data
    with open("data/" + name + "/train.csv", 'r') as d:
        reader = csv.reader(d, delimiter=',')
        train_data = list(reader)
        train_data = np.array(train_data, dtype="float")

    with open("data/" + name + "/test.csv", 'r') as d:
        reader = csv.reader(d, delimiter=',')
        test_data = list(reader)
        test_data = np.array(test_data, dtype="float")


    for i in range(1, 50):
        print("---------------------------------------------------------------")
        print("Config {}/{}".format(i, 50))
        config = "config/"+name+'_'+str(i)+".json"

        # Generate new metadata
        run(config, train=train_data, test=test_data)

        # Append new data to metadataset
        data = np.vstack((data, np.array(encode(config))))

        # quick_train(model, data)

    np.save("meta/data.npy", data)

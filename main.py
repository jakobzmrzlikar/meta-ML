import json
import csv
import numpy as np
from keras.models import model_from_json

from core import run
from meta_encoder import encode
from meta_model import quick_train, cross_validation

def load(name):
    with open("data/" + name + "/train.csv", 'r') as d:
        reader = csv.reader(d, delimiter=',')
        train_data = list(reader)
        train_data = np.array(train_data, dtype="float")

    with open("data/" + name + "/test.csv", 'r') as d:
        reader = csv.reader(d, delimiter=',')
        test_data = list(reader)
        test_data = np.array(test_data, dtype="float")
    
    return (train_data, test_data)

if __name__ == "__main__":

    # Load the metadataset
    data = np.load("meta/data.npy")

    # Load training and test data
    name = input("Name of the dataset: ")
    train_data, test_data = load(name)

    for i in range(1, 50):
        print("---------------------------------------------------------------")
        print("Config {}/{}".format(i, 50))
        config = "config/generated/"+name+'_'+str(i)+".json"

        # Generate new metadata
        run(config, train=train_data, test=test_data, preprocess_data=True, binary=False)

        # Append new data to metadataset
        data = np.vstack((data, np.array(encode(config))))

    np.save("meta/data.npy", data)

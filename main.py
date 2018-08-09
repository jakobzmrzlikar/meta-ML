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
    #
    # Load training and test data
    # with open("data/" + name + "/train.csv", 'r') as d:
    #     reader = csv.reader(d, delimiter=',')
    #     train_data = list(reader)
    #     train_data = np.array(train_data, dtype="float")
    #
    # with open("data/" + name + "/test.csv", 'r') as d:
    #     reader = csv.reader(d, delimiter=',')
    #     test_data = list(reader)
    #     test_data = np.array(test_data, dtype="float")

    import keras
    from keras.datasets import reuters
    from keras.preprocessing.text import Tokenizer

    max_words = 1000
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words, test_split=0.2)
    num_classes = np.max(y_train) + 1
    tokenizer = Tokenizer(num_words=max_words)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    train_data, test_data = (x_train, y_train), (x_test, y_test)

    for i in range(1, 50):
        print("---------------------------------------------------------------")
        print("Config {}/{}".format(i, 50))
        config = "config/"+name+'_'+str(i)+".json"

        # Generate new metadata
        run(config, train=train_data, test=test_data, preprocess_data=False)

        # Append new data to metadataset
        data = np.vstack((data, np.array(encode(config))))

        # quick_train(model, data)

    np.save("meta/data.npy", data)

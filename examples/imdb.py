import json
import csv
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import model_from_json
from keras.preprocessing import sequence

from core import run
from meta_encoder import encode
from meta_model import quick_train, cross_validation

if __name__ == "__main__":

    # Load the metadataset
    data = np.load("meta/data.npy")

    name = input("Name of the dataset: ")

    # Load and preprocess reuters dataset
    max_words = 1000
    maxlen = 400
    (x_train, y_train), (x_test, y_test) = imbd.load_data(num_words=max_words, test_split=0.2)
    num_classes = np.max(y_train) + 1
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
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

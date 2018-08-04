import json
from keras.models import model_from_json

from core import run
from meta_encoder import encode
from meta_model import quick_train, cross_validation

if __name__ == "__main__":
    # Load the meta model architecture
    with open("meta/model.json", 'r') as f:
        meta = json.load(f)
    model = model_from_json(meta)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    name = input("Name of the dataset: ")

    for i in range(1209, 1793):
        print("---------------------------------------------------------------")
        print("Config {}/{}".format(i, 1793))
        config = "config/"+name+'_'+str(i)+".json"

        # Generate new metadata
        run(config)

        # Append new data to metadataset
        data = np.vstack((data, np.array(encode(config))))

        quick_train(model, data)

    np.save("meta/data.npy", data)

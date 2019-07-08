import json
import numpy as np
from keras.models import model_from_json

from meta_model import quick_train, cross_validation


# Load the metadataset
data = np.load("meta/data.npy")

# Load the meta model architecture

with open("meta/model.json", 'r') as f:
    meta = json.load(f)
model = model_from_json(meta)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

quick_train(model, data, eval=True)

import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.model_selection import train_test_split, StratifiedKFold
import time

seed = 42
np.random.seed(seed)

def quick_train(model, data):

    # Prepare training data
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1]

    #Uncomment this if you want to evaluate the model at the end
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x,
    #     y,
    #     test_size=0.2,
    #     random_state=seed)

    model.load_weights("meta/weights.hdf5", by_name=False)

    model.fit(
        x,
        y,
        epochs=1,
        batch_size=16,
        verbose=0,
    )

    # loss_and_metrics = model.evaluate(x_test, y_test, verbose=0)
    # print("{}: {:.2f}".format(model.metrics_names[1], loss_and_metrics[1]))
    #
    # scores = []
    # scores.append(loss_and_metrics[1])
    # classes = model.predict(x_test)
    # big = small = 0
    # for i in range(len(classes)):
    #     error = int(abs(classes[i][0]-y_test[i]))
    #     print("Predicted: {}, actual: {}, error: {}".format(
    #     int(classes[i][0]), y_test[i], error))
    #
    # print()
    # print("Test loss:", loss_and_metrics)

    model.save_weights("meta/weights.hdf5")

def cross_validation(model, data):

    # Prepare training data
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1]

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    scores = []

    model.load_weights("meta/weights.hdf5", by_name=False)

    for train, test in kfold.split(x, y):
        model.fit(
            x[train],
            y[train],
            epochs=100,
            batch_size=16,
            verbose=0,
        )
        loss_and_metrics = model.evaluate(x[test], y[test], verbose=0)
        print("{}: {:.2f}".format(model.metrics_names[1], loss_and_metrics[1]))
        scores.append(loss_and_metrics[1])

    print("Average: {:.2f} (+/- {:.2f})".format(np.mean(scores), np.std(scores)))

    model.save_weights("meta/weights.hdf5")

if __name__ == "__main__":
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_dim=16))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1, activation='linear'))

    meta = model.to_json()
    with open("meta/model.json", 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

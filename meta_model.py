import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

data = np.load("meta/data.npy")
np.random.shuffle(data)
x = data[:, :-1]
y = data[:, -1]
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.15,
    random_state=2)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=16))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=1,
    verbose=1,
)
loss_and_metrics = model.evaluate(x_test, y_test)
print()
print("Test loss:", loss_and_metrics)
print()

classes = model.predict(x_test)
big = small = 0
for i in range(len(classes)):
    error = int(abs(classes[i][0]-y_test[i]))
    print("Predicted: {}, actual: {}, error: {}".format(
    int(classes[i][0]), y_test[i], error))

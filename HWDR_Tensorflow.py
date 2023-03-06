import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

## loading digit data
digit_dataset = load_digits()
    
## flattening the data
n_samples = len(digit_dataset.images)
data = digit_dataset.images.reshape((n_samples, -1))

## splitting the data
X_train, X_test, Y_train, Y_test =  train_test_split(data, digit_dataset.target,test_size=0.2)

## setting up neural network
model = keras.Sequential([
    keras.layers.Dense(35, input_shape=(64,), activation="gelu"),
    keras.layers.Dense(10, activation="sigmoid"),
    
])

# keras.layers.Dense(50, input_shape=(64,), activation="relu"),
# keras.layers.Dense(30, input_shape=(50,), activation="relu"),
# keras.layers.Dense(10, input_shape=(30,), activation="sigmoid")

model.compile(loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

## training the neural network
model.fit(X_train, Y_train, epochs=10)

## evaluating model against test data
model.evaluate(X_test, Y_test)

print(f"actual image value : {Y_test[0]}")

plt.imshow(X_test[0].reshape(8,8))
plt.show()

predictedVal = model.predict(X_test[0].reshape(-1, 64))
predictedVal = predictedVal[0]

print(f"predicted image value : {np.where(predictedVal == max(predictedVal))[0][0]}")
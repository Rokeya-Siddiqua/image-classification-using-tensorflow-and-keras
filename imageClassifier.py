import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#code starts
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

#load a pre-defined dataset (70k images of 28X28)
fashion_mnist = keras.datasets.fashion_mnist

#pull out data from dataset (60k for training and 10k for testing)
(train_images, train_levels), (test_images, test_levels) = fashion_mnist.load_data()

#show data
#print(train_levels[0])
#print(train_images[0])

plt.imshow(train_images[0], cmap=plt.gray, vmin=0, vmax=255)
plt.show()

#define our neural net structure
model = keras.Sequential([
    #input is a 28x28 images into a single 784x1 imput layer
    keras.layers.Flatten(input_shape=(28,28)),
    #hidden layer is 128 deep
    keras.layers.Dense(128, activation=tf.nn.relu),
    #output is 0-10
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
#compile our model
model.compile(optimizer=tf.optimizers.Admin(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#train our model
model.fit(train_images,train_levels,epochs=5)
#test our model
test_loss = model.evaluate(test_images,test_levels)
#make predictions
predictions = model.predict(test_images)

print(predictions[0])
print(list(predictions[0]).index(max(predictions[0])))
print(test_levels[0])

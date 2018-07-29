import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

print(tf.__version__)
fashion_mnist=keras.datasets.fashion_mnist ## A dataset consisting of 28*28 pixels images with a label from 0 to 9 respectively
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data() ##it returns four numpy arrays 60,000 training images,10,000 testing images
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker',
             'Bag','Ankle boot' ]
##print("Shape of training images {}".format(train_images.shape)

train_images= train_images / 255.0  ## we will preprocess the data to a value from 0 to 1 a float value
test_images=test_images / 255.0



plt.figure(figsize=(10,10)) ## checking 25 images 
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

model = keras.Sequential([                      ##Building the layers with the first layer flattens or convert matrix of 28 *28 to a 1d array
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])    ##metrics is used to monitor the training process in terms of accuracy
model.fit(train_images, train_labels, epochs=5)   ## start training 
test_loss, test_acc = model.evaluate(test_images, test_labels)   ## evaluating on test images

print('Test accuracy:', test_acc)
predictions = model.predict(test_images)

np.argmax(predictions[0]) ## finding which no. has the highest probability

## Now testing on a single image
img = test_images[0]

print(img.shape)
img = (np.expand_dims(img,0))  ## we can feed a batch at once so need to convert a single image in format of a batch 

print(img.shape)
predictions = model.predict(img)

print(predictions)
prediction = predictions[0]

np.argmax(prediction)

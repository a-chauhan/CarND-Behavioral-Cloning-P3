import os
import csv
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

import sklearn
from random import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, MaxPooling2D, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

from keras.models import Model
import matplotlib.pyplot as plt

# Read from the driving_log.csv file, the recorded training data
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
samples = lines

learning_rate = 0.0014


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Defining the generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            # images = []
            # angles = []
            # for batch_sample in batch_samples:
            #     name = './data/IMG/'+batch_sample[0].split('/')[-1]
            #     center_image = cv2.imread(name)
            #     center_angle = float(batch_sample[3])
            #     images.append(center_image)
            #     angles.append(center_angle)
            #     images.append(cv2.flip(center_image, 1))
            #     angles.append(center_angle*-1.0)

            # X_train = np.array(images)
            # y_train = np.array(angles)
            # print(len(angles), len(images))
            # print(len(X_train[0].shape))

            # Build the array for images and corresponding stearing angle measurement
            images = []
            angles = []
            correction = 0.01
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    angle = float(batch_sample[3])
                    if (i == 1):
                        angle += correction
                    elif (i == 2):
                        angle -= correction
                    angles.append(angle)

            # print(len(angles), len(images))

            # Inverting the images and adding to the data set
            augmented_images = []
            augmented_angles = []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            # X_train = np.array(images)
            # y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

keep_prob = 0.5

# Creating the model architecture
# <================================================= MODEL ARCHITECTURE START =========================================================>
model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
# model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))

model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(160, 320, 3),
        output_shape=(160, 320, 3)))


# Cropping the image
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))

model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))

# Classic five convolutional, Nvidia model and additional maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Dropout(keep_prob))
model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Dropout(keep_prob))
model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164, activation='relu'))

model.add(Dropout(keep_prob))
model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer=Adam(learning_rate))
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)

# =============================================== MODEL ARCHITECTURE END ===========================================================

model.save("./model.h5")

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

print("training completed")
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
import scipy.misc
import matplotlib.pyplot as plt
import csv

from utility import crop, flip, gamma, delete_file
import json
import pandas as pd

DRIVING_LOG_FILE = './data/driving_log.csv'
IMG_PATH = './data/IMG/'
STEERING_COEFFICIENT = 0.23

epochs = 8
samples_per_epoch = 20032
number_of_validation_samples = 6400
learning_rate = 1e-4
relu = 'relu'
model_name = 'model.json'
weights_name = 'model.h5'

def next_image_batch(batch_size=64):
    lines = []
    with open(DRIVING_LOG_FILE) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    data = lines
    num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in rnd_indices:
        value = np.random.randint(0, 3)
        angle = data[index][3]
        if value == 0:
            image = data[index][value].split('/')[-1]
            angle = float(data[index][3]) + float(STEERING_COEFFICIENT)
            image_files_and_angles.append((image, angle))
        if value == 1:
            image = data[index][value].split('/')[-1]
            angle = float(data[index][3])
            image_files_and_angles.append((image, angle))
        if value == 2:
            image = data[index][value].split('/')[-1]
            angle = float(data[index][3]) - float(STEERING_COEFFICIENT)
            image_files_and_angles.append((image, angle))


        # if rnd_image == 0:
        #     img = data.iloc[index]['left'].strip()
        #     angle = data.iloc[index]['steering'] + STEERING_COEFFICIENT
        #     image_files_and_angles.append((img, angle))

        # elif rnd_image == 1:
        #     img = data.iloc[index]['center'].strip()
        #     angle = data.iloc[index]['steering']
        #     image_files_and_angles.append((img, angle))
        # else:
        #     img = data.iloc[index]['right'].strip()
        #     angle = data.iloc[index]['steering'] - STEERING_COEFFICIENT
        #     image_files_and_angles.append((img, angle))

    return image_files_and_angles


def process_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1, resize_dim=(64, 64), do_shear_prob=0.9):

    image = crop(image, top_crop_percent, bottom_crop_percent)

    image, steering_angle = flip(image, steering_angle)

    image = gamma(image)

    image = scipy.misc.imresize(image, resize_dim)

    return image, steering_angle

def generate_next_batch(batch_size=64):
    while True:
        X_batch = []
        y_batch = []
        images = next_image_batch(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread(IMG_PATH + img_file)
            raw_angle = angle
            new_image, new_angle = process_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)

# Inspired from:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

# starts with five convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation(relu))

model.add(Dense(100))
model.add(Activation(relu))

model.add(Dense(50))
model.add(Activation(relu))

model.add(Dense(10))
model.add(Activation(relu))

model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate), loss="mse", )

# create two generators for training and validation
train_gen = generate_next_batch()
validation_gen = generate_next_batch()

history = model.fit_generator(train_gen,
                              samples_per_epoch=samples_per_epoch,
                              nb_epoch=epochs,
                              validation_data=validation_gen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)

# finally save our model and weights
delete_file(model_name)
delete_file(weights_name)

json_string = model.to_json()
with open(model_name, 'w') as outfile:
    json.dump(json_string, outfile)

model.save_weights(weights_name)

# FILE: TRAIN_PHONE_FINDER.PY
# PURPOSE: TRAINS PHONE FINDER NETWORK
# CMMD LINE 2 RUN: python train_phone_finder.py ./find_phone

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import pickle
import sys
import os
import cv2
import time
from sklearn.model_selection import train_test_split

def train_phone_finder(data_path):
    start = time.time()
    os.chdir(data_path)
    cwd = os.getcwd()
    label_data = []

    # Access label data
    for file in os.listdir(cwd):
        if file[-4:] == '.txt':
            with open ('labels.txt') as file:
                for line in file:
                    data_line = [l.strip() for l in line.split(' ')]
                    label_data.append(data_line)

    # Read + Resize image data
    x_var = []
    y_var = []
    for label in label_data:
        img = cv2.imread(label[0])
        resized_img = cv2.resize(img, (64,64))
        x_var.append(resized_img.tolist())
        y_var.append([float(label[1]), float(label[2])])
    x_var = np.asarray(x_var)
    y_var = np.asarray(y_var)

	# Rescale img values to be from 0 to 1
    x_var = np.interp(x_var, (x_var.min(), x_var.max()), (0,1))

	# Split data into training/test datasets
    (x_train, x_test, y_train, y_test) = train_test_split(x_var, y_var, test_size=0.25)

	# Parameters
    height = x_var.shape[-2]
    width = x_var.shape[-3]
    depth = x_var.shape[-1]
    input_shape = (height, width, depth)

    #Compile and train the model
    model = create_model(input_shape, y_var)
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    print("\n\n\nTraining begins ...\n\n\n")
    model.fit(x_train, y_train, epochs = 200, verbose=0, batch_size=8, validation_data=(x_test, y_test))
    model.save('./../train_phone_finder_weights.h5')
    print("\n\n\nTraining complete. \n\n\n")
    print("Program Runtime {0} seconds, Exiting ...".format(time.time()-start))


#Create model architecture
def create_model(input_shpe, y_var):
	model = tf.keras.models.Sequential([
	    tf.keras.layers.Conv2D(32, (3,3), padding="same", input_shape=input_shpe, activation='relu'),
	    tf.keras.layers.MaxPooling2D(2,2),
	    tf.keras.layers.Dropout(0.3),
	    tf.keras.layers.Conv2D(64, (3,3), padding="same", activation='relu'),
	    tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(64, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
	    tf.keras.layers.Dropout(0.3),
	    tf.keras.layers.Dense(256, activation="sigmoid"),
        tf.keras.layers.Dense(128, activation="sigmoid"),
        tf.keras.layers.Dense(y_var.shape[-1])
	])
	return model

def main():
	train_phone_finder(sys.argv[1])

if __name__ == "__main__":
	main()

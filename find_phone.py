# FILE: FIND_PHONE.PY
# PURPOSE: TEST PHONE FINDER NETWORK
# CMMD LINE 2 RUN: python find_phone.py ./find_phone_test_images/51.jpg

import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def find_phone(img_path):
        # Generate img_path without image filename
	file_name = list(img_path.split('/'))[-1]
	path_list = list(img_path.split('/'))[:-1]
	new_path = ''
	for i, element in enumerate (path_list):
		if i < len(path_list) - 1:
			new_path = new_path + element + '/'
		else:
			new_path = new_path + element

	# Setting path to folder containing weights
	os.chdir(new_path)

	# Read + Resize image data
	x_var = []
	img = cv2.imread(file_name)
	resized_img = cv2.resize(img, (64,64))
	x_var.append(resized_img.tolist())
	x_var = np.asarray(x_var)

	# Rescale img values to be from 0 to 1
	x_var = np.interp(x_var, (x_var.min(), x_var.max()), (0,1))

	# Load model and predict with it
	model = load_model('./../train_phone_finder_weights.h5')
	result = model.predict(x_var)
	print("\n\nPhone in image {0} is located at x-y coordinates given below.".format(str(file_name)))
	print("\n{:.4f} {:.4f}".format(result[0][0], result[0][1]))


def main():
    find_phone(sys.argv[1])

if __name__ == "__main__":
    main()

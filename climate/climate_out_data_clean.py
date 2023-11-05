import cv2
import random
import numpy as np

data_name = "data/climate_out/climate_"

img = cv2.imread(data_name+"0001"+".jpg")
img = cv2.resize(img, dsize=(28,28))
img = np.array([img])

li = list(range(1779))
random.shuffle(li)
for i in li:
	num = str(i+2).zfill(4)
	img1 = cv2.imread(data_name+num+".jpg")
	img1 = cv2.resize(img1, dsize=(28,28))
	img1 = np.array([img1])
	img = np.vstack((img, img1))

np.save('climate_out.npy', img)
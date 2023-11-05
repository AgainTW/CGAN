import cv2
import random
import numpy as np

data_name = "data/ftfm_out/"

img = cv2.imread(data_name+"1"+".jpg")
img = cv2.resize(img, dsize=(28,28))
img = np.array([img])

li = list(range(769))
random.shuffle(li)
for i in li:
	img1 = cv2.imread(data_name+str(i+2)+".jpg")
	img1 = cv2.resize(img1, dsize=(28,28))
	img1 = np.array([img1])
	img = np.vstack((img, img1))

#print(img.shape)
np.save('ftfm_out.npy', img)
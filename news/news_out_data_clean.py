import cv2
import random
import numpy as np

data_name = "data/news_out/news-"

img = cv2.imread(data_name+"0000000"+".jpg")
img = cv2.resize(img, dsize=(28,28))
img = np.array([img])

li = list(range(1379))
random.shuffle(li)
for i in li:
	num = str(i+1).zfill(7)
	img1 = cv2.imread(data_name+num+".jpg")
	img1 = cv2.resize(img1, dsize=(28,28))
	img1 = np.array([img1])
	img = np.vstack((img, img1))

print(img.shape)
np.save('news_out.npy', img)
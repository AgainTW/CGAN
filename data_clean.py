from keras.datasets.mnist import load_data
import cv2
import numpy as np

data_name = "data/climate_out/climate_"

img1 = cv2.imread(data_name+"0001"+".jpg")
img2 = cv2.imread(data_name+"0002"+".jpg")
img1 = cv2.resize(img1, dsize=(28,28))
img2 = cv2.resize(img2, dsize=(28,28))
img1 = np.array([img1])
img2 = np.array([img2])
img = np.vstack((img1, img2))
print(img.shape)


#(trainX, trainy), (_, _) = load_data()
#print(trainX.shape)
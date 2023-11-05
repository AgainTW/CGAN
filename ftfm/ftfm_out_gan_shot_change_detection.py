'''
20230320的訓練
預測ftfm_out的class
並根據預測判斷shot change
'''
import numpy as np
from math import sqrt
from matplotlib import pyplot
from numpy import asarray
from numpy.random import randn
import gan
import cv2
import pandas as pd

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model
from keras.models import load_model

def decode(arr):
	for i in range(arr.shape[1]):
		if( arr[0,i] > 0.1 ):
			return i

model = load_model('20230320_7_ftfm/d_model_3600.h5')
# 對unshuffle進行判斷
x = np.load('np_data/ftfm_out_unshuffle.npy')

y = []
clip = [0,8,29,49,66,90,134,148,157,178,206,225,298,305,331,355,372,394,429,446,483,518
,549,576,594,630,655,674,692,730,770]
for i in range(1, len(clip)):
	for j in range(clip[i]-clip[i-1]):
		y.append(i)

shot_list = []

for i in range(x.shape[0]):
	x_temp = np.array([x[i]])
	_,temp = model.predict(x_temp)
	shot_list.append(decode(temp))

shot_class = pd.DataFrame()
shot_class["real"] = y
shot_class["predict"] = shot_list
shot_class.to_csv('ftfm_out_shot_class.csv', index=False)


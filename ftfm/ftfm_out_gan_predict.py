import numpy as np
from math import sqrt
from matplotlib import pyplot
from numpy import asarray
from numpy.random import randn
import cv2
import gan

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model
from keras.models import load_model


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_class):
  # generate points in the latent space
  x_input = randn(latent_dim * n_samples)
  # reshape into a batch of inputs for the network
  z_input = x_input.reshape(n_samples, latent_dim)
  # generate labels
  labels = asarray([n_class for _ in range(n_samples)])
  return [z_input, labels]

# create and save a plot of generated images
def save_plot(examples, n_examples, filename):
  # plot images
  for i in range(n_examples):
    # define subplot
    pyplot.subplot(int(sqrt(n_examples)), int(sqrt(n_examples)), 1 + i)
    # turn off axis
    pyplot.axis('off')
    # plot raw pixel data
    pyplot.imshow(examples[i, :, :, :])
  pyplot.savefig(filename)
  #pyplot.show()

# load model
model = load_model('20230320_7_ftfm/g_model_3600.h5')

# plot class
latent_dim = 100
n_examples = 100 # must be a square

for i in range(50):
  name = "predict_plot_"+str(i)
  n_class = i
  # generate images
  latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
  # generate images
  X  = model.predict([latent_points, labels])
  # scale from [-1,1] to [0,1]
  X = (X + 1) / 2.0
  # plot the result
  save_plot(X, n_examples, name)

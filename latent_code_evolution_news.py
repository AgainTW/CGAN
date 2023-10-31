import cv2
import numpy as np
from matplotlib import pyplot
from numpy import asarray
from numpy.random import randn
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

latent_dim = 100
n_examples = 1

model = load_model('20230319_6_train_good/g_model_'+str(420).zfill(4)+'.h5')
latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class=23)
X  = model.predict([latent_points, labels])
X = (X + 1) / 2.0
evolution = np.array(X)

for i in range(1,15):
	# load model
	model = load_model('20230319_6_train_good/g_model_'+str(i*420+420).zfill(4)+'.h5')
	latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class=23)
	X  = model.predict([latent_points, labels])
	X = (X + 1) / 2.0
	evolution = np.vstack((evolution, X))

for i in range(15):
	pyplot.subplot(3, 5, 1+i)
	pyplot.axis('off')
	pyplot.imshow(evolution[i, :, :, :])
pyplot.savefig('evolution')

'''
# load model
model = load_model('20230319_6_train_good/g_model_7560.h5')

# plot class
latent_dim = 100
n_examples = 1

latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class=11)
X  = model.predict([latent_points, labels])
X = (X + 1) / 2.0
pyplot.imshow(X[0,:,:,:])
pyplot.show()
'''
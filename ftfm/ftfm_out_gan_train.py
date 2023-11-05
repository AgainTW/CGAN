import numpy as np
from math import sqrt
from matplotlib import pyplot
import gan
from keras.utils.vis_utils import plot_model
import time

#start = time.time()

x = np.load('ftfm_out.npy')
# 格式化climate_out
y = []
clip = [0,8,29,49,66,90,134,148,157,178,206,225,298,305,331,355,372,394,429,446,483,518
,549,576,594,630,655,674,692,730,770]
for i in range(1, len(clip)):
	for j in range(clip[i]-clip[i-1]):
		y.append(i)
y = np.array(y)
data = (x, y)

# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = gan.define_discriminator(in_shape=(28,28,3), n_classes=50)
#plot_model(discriminator, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
# create the generator
generator = gan.define_generator(latent_dim, n_classes=50)
#plot_model(generator, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
# create the gan
gan_model = gan.define_gan(generator, discriminator)

# load image data
dataset = gan.load_real_samples(data,x.shape[0])
# train model
gan.train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs=300, n_batch=64, n_classes=30)

#end = time.time()
#print("執行時間：%f 秒" % (end - start))
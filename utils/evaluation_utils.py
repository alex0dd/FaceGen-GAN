import tensorflow as tf
import numpy as np
import scipy as sp

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

class FidComputer:

    def __init__(self):
        self.inception_input_shape = (299, 299, 3)
        self.inception_input_size = (299, 299)
        self.inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=self.inception_input_shape)

    def get_inception_activations(self, images):
        images = np.copy(images)
        # assume data in [0, 1] range, and convert it to [-1, 1] range
        images = (images * 255).astype(np.uint8)
        images = preprocess_input(images)
        # assume data in [0, 1] range, and convert it to [-1, 1] range
        #images = 2 * images - 1
        images = tf.image.resize(images, self.inception_input_size, method=tf.image.ResizeMethod.BILINEAR)
        activations = self.inception_model.predict(images)
        return activations

    def compute_fid(self, activations_real, activations_gen):
        mu_real, sigma_real = activations_real.mean(axis=0), np.cov(activations_real, rowvar=False)
        mu_gen, sigma_gen = activations_gen.mean(axis=0), np.cov(activations_gen, rowvar=False)

        ssdiff = np.sum((mu_real - mu_gen) ** 2.0)
        covmean, _ = sp.linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)

        return fid
import tensorflow as tf
from utils.visualization_utils import save_plot_batch

class ImagesLoggingCallbackEMA(tf.keras.callbacks.Callback):

    def __init__(self, n_images, latent_dim, view_cond, real_view_conditions, images_dir):
        super(ImagesLoggingCallbackEMA, self).__init__()
        self.n_images = n_images
        self.latent_dim = latent_dim
        self.images_dir = images_dir
        self.view_cond = view_cond
        self.real_view_conditions = real_view_conditions

    def on_epoch_begin(self, epoch, logs=None):

        random_latent_vectors = tf.random.normal(shape=(self.n_images, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors, self.view_cond, training=False)
        generated_images_real_cond = self.model.generator(random_latent_vectors, self.real_view_conditions, training=False)
        generated_images_real_cond_ema = self.model.generator_ema(random_latent_vectors, self.real_view_conditions, training=False)
        generated_images = (generated_images + 1) / 2.0
        generated_images_real_cond = (generated_images_real_cond + 1) / 2.0
        generated_images_real_cond_ema = (generated_images_real_cond_ema + 1) / 2.0
        #generated_images *= 255
        generated_images.numpy()
        generated_images_real_cond.numpy()
        generated_images_real_cond_ema.numpy()

        save_plot_batch(generated_images, self.images_dir+"/sample/sample_{}.png".format(epoch))
        save_plot_batch(generated_images_real_cond, self.images_dir+"/real_cond/sample_{}.png".format(epoch))
        save_plot_batch(generated_images_real_cond_ema, self.images_dir+"/real_cond_ema/sample_{}.png".format(epoch))


class ImagesLoggingCallback(tf.keras.callbacks.Callback):

    def __init__(self, n_images, latent_dim, view_cond, real_view_conditions, images_dir):
        super(ImagesLoggingCallback, self).__init__()
        self.n_images = n_images
        self.latent_dim = latent_dim
        self.images_dir = images_dir
        self.view_cond = view_cond
        self.real_view_conditions = real_view_conditions

    def on_epoch_begin(self, epoch, logs=None):

        random_latent_vectors = tf.random.normal(shape=(self.n_images, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors, self.view_cond, training=False)
        generated_images_real_cond = self.model.generator(random_latent_vectors, self.real_view_conditions, training=False)
        generated_images = (generated_images + 1) / 2.0
        generated_images_real_cond = (generated_images_real_cond + 1) / 2.0
        #generated_images *= 255
        generated_images.numpy()
        generated_images_real_cond.numpy()

        save_plot_batch(generated_images, self.images_dir+"/sample/sample_{}.png".format(epoch))
        save_plot_batch(generated_images_real_cond, self.images_dir+"/real_cond/sample_{}.png".format(epoch))
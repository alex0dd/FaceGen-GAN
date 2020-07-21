
import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self, latent_dim, filters=128, kernel_size=4):
        super(Generator, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim

        self.cond_dense1 = tf.keras.layers.Dense(4 * 4 * filters)#, input_shape=(40,))
        self.cond_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.cond_reshape1 = tf.keras.layers.Reshape((4, 4, filters))

        self.dense1 = tf.keras.layers.Dense(4 * 4 * filters)
        self.lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.reshape1 = tf.keras.layers.Reshape((4, 4, filters))
        # 4x4 -> 8x8
        self.block1_deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size, strides=(2, 2), 
            padding="same")
        self.block1_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        # 8x8 -> 16x16
        self.block2_deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size, strides=(2, 2), 
            padding="same")
        self.block2_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        # 16x16 -> 32x32
        self.block3_deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size, strides=(2, 2), 
            padding="same")
        self.block3_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        # 32x32 -> 64x64
        self.block4_deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size, strides=(2, 2), 
            padding="same")
        self.block4_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        # 64 x 64 x FILTERS -> 64 x 64 x 3
        self.block5_deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=kernel_size, strides=(1, 1), 
            padding="same", activation="tanh")

    @tf.function
    def call(self, z, conditions, training=False):

        cond_emb = self.cond_dense1(conditions)
        cond_emb = self.cond_lrelu1(cond_emb)
        cond_emb = self.cond_reshape1(cond_emb)

        x = self.dense1(z)
        x = self.lrelu1(x)
        x = self.reshape1(x)

        x = tf.concat([x, cond_emb], axis=-1)

        x = self.block1_deconv1(x)
        x = self.block1_lrelu1(x)

        x = self.block2_deconv1(x)
        x = self.block2_lrelu1(x)

        x = self.block3_deconv1(x)
        x = self.block3_lrelu1(x)

        x = self.block4_deconv1(x)
        x = self.block4_lrelu1(x)

        images = self.block5_deconv1(x)
        return images
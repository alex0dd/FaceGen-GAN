
import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self, latent_dim, filters=128, kernel_size=4):
        super(Generator, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim

        self.cond_mlp = tf.keras.models.Sequential([
            tf.keras.Input(shape=(40,)),
            tf.keras.layers.Dense(4 * 4 * filters),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            #tf.keras.layers.Reshape((4, 4, filters))
        ])

        self.latent_adapt_mlp = tf.keras.models.Sequential([
            tf.keras.layers.Dense(4 * 4 * filters),
            tf.keras.layers.LeakyReLU(alpha=0.2)
        ])

        self.post_adapt_dense = tf.keras.layers.Dense(4 * 4 * filters)
        #self.dense1 = tf.keras.layers.Dense(4 * 4 * filters)
        #self.lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        # Separate reshape
        self.reshape1 = tf.keras.layers.Reshape((4, 4, filters))
        self.relu = tf.keras.layers.ReLU()
        """
        # 4x4 -> 8x8
        self.block1_deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size, strides=(2, 2), 
            padding="same")
        # 8x8 -> 16x16
        self.block2_deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size, strides=(2, 2), 
            padding="same")
        # 16x16 -> 32x32
        self.block3_deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size, strides=(2, 2), 
            padding="same")
        # 32x32 -> 64x64
        self.block4_deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size, strides=(2, 2), 
            padding="same")
        # 64 x 64 x FILTERS -> 64 x 64 x 3
        self.block5_deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=kernel_size, strides=(1, 1), 
            padding="same", activation="tanh")
        """
        self.upsample_network = tf.keras.models.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(2, 2), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(2, 2), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(2, 2), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(2, 2), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=kernel_size, strides=(1, 1), padding="same", activation="tanh")
        ])

    @tf.function
    def call(self, inputs, training=False):

        z, conditions = inputs

        cond_adapted = self.cond_mlp(conditions)
        latent_adapted = self.latent_adapt_mlp(z)

        conc_adapted = tf.concat([latent_adapted, cond_adapted], axis=-1)

        x = self.post_adapt_dense(conc_adapted)
        x = self.reshape1(x)

        """
        x = self.block1_deconv1(x)
        x = self.relu(x)

        x = self.block2_deconv1(x)
        x = self.relu(x)

        x = self.block3_deconv1(x)
        x = self.relu(x)

        x = self.block4_deconv1(x)
        x = self.relu(x)

        images = self.block5_deconv1(x)
        """
        images = self.upsample_network(x)
        return images
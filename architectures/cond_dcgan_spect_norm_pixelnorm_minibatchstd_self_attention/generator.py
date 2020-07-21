
import tensorflow as tf

from architectures.custom_layers import SNConv2D, SNConv2DTranspose, MinibatchStdev, SelfAttention, PixelNormalization

class Generator(tf.keras.Model):
    def __init__(self, latent_dim, filters=128, kernel_size=4, cond_dim=40):
        super(Generator, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim

        # NOTE: filters * 8 == 512, so usually filters==64
        self.dense1 = tf.keras.layers.Dense(4 * 4 * filters * 8, input_shape=(latent_dim + cond_dim,))
        self.relu1 = tf.keras.layers.ReLU()
        self.reshape1 = tf.keras.layers.Reshape((4, 4, filters * 8))
        self.bn1 = PixelNormalization()
        # 4x4 -> 8x8
        self.block1_upscale = tf.keras.layers.UpSampling2D()
        self.block1_conv1 = SNConv2D(
            filters=filters * 4, kernel_size=kernel_size, strides=(1, 1), 
            padding="same", kernel_initializer="orthogonal")
        self.block1_bn = PixelNormalization()
        self.block1_relu1 = tf.keras.layers.ReLU()
        # 8x8 -> 16x16
        self.block2_upscale = tf.keras.layers.UpSampling2D()
        self.block2_conv1 = SNConv2D(
            filters=filters * 2, kernel_size=kernel_size, strides=(1, 1), 
            padding="same", kernel_initializer="orthogonal")
        self.block2_bn = PixelNormalization()
        self.block2_relu1 = tf.keras.layers.ReLU()
        # 16x16 -> 32x32
        self.block3_upscale = tf.keras.layers.UpSampling2D()
        self.block3_conv1 = SNConv2D(
            filters=filters, kernel_size=kernel_size, strides=(1, 1), 
            padding="same", kernel_initializer="orthogonal")
        self.block3_bn = PixelNormalization()
        self.block3_relu1 = tf.keras.layers.ReLU()
        # attention layer
        self.attention = SelfAttention(filters, dtype=tf.float32)
        # 32x32 -> 64x64
        self.block4_upscale = tf.keras.layers.UpSampling2D()
        self.block4_conv1 = SNConv2D(
            filters=filters // 2, kernel_size=kernel_size, strides=(1, 1), 
            padding="same", kernel_initializer="orthogonal")
        self.block4_bn = PixelNormalization()
        self.block4_relu1 = tf.keras.layers.ReLU()
        # 64 x 64 x FILTERS -> 64 x 64 x 3
        self.block5_conv1 = SNConv2D(
            filters=3, kernel_size=4, strides=(1, 1), 
            padding="same", activation="tanh", kernel_initializer="orthogonal")

    @tf.function
    def call(self, z, conditions, training=False):

        x = tf.concat([z, conditions], axis=-1)

        x = self.dense1(x)
        x = self.relu1(x)
        x = self.reshape1(x)
        x = self.bn1(x, training=training)

        x = self.block1_upscale(x)
        x = self.block1_conv1(x)
        x = self.block1_bn(x, training=training)
        x = self.block1_relu1(x)

        x = self.block2_upscale(x)
        x = self.block2_conv1(x)
        x = self.block2_bn(x, training=training)
        x = self.block2_relu1(x)

        x = self.block3_upscale(x)
        x = self.block3_conv1(x)
        x = self.block3_bn(x, training=training)
        x = self.block3_relu1(x)

        x = self.attention(x)

        x = self.block4_upscale(x)
        x = self.block4_conv1(x)
        x = self.block4_bn(x, training=training)
        x = self.block4_relu1(x)

        images = self.block5_conv1(x)
        return images
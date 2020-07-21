import tensorflow as tf

from architectures.custom_layers import SNConv2D, SNConv2DTranspose, MinibatchStdev, SelfAttention

class Discriminator(tf.keras.Model):
    def __init__(self, filters=128, kernel_size=3, n_classes=40):
        super(Discriminator, self).__init__()

        self.filters = filters

        # 64 x 64 x FILTERS
        self.block1_conv1 = SNConv2D(filters, kernel_size=kernel_size, strides=(2, 2), padding="same", kernel_initializer="orthogonal")
        self.block1_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.block1_attention = SelfAttention(filters, dtype=tf.float32)
        self.block1_minibatch_std = MinibatchStdev()

        # 32 x 32 x FILTERS
        self.block2_conv1 = SNConv2D(filters * 2, kernel_size=kernel_size, strides=(2, 2), padding="same", kernel_initializer="orthogonal")
        self.block2_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.1)

        # 16 x 16 x FILTERS
        self.block3_conv1 = SNConv2D(filters * 4, kernel_size=kernel_size, strides=(2, 2), padding="same", kernel_initializer="orthogonal")
        self.block3_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.1)

        # 8 x 8 x FILTERS
        self.block4_conv1 = SNConv2D(filters * 8, kernel_size=kernel_size, strides=(2, 2), padding="same", kernel_initializer="orthogonal")
        self.block4_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.1)

        # Current size: 4 x 4 x FILTERS 
        self.avg_pool = tf.keras.layers.GlobalAvgPool2D()

        self.scoring = tf.keras.layers.Dense(1)
        self.classifier = tf.keras.layers.Dense(n_classes, activation="sigmoid")
    
    @tf.function
    def call(self, images, training=False):
        x = images

        x = self.block1_conv1(x)
        x = self.block1_lrelu1(x)
        x = self.block1_attention(x)
        x = self.block1_minibatch_std(x)

        x = self.block2_conv1(x)
        x = self.block2_lrelu1(x)

        x = self.block3_conv1(x)
        x = self.block3_lrelu1(x)

        x = self.block4_conv1(x)
        x = self.block4_lrelu1(x)

        x = self.avg_pool(x)

        scores = self.scoring(x)
        classes = self.classifier(x)

        return scores, classes
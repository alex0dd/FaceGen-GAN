import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self, filters=128, kernel_size=5):
        super(Discriminator, self).__init__()

        self.filters = filters

        self.cond_dense1 = tf.keras.layers.Dense(64 * 64 * 3)#, input_shape=(40,))
        self.cond_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.cond_reshape1 = tf.keras.layers.Reshape((64, 64, 3))

        # 64 x 64 x FILTERS
        self.block1_conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding="same")
        self.block1_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        # 32 x 32 x FILTERS
        self.block2_conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same")
        self.block2_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        # 16 x 16 x FILTERS
        self.block3_conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same")
        self.block3_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        # 8 x 8 x FILTERS
        self.block4_conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same")
        self.block4_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        # 4 x 4 x FILTERS
        self.block5_conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same")
        self.block5_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.scoring = tf.keras.layers.Dense(1, activation="sigmoid")
    
    @tf.function
    def call(self, images, conditions, training=False):

        #print(conditions)
        
        x = images

        cond_emb = self.cond_dense1(conditions)
        cond_emb = self.cond_lrelu1(cond_emb)
        cond_emb = self.cond_reshape1(cond_emb)

        x = tf.concat([x, cond_emb], axis=-1)

        x = self.block1_conv1(x)
        x = self.block1_lrelu1(x)

        x = self.block2_conv1(x)
        x = self.block2_lrelu1(x)

        x = self.block3_conv1(x)
        x = self.block3_lrelu1(x)

        x = self.block4_conv1(x)
        x = self.block4_lrelu1(x)

        x = self.block5_conv1(x)
        x = self.block5_lrelu1(x)

        x = self.flatten(x)
        x = self.dropout(x, training=training)
        scores = self.scoring(x)

        return scores
import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self, filters=128, kernel_size=5):
        super(Discriminator, self).__init__()

        self.filters = filters

        self.cond_mlp = tf.keras.models.Sequential([
            tf.keras.Input(shape=(40,)),
            tf.keras.layers.Dense(filters),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])
        #self.cond_dense1 = tf.keras.layers.Dense(64 * 64 * 3, input_shape=(40,))
        #self.cond_lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        #self.cond_reshape1 = tf.keras.layers.Reshape((64, 64, 3))

        """
        self.relu = tf.keras.layers.ReLU()
        # 64 x 64 x FILTERS
        self.block1_conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding="same")

        # 32 x 32 x FILTERS
        self.block2_conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same")

        # 16 x 16 x FILTERS
        self.block3_conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same")

        # 8 x 8 x FILTERS
        self.block4_conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same")

        # 4 x 4 x FILTERS
        self.block5_conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same")
        """
        self.downsample_network = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same"),
            tf.keras.layers.ReLU()
        ])

        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.scoring = tf.keras.layers.Dense(1)
    
    @tf.function
    def call(self, inputs, training=False):

        images, conditions = inputs
        
        x = images

        """
        x = self.block1_conv1(x)
        x = self.relu(x)

        x = self.block2_conv1(x)
        x = self.relu(x)

        x = self.block3_conv1(x)
        x = self.relu(x)

        x = self.block4_conv1(x)
        x = self.relu(x)

        x = self.block5_conv1(x)
        x = self.relu(x)
        """
        x = self.downsample_network(x)

        latent_repr = self.flatten(x)

        # concat embeddings with latent representation
        cond_emb = self.cond_mlp(conditions)
        latent_with_cond = tf.concat([latent_repr, cond_emb], axis=-1)
        latent_with_cond = self.dropout(latent_with_cond, training=training)
        # classify
        scores = self.scoring(latent_with_cond)

        return scores
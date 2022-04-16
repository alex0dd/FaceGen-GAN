import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self, condition_dim, filters=128, kernel_size=5):
        super(Discriminator, self).__init__()

        self.filters = filters
        self.condition_dim = condition_dim

        # TODO: add dimensionality check for > 0
        self.cond_mlp = tf.keras.models.Sequential([
            tf.keras.Input(shape=(condition_dim,)),
            tf.keras.layers.Dense(filters),
            tf.keras.layers.ReLU()
        ])

        """
        # 64 x 64 x FILTERS
        # 32 x 32 x FILTERS
        # 16 x 16 x FILTERS
        # 8 x 8 x FILTERS
        # 4 x 4 x FILTERS
        """
        self.downsample_network = tf.keras.models.Sequential([
            #tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding="same"),
            #tf.keras.layers.ReLU(),
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

        x = self.downsample_network(x)

        latent_repr = self.flatten(x)

        # concat embeddings with latent representation
        cond_emb = self.cond_mlp(conditions)
        latent_with_cond = tf.concat([latent_repr, cond_emb], axis=-1)
        latent_with_cond = self.dropout(latent_with_cond, training=training)
        # classify
        scores = self.scoring(latent_with_cond)

        return scores
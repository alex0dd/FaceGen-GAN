import tensorflow as tf

class GAN_Wrapper(tf.keras.Model):

    def __init__(self, discriminator, generator, **kwargs):
        super(GAN_Wrapper, self).__init__(**kwargs)

        self.discriminator = discriminator
        self.generator = generator

        self.latent_dim = self.generator.latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN_Wrapper, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def call(self, data, training=False):
        #z, conditions = data
        #return self.generator(z, conditions, training=training)
        pass

    #@tf.function
    def train_step(self, data):
        # conditions are ignored
        (real_images, conditions), _ = data

        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors, conditions, training=True)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)
        combined_conds = tf.concat([conditions, conditions], axis=0)

        # NOTE: THIS LINE IS IMPORTANT, AS TENSORFLOW CAN'T FIGURE OUT THE SHAPE
        combined_conds = tf.reshape(combined_conds, (batch_size * 2, 40))

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, combined_conds, training=True)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors, conditions), conditions, training=False)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss, "D(G(z))": tf.reduce_mean(predictions)}
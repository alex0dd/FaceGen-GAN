import tensorflow as tf

class GAN_Wrapper(tf.keras.Model):

    def __init__(self, discriminator, generator, **kwargs):
        super(GAN_Wrapper, self).__init__(**kwargs)

        self.discriminator = discriminator
        self.generator = generator

        self.latent_dim = self.generator.latent_dim

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(GAN_Wrapper, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_classifier_loss = tf.keras.losses.BinaryCrossentropy()
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def call(self, x, training=False):
        """
        This method is overridden only because it is required by tf.keras.Model
        """
        pass

    @tf.function
    def train_step(self, data):
        # conditions are ignored
        (real_images, conditions), _ = data

        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]

        # NOTE: THIS LINE IS IMPORTANT, AS TENSORFLOW CAN'T FIGURE OUT THE SHAPE
        conditions = tf.reshape(conditions, (batch_size, 40))

        # Train the discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images from the latent vector
            fake_images = self.generator(random_latent_vectors, conditions, training=False)
            # Get the logits for the fake images
            fake_logits, fake_classes = self.discriminator(fake_images, training=True)
            # Get the logits for real images
            real_logits, real_classes = self.discriminator(real_images, training=True)
            # Calculate discriminator loss using fake and real logits
            d_cost = self.d_loss_fn(logits_real=real_logits, logits_fake=fake_logits)
            d_class_real = self.d_classifier_loss(y_true=conditions, y_pred=real_classes)
            # Add the class loss to the original discriminator loss
            d_loss = d_cost + d_class_real

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.d_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train the generator
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, conditions, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits, gen_img_classes = self.discriminator(generated_images, training=False)
            # Calculate the generator loss
            g_cost = self.g_loss_fn(gen_img_logits)
            g_class_gen = self.d_classifier_loss(y_true=conditions, y_pred=gen_img_classes)
            g_loss = g_cost + g_class_gen
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {
            "d_loss": d_loss, 
            "g_loss": g_loss, 
            "D(G(z))": tf.reduce_mean(gen_img_logits), 
            "D(x)": tf.reduce_mean(real_logits)
        }

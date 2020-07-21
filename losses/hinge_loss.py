import tensorflow as tf

@tf.function
def discriminator_loss(logits_real, logits_fake):
    real_loss = tf.keras.backend.mean(tf.keras.backend.relu(1 - logits_real))
    fake_loss = tf.keras.backend.mean(tf.keras.backend.relu(1 + logits_fake))
    return (fake_loss + real_loss) / 2

@tf.function
def generator_loss(logits_fake):
    return -tf.keras.backend.mean(logits_fake)
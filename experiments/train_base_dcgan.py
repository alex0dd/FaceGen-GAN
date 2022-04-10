import tensorflow as tf

from architectures.base_dcgan import Generator, Discriminator, GAN_Wrapper
from .base_experiment import *

# TRAINING PARAMETERS
model_name = "base_dcgan"
batch_size = 128#32
n_epochs = 100
# GENERATOR PARAMETERS
conditional_dim = 40
latent_dim = 128
filters_gen = 64
kernel_size_gen = 4
# DISCRIMINATOR PARAMETERS
filters_disc = 64
kernel_size_disc = 5
# TRAINING OPTIMIZER PARAMETERS
init_learning_rate_gen = 0.0002
init_learning_rate_disc = 0.0002
beta_1 = 0.5
# RESUME PARAMS
load_model = False
load_epoch = 0

# create models
generator_model = Generator(latent_dim, filters=filters_gen, kernel_size=kernel_size_gen)
discriminator_model = Discriminator(filters=filters_disc, kernel_size=kernel_size_disc)
# create gan wrapper model
gan_model = GAN_Wrapper(discriminator_model, generator_model)
# compile model
gan_model.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=init_learning_rate_disc, beta_1=beta_1),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=init_learning_rate_gen, beta_1=beta_1),
    loss_fn=tf.keras.losses.BinaryCrossentropy(),
)
# callbacks
train_callbacks = [
    #tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/'+model_name+'/model_{epoch}.h5')
]

import tensorflow as tf

from architectures.cond_dcgan_spect_norm_pixelnorm_minibatchstd_self_attention import Generator, Discriminator, GAN_Wrapper
from .base_experiment import *
from losses.hinge_loss import discriminator_loss, generator_loss

# TRAINING PARAMETERS
model_name = "hinge_dcgan_spect_norm_pixelnorm_minibatchstd_self_attention"
batch_size = 32
n_epochs = 100
n_steps_disc = 2
# GENERATOR PARAMETERS
latent_dim = 128
conditional_dim = 40
filters_gen = 64
kernel_size_gen = 3
# DISCRIMINATOR PARAMETERS
filters_disc = 64
kernel_size_disc = 3
# TRAINING OPTIMIZER PARAMETERS
init_learning_rate_gen = 0.00005
init_learning_rate_disc = 0.0002
beta_1 = 0.0
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
    d_loss_fn=discriminator_loss,
    g_loss_fn=generator_loss,
)
# callbacks
train_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/'+model_name+'/model_{epoch}.h5')
]
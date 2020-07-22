import tensorflow as tf

from architectures.cond_dcgan_spect_norm_pixelnorm_minibatchstd_self_attention import Generator, Discriminator, GAN_WrapperEMA
from losses.hinge_loss import generator_loss, discriminator_loss

from .base_experiment import *


# TRAINING PARAMETERS
model_name = "hinge_dcgan_spect_norm_pixelnorm_minibatchstd_self_attention_ema"
batch_size = 32
n_epochs = 100
latent_dim = 128 # encoder will be latentdim
conditional_dim = 40
n_steps_disc = 2
# GENERATOR PARAMETERS
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
# REPORT PARAMETERS
print_output_every = 200
# EMA params
ema_beta = 0.9999
ema_start = 10#5000
ema_every = 2

# create models
generator_model = Generator(latent_dim, filters=filters_gen, kernel_size=kernel_size_gen)
generator_model_ema = Generator(latent_dim, filters=filters_gen, kernel_size=kernel_size_gen)
discriminator_model = Discriminator(filters=filters_disc, kernel_size=kernel_size_disc)
# create gan wrapper model
gan_model = GAN_WrapperEMA(discriminator_model, generator_model, generator_model_ema, beta=ema_beta)
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

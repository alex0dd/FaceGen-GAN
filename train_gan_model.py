import os
import pickle

import numpy as np
import pandas as pd

import tensorflow as tf

from datasets.celeba.dataloader import DataSequence

from utils.file_utils import makedir_if_not_exists
from utils.visualization_utils import save_plot_batch

from callbacks import ImagesLoggingCallback

# Load an experiment
from experiments.train_base_dcgan import *
#from experiments.train_hinge_dcgan_spect_norm_pixelnorm_minibatchstd_self_attention import *

# AMP
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


# FILE PARAMETERS
model_save_dir = "saved_models/{}/".format(model_name)
model_images_save_base_dir = "gen/{}".format(model_name)
model_gen_sample_dir = "gen/{}/sample/".format(model_name)
model_gen_real_dir = "gen/{}/real_cond/".format(model_name)

# make model directories if they no exist
makedir_if_not_exists(model_save_dir)
makedir_if_not_exists(model_gen_sample_dir)
makedir_if_not_exists(model_gen_real_dir)

# prepare train data sequence
train_data_df = pd.read_csv(dataset_attr_file_train, sep="\s+")
training_generator = DataSequence(train_data_df, dataset_images_path,  batch_size=batch_size)

# take first batch of validation dataset for visual results report 
# (i.e. conditioned generation based on first batch conditions)
valid_cond_batch = DataSequence(train_data_df, dataset_images_path,  batch_size=batch_size, mode="valid")
_, real_view_conditions = next(iter(valid_cond_batch))
real_view_conditions = real_view_conditions[:25]

# take apart a batch for reconstruction
view_cond = np.zeros((25, conditional_dim), dtype=np.float32)
view_cond[:, 31] = 1.0 # all smile
view_cond = view_cond.astype(np.float32)

if load_model:
    # fit just for shape (no steps are performed)
    gan_model.fit(training_generator, epochs=1, steps_per_epoch=1)
    # load model's weights
    gan_model.load_weights(model_save_dir+"/model_{}.h5".format(load_epoch))
    # load optimizer's state
    with open('saved_optimizers/{}_d_optimizer_weights.pkl'.format(model_name), 'rb') as f:
        weights = pickle.load(f)
        # set manually
        for i in range(len(weights)):
            gan_model.d_optimizer.weights[i] = weights[i]
    with open('saved_optimizers/{}_g_optimizer_weights.pkl'.format(model_name), 'rb') as f:
        weights = pickle.load(f)
        # set manually
        for i in range(len(weights)):
            gan_model.g_optimizer.weights[i] = weights[i]
else:
    load_epoch = 0

train_callbacks.append(ImagesLoggingCallback(25, latent_dim, view_cond, real_view_conditions, model_images_save_base_dir))

# Train the model
"""
history = gan_model.fit(training_generator,
    use_multiprocessing=True,
    workers=8,
    epochs=n_epochs,
    callbacks=train_callbacks,
    initial_epoch=load_epoch
)
"""
history = gan_model.fit(training_generator,
    use_multiprocessing=True,
    workers=8,
    steps_per_epoch=30,
    callbacks=train_callbacks,
    initial_epoch=load_epoch
)

"""
# Save optimizer's state manually
import pickle
with open('saved_optimizers/{}_d_optimizer_weights.pkl'.format(model_name), 'wb') as f:
    pickle.dump(getattr(gan_model.d_optimizer, 'weights'), f)
with open('saved_optimizers/{}_g_optimizer_weights.pkl'.format(model_name), 'wb') as f:
    pickle.dump(getattr(gan_model.g_optimizer, 'weights'), f)

# gan_model.save('dcgan_spect_norm/model_dcgan') # should also save optimizer state
"""

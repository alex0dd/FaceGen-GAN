import os
import pickle

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# ----- CONFIGURE TENSORFLOW -----
# This step might be needed in case cuDNN
# gives problems with convolutions
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# --------------------------------

from datasets.celeba.dataloader import DataSequence

from utils.file_utils import makedir_if_not_exists
from utils.visualization_utils import save_plot_batch

from callbacks import ImagesLoggingCallback, ImagesLoggingCallbackEMA

# Load an experiment
#from experiments.train_base_dcgan import *
#from experiments.train_hinge_dcgan_spect_norm_pixelnorm_minibatchstd_self_attention import *
from experiments.train_hinge_dcgan_spect_norm_pixelnorm_minibatchstd_self_attention_ema import *

# FILE PARAMETERS
model_save_dir = "saved_models/{}/".format(model_name)
model_images_save_base_dir = "gen/{}".format(model_name)
model_gen_sample_dir = "gen/{}/sample/".format(model_name)
model_gen_real_dir = "gen/{}/real_cond/".format(model_name)

# TODO: check if is_ema
model_gen_real_ema_dir = "gen/{}/real_cond_ema/".format(model_name)

# make model directories if they no exist
makedir_if_not_exists(model_save_dir)
makedir_if_not_exists(model_gen_sample_dir)
makedir_if_not_exists(model_gen_real_dir)
# TODO: check if is_ema
makedir_if_not_exists(model_gen_real_ema_dir)

# prepare train data sequence
train_data_df = pd.read_csv(dataset_attr_file_train)
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

#train_callbacks.append(ImagesLoggingCallback(25, latent_dim, view_cond, real_view_conditions, model_images_save_base_dir))
# TODO: check if is_ema
image_save_callback = ImagesLoggingCallbackEMA(25, latent_dim, view_cond, real_view_conditions, model_images_save_base_dir)
train_callbacks.append(image_save_callback)

image_save_callback.model = gan_model

def update_dict(orig_dict, new_dict):
    for key in orig_dict.keys():
        orig_dict[key].append(new_dict[key])
    return orig_dict

def print_stats(epoch, step, stats_dict):
    print("Epoch: {}, Step: {}, {}".format(epoch, step, {key: tf.reduce_mean(value).numpy() for key, value in stats_dict.items()}))

# Train the model
for epoch in range(load_epoch, n_epochs):
    image_save_callback.on_epoch_begin(epoch)
    stats_dict = {"d_loss": [], "g_loss": [], "D(G(z))": [], "D(x)": []}
    for step, x in enumerate(training_generator):
        train_details = gan_model.train_step(x)
        update_dict(stats_dict, train_details)
        if gan_model.current_step > ema_start and gan_model.current_step % ema_every == 0:
            gan_model.ema_step()
        if step % print_output_every == 0:
            print_stats(epoch, step, stats_dict)

"""
# Save optimizer's state manually
import pickle
with open('saved_optimizers/{}_d_optimizer_weights.pkl'.format(model_name), 'wb') as f:
    pickle.dump(getattr(gan_model.d_optimizer, 'weights'), f)
with open('saved_optimizers/{}_g_optimizer_weights.pkl'.format(model_name), 'wb') as f:
    pickle.dump(getattr(gan_model.g_optimizer, 'weights'), f)

# gan_model.save('dcgan_spect_norm/model_dcgan') # should also save optimizer state
"""
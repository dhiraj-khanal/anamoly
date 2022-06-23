from functools import partial
import numpy as np
import pandas as pd
import os
import random
import time
import tensorflow as tf, re, math
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import gc
import uproot

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()

!nvidia-smi

batch_size = 32


def get_df(root_file_name, filter_name):
    events = uproot.open(root_file_name, filter_name=filter_name)["tree"]
    df = events.arrays(library="pd")
    return df

features = []
# variables: general
features += ['FatJet_pt', 'FatJet_eta', 'FatJet_phi', 'FatJet_DDX_jetNSecondaryVertices', \
             'FatJet_DDX_jetNTracks', 'FatJet_DDX_z_ratio', 'FatJet_Proba', 'FatJet_area', \
             'FatJet_jetId', 'FatJet_lsf3', 'FatJet_rawFactor', 'FatJet_n2b1', 'FatJet_n3b1', \
             ]

# variables: tau1
features += ['FatJet_tau1', 'FatJet_DDX_tau1_flightDistance2dSig', 'FatJet_DDX_tau1_trackEtaRel_0', \
             'FatJet_DDX_tau1_trackEtaRel_1', 'FatJet_DDX_tau1_trackEtaRel_2', 'FatJet_DDX_tau1_trackSip3dSig_0', \
             'FatJet_DDX_tau1_trackSip3dSig_1', 'FatJet_DDX_tau1_vertexDeltaR', 'FatJet_DDX_tau1_vertexEnergyRatio', \
             ]

# variables: tau2
features += ['FatJet_tau2', 'FatJet_DDX_tau2_flightDistance2dSig', 'FatJet_DDX_tau2_trackEtaRel_0', \
             'FatJet_DDX_tau2_trackEtaRel_1', 'FatJet_DDX_tau2_trackEtaRel_3', 'FatJet_DDX_tau2_trackSip3dSig_0', \
             'FatJet_DDX_tau2_trackSip3dSig_1', 'FatJet_DDX_tau2_vertexEnergyRatio', \
             ]

# variables: tau3 and tau4
features += ['FatJet_tau3', 'FatJet_tau4',]

# variables: track
features += ['FatJet_DDX_trackSip2dSigAboveBottom_0', 'FatJet_DDX_trackSip2dSigAboveBottom_1', \
             'FatJet_DDX_trackSip2dSigAboveCharm', 'FatJet_DDX_trackSip3dSig_0', \
             'FatJet_DDX_trackSip3dSig_1', 'FatJet_DDX_trackSip3dSig_2', 'FatJet_DDX_trackSip3dSig_3', \
             ]

# variables: subjet 1
features += ['FatJet_subjet1_pt', 'FatJet_subjet1_eta', 'FatJet_subjet1_phi', \
             'FatJet_subjet1_Proba', 'FatJet_subjet1_tau1', 'FatJet_subjet1_tau2', \
             'FatJet_subjet1_tau3', 'FatJet_subjet1_tau4', 'FatJet_subjet1_n2b1', 'FatJet_subjet1_n3b1', \
             ]

# variables: subjet 2
features += ['FatJet_subjet2_pt', 'FatJet_subjet2_eta', 'FatJet_subjet2_phi', \
             'FatJet_subjet2_Proba', 'FatJet_subjet2_tau1', 'FatJet_subjet2_tau2', \
             'FatJet_subjet2_tau3', 'FatJet_subjet2_tau4', 'FatJet_subjet2_n2b1', 'FatJet_subjet2_n3b1', \
             ]

# variables: fatjet sv
features += ['FatJet_sv_costhetasvpv', 'FatJet_sv_d3dsig', 'FatJet_sv_deltaR', 'FatJet_sv_dxysig', \
             'FatJet_sv_enration', 'FatJet_sv_normchi2', 'FatJet_sv_ntracks', 'FatJet_sv_phirel', \
             'FatJet_sv_pt', 'FatJet_sv_ptrel', \
             ]

features = sorted(features)

original_dim = len(features)

inputfile = '"/eos/user/d/dkhanal/data.root"'
df = get_df(inputfile, '*')

df.dropna(inplace=True)
df = df[features]

X = df.to_numpy()
X = X.astype("float32")

# Scale our data using a MinMaxScaler that will scale
# each number so that it will be between 0 and 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

x_train, x_test = train_test_split(X, test_size=0.20)

def build_dset(df):
    df = df.copy()
    dataset = tf.data.Dataset.from_tensor_slices((df, df))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

x_train_dataset = build_dset(x_train)
x_test_dataset = build_dset(x_test)

intermediate_dim_1 = 32
intermediate_dim_2 = 16
latent_dim = 8

with strategy.scope():
    model = get_autoencoder(original_dim, intermediate_dim_1, intermediate_dim_2, latent_dim)
    #model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss=tf.keras.losses.MeanSquaredError())
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
    model.summary()

def get_lr_callback():
    lr_start   = 0.000001
    lr_max     = 0.01
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 10
    lr_decay   = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
    return lr_callback

checkpoint_path = "weights.{epoch:05d}.hdf5"
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor = 'val_loss',
                                                 save_weights_only=True,
                                                 save_best_only=False,
                                                 mode = 'min',
                                                 verbose=1)

num_epochs = 20

history = model.fit(
    x_train_dataset,
    shuffle=True,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=x_test_dataset,
    callbacks=[cp_callback, get_lr_callback()]
)


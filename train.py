# Init wandb
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from callbacks import TrainCallback
from preprocessing import audio_array, tospec, splitcut
from models import Adverserial
from hparams import hparams as hp
from losses import adv_loss
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
import os
import numpy as np

# wandb.init(project="melgan-music-transfer")
wandb.init()

awv = audio_array('../adaIN/GTZAN/genres/rock')                               #get waveform array from folder containing wav files
aspec = tospec(awv)                                                                 #get spectrogram array
adata = splitcut(aspec)                                                             #split spectrogams to fixed length

bwv = audio_array('../adaIN/GTZAN/genres/reggae')
bspec = tospec(bwv)
bdata = splitcut(bspec)

def dataset_gen():
	for a, b in zip(adata, bdata):
		a_cropped = tf.image.random_crop(a, size=[hp.hop, 3*hp.shape, 1])
		b_cropped = tf.image.random_crop(b, size=[hp.hop, 3*hp.shape, 1])
		return a_cropped, b_cropped
		# yield {"content":a_cropped, "style":b_cropped}

@tf.function
def proc(x):
	print(x.shape)
	return tf.image.random_crop(x, size=[hp.hop, 3*hp.shape, 1])

dsa = tf.data.Dataset.from_tensor_slices(adata).repeat(50).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(hp.bs, drop_remainder=True)
dsb = tf.data.Dataset.from_tensor_slices(bdata).repeat(50).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(hp.bs, drop_remainder=True)

dataset = tf.data.Dataset.zip((dsa, dsb))

model = Adverserial()

gen_opt = Adam(ExponentialDecay(hp.lr, decay_steps=hp.epochs*2, decay_rate=0.1), hp.opt_alpha)
disc_opt = Adam(ExponentialDecay(hp.lr, decay_steps=hp.epochs*2, decay_rate=0.1), hp.opt_alpha)

model.compile(gen_opt, disc_opt, adv_loss)

# Log metrics with wandb
model.fit(dataset, epochs=hp.epochs, callbacks=[WandbCallback(), TrainCallback(model, aspec), EarlyStopping(monitor="loss_g", patience=50)]) 

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
import argparse


def dataset_gen():
	for a, b in zip(adata, bdata):
		a_cropped = tf.image.random_crop(a, size=[hp.hop, 3*hp.shape, 1])
		b_cropped = tf.image.random_crop(b, size=[hp.hop, 3*hp.shape, 1])
		return a_cropped, b_cropped

@tf.function
def proc(x, hp):
	return tf.image.random_crop(x, size=[hp.hop, 3*hp.shape, 1])


def train(args):
	awv = audio_array(args.content)                			 		               #get waveform array from folder containing wav files
	aspec = tospec(awv)                                                                 #get spectrogram array
	adata = splitcut(aspec)                                                             #split spectrogams to fixed length

	bwv = audio_array(args.style)
	bspec = tospec(bwv)
	bdata = splitcut(bspec)

	dsa = tf.data.Dataset.from_tensor_slices(adata).repeat(50).map(lambda x: proc(x, hp), num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(hp.bs, drop_remainder=True)
	dsb = tf.data.Dataset.from_tensor_slices(bdata).repeat(50).map(lambda x: proc(x, hp), num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(hp.bs, drop_remainder=True)

	dataset = tf.data.Dataset.zip((dsa, dsb))

	model = Adverserial()

	gen_opt = Adam(ExponentialDecay(hp.lr, decay_steps=hp.epochs*2, decay_rate=0.1), hp.opt_alpha)
	disc_opt = Adam(ExponentialDecay(hp.lr, decay_steps=hp.epochs*2, decay_rate=0.1), hp.opt_alpha)

	model.compile(gen_opt, disc_opt, adv_loss)

	# Log metrics with wandb
	callbacks = [TrainCallback(model, aspec), EarlyStopping(monitor="loss_g", patience=50)]
	model.fit(dataset, epochs=hp.epochs, callbacks=callbacks) 


def main():
	parser = argparse.ArgumentParser(description='Train a music style transfer model')
	parser.add_argument('content', type=str, help='the directory of the content audio files')
	parser.add_argument('style', type=str, help='the directory of the style audio files')

	args = parser.parse_args()

	train(args)

if __name__ == '__main__':
	main()
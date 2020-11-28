import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from preprocessing import *
from models import Generator
from hparams import hparams as hp

def synthesize(audio_path, checkpoint_path):

	gen = Generator((hp.hop, hp.shape, 1), use_lstm=hp.use_lstm)
	gen.compile()
	test_input = np.zeros((1, hp.hop, hp.shape, 1))
	gen(test_input)
	gen.load_weights("{}/gen.h5".format(checkpoint_path))

	#Wav to wav conversion
	# wv, sr = librosa.load("../adaIN/GTZAN/genres/rock/rock.00005.wav", sr=sr)
	wv, sr = librosa.load(audio_path, sr=hp.sr)
	wv = wv[7*hp.sr:27*hp.sr]
	print(wv.shape)
	wv = np.expand_dims(wv, axis=1)
	speca = prep(wv)                                                    #Waveform to Spectrogram
	print(speca.shape)

	plt.figure(figsize=(50,1))                                          #Show Spectrogram
	plt.imshow(np.flip(speca, axis=0), cmap=None)
	plt.axis('off')
	plt.show()

	abwv = towave(speca, name='test', gen=gen, path='.')           #Convert and save wav

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('audio_path', type=str)
	parser.add_argument('checkpoint_path', type=str)
	args = parser.parse_args()

	synthesize(args.audio_path, args.checkpoint_path)
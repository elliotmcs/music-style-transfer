# music-style-transfer

Based off MelGAN-VC by Marco Pasini ([github](https://github.com/marcoppasini/MelGAN-VC) | [paper](https://arxiv.org/abs/1910.03713))

## Installation

Install python requirements:
	virtualenv venv
	source venv/bin/activate
	pip3 install -r requirements.txt

## Training

Dataset:
The recommend dataset for this project is GTZAN available: http://marsyas.info/downloads/datasets.html
All audio files in the training data must be converted to .wav files before training.

Run train script:
	python3 train.py DIRECTORY_CONTAINING_CONTENT_AUDIO_FILES DIRECTORY_CONTAINING_STYLE_AUDIO_FILES
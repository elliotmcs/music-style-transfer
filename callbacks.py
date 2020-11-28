import matplotlib.pyplot as plt
from hparams import hparams as hp
import numpy as np
from preprocessing import *
import soundfile as sf
import os

from tensorflow.keras.callbacks import Callback


#Generate a random batch to display current training results
def testgena(aspec):
    sw = True
    while sw:
        a = np.random.choice(aspec)
        if a.shape[1]//hp.shape!=1:
            sw=False
    dsa = []
    if a.shape[1]//hp.shape>6:
        num=6
    else:
        num=a.shape[1]//hp.shape
    rn = np.random.randint(a.shape[1]-(num*hp.shape))
    for i in range(num):
        im = a[:,rn+(i*hp.shape):rn+(i*hp.shape)+hp.shape]
        im = np.reshape(im, (im.shape[0],im.shape[1],1))
        dsa.append(im)
    return np.array(dsa, dtype=np.float32)

#Show results mid-training
def save_test_image_full(path, model, aspec):
    a = testgena(aspec)
    ab = model.gen(a, training=False)
    ab = testass(ab)
    a = testass(a)
    abwv = deprep(ab)
    awv = deprep(a)
    sf.write(path+'/new_file.wav', abwv, hp.sr)
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(np.flip(a, -2), cmap=None)
    axs[0].axis('off')
    axs[0].set_title('Source')
    axs[1].imshow(np.flip(ab, -2), cmap=None)
    axs[1].axis('off')
    axs[1].set_title('Generated')
    plt.savefig("test_out.png")



class TrainCallback(Callback):
    def __init__(self, model, aspec):
        super(TrainCallback, self).__init__()
        self.model=model
        self.aspec = aspec
    def on_epoch_end(self, epoch, logs=None):
        if epoch % hp.n_save == 0:
            print("Saving model...")
            gloss = logs['loss_g']
            dloss = logs['loss_d']
            path = f'{hp.save_path}/latest'
            if not os.path.isdir(path):
                os.mkdir(path)
            self.model.gen.save_weights(path+'/gen.h5')
            self.model.disc.save_weights(path+'/critic.h5')
            self.model.siam.save_weights(path+'/siam.h5')
            # save_test_image_full(path, self.model, self.aspec)

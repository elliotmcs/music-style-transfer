import tensorflow as tf
from hparams import hparams as hp

def mae(x,y):
    return tf.reduce_mean(tf.abs(x-y))

def mse(x,y):
    return tf.reduce_mean((x-y)**2)

def loss_travel(sa,sab,sa1,sab1):
    l1 = tf.reduce_mean(((sa-sa1) - (sab-sab1))**2)
    l2 = tf.reduce_mean(tf.reduce_sum(-(tf.nn.l2_normalize(sa-sa1, axis=[-1]) * tf.nn.l2_normalize(sab-sab1, axis=[-1])), axis=-1))
    return l1+l2

def loss_siamese(sa,sa1):
    logits = tf.sqrt(tf.reduce_sum((sa-sa1)**2, axis=-1, keepdims=True))
    return tf.reduce_mean(tf.square(tf.maximum((hp.delta - logits), 0)))

def d_loss_f(fake):
    return tf.reduce_mean(tf.maximum(1 + fake, 0))

def d_loss_r(real):
    return tf.reduce_mean(tf.maximum(1 - real, 0))

def g_loss_f(fake):
    return tf.reduce_mean(- fake)

def adv_loss(bb, bb2, bb3, fid, fid2, fid3, cab, cb, sa, sa2, sab, sab2):
    #identity mapping loss
    loss_id = (mae(bb,fid)+mae(bb2,fid2)+mae(bb3,fid3))/3.                                                 #loss_id = 0. IF THE IDENTITY LOSS TERM IS NOT NEEDED
    #travel loss
    loss_m = loss_travel(sa,sab,sa2,sab2)+loss_siamese(sa,sa2)
    #generator and critic losses
    loss_g = g_loss_f(cab)
    loss_dr = d_loss_r(cb)
    loss_df = d_loss_f(cab)
    loss_d = (loss_dr+loss_df)/2.
    return loss_id, loss_m, loss_g, loss_d
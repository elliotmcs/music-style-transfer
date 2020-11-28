from custom_layers import *
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, BatchNormalization, UpSampling2D, Concatenate, Cropping2D, ZeroPadding2D, Flatten, LSTM
from tensorflow.keras import Model
from hparams import hparams as hp

init = tf.keras.initializers.he_uniform()

def conv2d(layer_input, filters, kernel_size=4, strides=2, padding='same', leaky=True, bnorm=True, sn=True):
    if leaky:
        Activ = LeakyReLU(alpha=0.2)
    else:
        Activ = ReLU()
    if sn:
        d = ConvSN2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=init, use_bias=False)(layer_input)
    else:
        d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=init, use_bias=False)(layer_input)
    if bnorm:
        d = BatchNormalization()(d)
    d = Activ(d)
    return d

def deconv2d(layer_input, layer_res, filters, kernel_size=4, conc=True, scalev=False, bnorm=True, up=True, padding='same', strides=2):
    if up:
        u = UpSampling2D((1,2))(layer_input)
        u = ConvSN2D(filters, kernel_size, strides=(1,1), kernel_initializer=init, use_bias=False, padding=padding)(u)
    else:
        u = ConvSN2DTranspose(filters, kernel_size, strides=strides, kernel_initializer=init, use_bias=False, padding=padding)(layer_input)
    if bnorm:
        u = BatchNormalization()(u)
    u = LeakyReLU(alpha=0.2)(u)
    if conc:
        u = Concatenate()([u,layer_res])
    return u

#Extract function: splitting spectrograms
def extract_image(im):
    im1 = Cropping2D(((0,0), (0, 2*(im.shape[2]//3))))(im)
    im2 = Cropping2D(((0,0), (im.shape[2]//3,im.shape[2]//3)))(im)
    im3 = Cropping2D(((0,0), (2*(im.shape[2]//3), 0)))(im)
    return im1,im2,im3

#Assemble function: concatenating spectrograms
def assemble_image(lsim):
    im1,im2,im3 = lsim
    imh = Concatenate(2)([im1,im2,im3])
    return imh

class Generator(Model):
    def __init__(self, input_shape, use_lstm=False):
        super(Generator, self).__init__()
        h,w,c = input_shape
        self.use_lstm = use_lstm
        self.conv0_0 = ConvSN2D(hp.gen_units, kernel_size=(h,3), strides=1, padding='valid', kernel_initializer=init, use_bias=False)
        self.bn0_0 = BatchNormalization()

        self.conv0_1 = ConvSN2D(hp.gen_units, kernel_size=(1,9), strides=(1,2), padding='same', kernel_initializer=init, use_bias=False)
        self.bn0_1 = BatchNormalization()

        self.conv0_2 =ConvSN2D(hp.gen_units, kernel_size=(1,7), strides=(1,2), padding='same', kernel_initializer=init, use_bias=False)
        self.bn0_2 = BatchNormalization()
        
        if use_lstm:
            self.lstm = LSTM(hp.gen_units, return_sequences=True)

        self.conv1_0 = ConvSN2D(hp.gen_units, kernel_size=(1,7), strides=(1,1), padding='same', kernel_initializer=init, use_bias=False)
        self.bn1_0 = BatchNormalization()

        self.conv1_1 = ConvSN2D(hp.gen_units, kernel_size=(1,9), strides=(1,1), padding='same', kernel_initializer=init, use_bias=False)
        self.conv1_T = ConvSN2DTranspose(1, kernel_size=(h,1), strides=(1,1), kernel_initializer=init, padding='valid', activation='tanh')


    def call(self, x):
        x0 = ZeroPadding2D((0,1))(x)
        # x1 = ConvSN2D(256, kernel_size=(x.shape[1],3), strides=1, padding='valid', kernel_initializer=init, use_bias=False)(x0)
        x1 = self.conv0_0(x0)
        x1 = self.bn0_0(x1)
        x1 = LeakyReLU()(x1)

        x2 = self.conv0_1(x1)
        x2 = self.bn0_1(x2)
        x2 = LeakyReLU()(x2)
        
        x3 = self.conv0_2(x2)
        x3 = self.bn0_2(x3)
        x3 = LeakyReLU()(x3)

        if self.use_lstm:
            x3 = tf.reshape(x3, shape=(-1, x3.shape[2], x3.shape[3]))
            x3 = self.lstm(x3)
            x3 = tf.expand_dims(x3, axis=1)

        x4 = UpSampling2D((1,2))(x3)
        x4 = self.conv1_0(x4)
        x4 = self.bn1_0(x4)
        x4 = LeakyReLU()(x4)
        x4 = Concatenate()([x4,x2])

        x5 = UpSampling2D((1,2))(x4)
        x5 = self.conv1_1(x5)
        x5 = LeakyReLU()(x5)
        x5 = Concatenate()([x5,x1])
        
        # x6 = ConvSN2DTranspose(1, kernel_size=(x.shape[1],1), strides=(1,1), kernel_initializer=init, padding='valid', activation='tanh')(x5)
        x6 = self.conv1_T(x5)
        return x6

class Siamese(Model):
    def __init__(self, input_shape):
        super(Siamese, self).__init__()
        h,w,c = input_shape
        self.conv0_0 = Conv2D(hp.siam_units, kernel_size=(h,3), strides=1, padding='valid', kernel_initializer=init, use_bias=False)
        self.bn0_0 = BatchNormalization()

        self.conv0_1 = Conv2D(hp.siam_units, kernel_size=(1,9), strides=(1,2), padding='same', kernel_initializer=init, use_bias=False)
        self.bn0_1 = BatchNormalization()

        self.conv0_2 =Conv2D(hp.siam_units, kernel_size=(1,7), strides=(1,2), padding='same', kernel_initializer=init, use_bias=False)
        self.bn0_2 = BatchNormalization()

        self.fc = Dense(hp.vec_len)

    def call(self, x):
        # x = Conv2D(256, kernel_size=(x.shape[1],3), strides=1, padding='valid', kernel_initializer=init, use_bias=False)(x)
        x = self.conv0_0(x)
        x = self.bn0_0(x)
        x = LeakyReLU()(x)

        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = LeakyReLU()(x)
        
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = self.fc(x)
        return x

class Discriminator(Model):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        h,w,c = input_shape
        self.conv0_0 = ConvSN2D(hp.disc_units, kernel_size=(h,3), strides=1, padding='valid', kernel_initializer=init, use_bias=False)
        self.bn0_0 = BatchNormalization()

        self.conv0_1 = ConvSN2D(hp.disc_units, kernel_size=(1,9), strides=(1,2), padding='same', kernel_initializer=init, use_bias=False)

        self.conv0_2 =ConvSN2D(hp.disc_units, kernel_size=(1,7), strides=(1,2), padding='same', kernel_initializer=init, use_bias=False)

        self.fc = DenseSN(1, kernel_initializer=init)

    def call(self, x):
        # x = ConvSN2D(512, kernel_size=(x.shape[1],3), strides=1, padding='valid', kernel_initializer=init, use_bias=False)(x)
        x = self.conv0_0(x)
        x = self.bn0_0(x)
        x = LeakyReLU()(x)

        x = self.conv0_1(x)
        x = LeakyReLU()(x)
        
        x = self.conv0_2(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = self.fc(x)
        return x

class Adverserial(Model):
    def __init__(self):
        super(Adverserial, self).__init__()
        self.gen = Generator((hp.hop, hp.shape, 1), use_lstm=hp.use_lstm)
        self.siam = Siamese((hp.hop, hp.shape, 1))
        self.disc = Discriminator((hp.hop, 3*hp.shape, 1))
        
    def compile(self, gen_opt, disc_opt, loss_fn):
        super(Adverserial, self).compile()
        self.gen_opt = gen_opt
        self.disc_opt = disc_opt
        self.loss_fn = loss_fn

    def train_step(self, x):
        if isinstance(x, tuple):
            a = x[0]
            b = x[1]
        else:
            raise "Input must be a 2-tuple"
        aa,aa2,aa3 = extract_image(a) 
        bb,bb2,bb3 = extract_image(b)
        with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:

            #translating A to B
            fab = self.gen(aa, training=True)
            fab2 = self.gen(aa2, training=True)
            fab3 = self.gen(aa3, training=True)
            #identity mapping B to B                                                                                                                COMMENT THESE 3 LINES IF THE IDENTITY LOSS TERM IS NOT NEEDED
            fid = self.gen(bb, training=True) 
            fid2 = self.gen(bb2, training=True)
            fid3 = self.gen(bb3, training=True)
            #concatenate/assemble converted spectrograms
            fabtot = assemble_image([fab,fab2,fab3])

            #feed concatenated spectrograms to critic
            cab = self.disc(fabtot, training=True)
            cb = self.disc(b, training=True)
            #feed 2 pairs (A,G(A)) extracted spectrograms to Siamese
            sab = self.siam(fab, training=True)
            sab2 = self.siam(fab3, training=True)
            sa = self.siam(aa, training=True)
            sa2 = self.siam(aa3, training=True)

            loss_id, loss_m, loss_g, loss_d = self.loss_fn(bb, bb2, bb3, fid, fid2, fid3, cab, cb, sa, sa2, sab, sab2)

            #identity mapping loss
            # loss_id = (mae(bb,fid)+mae(bb2,fid2)+mae(bb3,fid3))/3.                                                 #loss_id = 0. IF THE IDENTITY LOSS TERM IS NOT NEEDED
            #travel loss
            # loss_m = loss_travel(sa,sab,sa2,sab2)+loss_siamese(sa,sa2)
            #generator and critic losses
            # loss_g = g_loss_f(cab)
            # loss_dr = d_loss_r(cb)
            # loss_df = d_loss_f(cab)
            # loss_d = (loss_dr+loss_df)/2.
            #generator+siamese total loss
            if not hp.use_id:
                loss_id=0
            lossgtot = loss_g+hp.gloss_delta*loss_m+hp.mloss_delta*loss_id*hp.idloss_delta
        
        #computing and applying gradients
        grad_gen = tape_gen.gradient(lossgtot, self.gen.trainable_variables+self.siam.trainable_variables)
        self.gen_opt.apply_gradients(zip(grad_gen, self.gen.trainable_variables+self.siam.trainable_variables))

        grad_disc = tape_disc.gradient(loss_d, self.disc.trainable_variables)
        self.disc_opt.apply_gradients(zip(grad_disc, self.disc.trainable_variables))
        
        # return {"loss_dr":loss_dr,"loss_df":loss_df,"loss_g":loss_g,"loss_id":loss_id}
        return {"loss_g":lossgtot, "loss_d":loss_d}

    def call(self, x, training=True):
        # a = x["content"]
        # b = x["style"]

        aa,aa2,aa3 = extract_image(x)
        # bb,bb2,bb3 = extract_image(b)

        fab = self.gen(aa, training=training)
        fab2 = self.gen(aa2, training=training)
        fab3 = self.gen(aa3, training=training)        
        fabtot = assemble_image([fab, fab2, fab3])

        return fabtot
        # cab = self.disc(fabtot, training=training)
        # cb = self.disc(b, training=training)

        # sab = self.siam(fab, training=training)
        # sab2 = self.siam(fab3, training=training)
        # sa = self.siam(aa, training=training)
        # sa2 = self.siam(aa3, training=training)

        # gen_out = self.gen(x)
        # siam_out = self.siam(x)
        # disc_out = self.disc(x)
        # return {"gen_out":gen_out, "siam_out":siam_out, "disc_out":disc_out}
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow.keras as keras

tf.compat.v1.disable_eager_execution()

INPUT_DIM = (64,64,3)
IM_DIM = 64

# VAEGAN ENC
CONV_FILTERS = [32,64,64, 128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

#CONV_T_FILTERS = [64,64,32,3]
#CONV_T_KERNEL_SIZES = [5,5,6,6]
#CONV_T_STRIDES = [2,2,2,2]
#CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

DEPTH = 32
LATENT_DEPTH = 512
K_SIZE = 5

# VAEGAN GEN
CONV_G_FILTERS = [128, 64, 64, 32]
CONV_G_KERNEL_SIZES = [4, 4, 4, 4]
CONV_G_STRIDES = [2, 2, 2, 2]
CONV_G_ACTIVATIONS = ['relu', 'relu', 'relu', 'relu']

# LATENT VARIABLE SIZE
Z_DIM = 32

BATCH_SIZE = 100 
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5

INNER_LOSS_COEF = 1
NORMAL_COEF = 0.1
KL_COEF = 0.01

class Sampling(Layer):
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * epsilon


class VAEGANModel(Model):
    def __init__(self, enc, gen, disc, r_loss_factor, opt, **kwargs):
        super(VAEGANModel, self).__init__(**kwargs)
        self.E = enc
        self.G = gen
        self.D = disc
        self.r_loss_factor = r_loss_factor
        self.opti = opt

    #@tf.function
    def train_step(self, data):
        latent_r = tf.random.normal((BATCH_SIZE, LATENT_DEPTH))
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape(persistent=True) as tape:
            latent, kl_loss = self.E(data)
            fake = self.G(latent)

            dis_fake, dis_inner_fake = self.D(fake)
            dis_fake_r, _ = self.D(self.G(latent_r))
            dis_true, dis_inner_true = self.D(data)

            vae_inner = dis_inner_fake - dis_inner_true
            vae_inner = vae_inner*vae_inner
                                                                
        #mean, var = tf.nn.moments(E(data)[0], axes=0)
            mean, var = tf.nn.moments(latent, axes = 0)
            var_to_one = var - 1
                                                                                        
            normal_loss = tf.reduce_mean(mean*mean) + tf.reduce_mean(var_to_one*var_to_one)
                                                                                                        
            kl_loss = tf.reduce_mean(kl_loss)
            vae_diff_loss = tf.reduce_mean(vae_inner)
            f_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake), dis_fake))
            r_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake_r), dis_fake_r))
            t_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(dis_true), dis_true))
        
            gan_loss = (0.5*t_dis_loss + 0.25*f_dis_loss + 0.25*r_dis_loss)
            vae_loss = tf.reduce_mean(tf.abs(data - fake)) 
            E_loss = vae_diff_loss + KL_COEF*kl_loss + NORMAL_COEF*normal_loss
            G_loss = INNER_LOSS_COEF*vae_diff_loss - gan_loss
            D_loss = gan_loss
        return {gan_loss, vae_loss, f_dis_loss, r_dis_loss, t_dis_loss, vae_diff_loss, E_loss, D_loss, kl_loss, normal_loss}

    def call(self,inputs):
        latent = self.E(inputs)
        return self.G(latent)


def encoder(self):
    input_E = keras.layers.Input(shape=(IM_DIM, IM_DIM, 3))
        
    X = keras.layers.Conv2D(filters=DEPTH*2, kernel_size=K_SIZE, strides=2, padding='same')(input_E)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2D(filters=DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    X = keras.layers.Conv2D(filters=DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(LATENT_DEPTH)(X)    
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
    mean = keras.layers.Dense(LATENT_DEPTH,activation="tanh")(X)
    logsigma = keras.layers.Dense(LATENT_DEPTH,activation="tanh")(X)
    latent = keras.layers.Lambda(sampling, output_shape=(LATENT_DEPTH,))([mean, logsigma])
                                                                                    
    kl_loss = 1 + logsigma - keras.backend.square(mean) - keras.backend.exp(logsigma)
    kl_loss = keras.backend.mean(kl_loss, axis=-1)
    kl_loss *= -0.5
                                                                                                       
    return keras.models.Model(input_E, [latent,kl_loss])

def generator(self):
    input_G = keras.layers.Input(shape=(LATENT_DEPTH,))
    
    X = keras.layers.Dense(8*8*DEPTH*8)(input_G)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    X = keras.layers.Reshape((8, 8, DEPTH * 8))(X)
                        
    X = keras.layers.Conv2DTranspose(filters=DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2DTranspose(filters=DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
                                                       
    X = keras.layers.Conv2DTranspose(filters=DEPTH, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
                                                                        
    X = keras.layers.Conv2D(filters=3, kernel_size=K_SIZE, padding='same')(X)
    X = keras.layers.Activation('sigmoid')(X)

    return keras.models.Model(input_G, X)

def discriminator(self):
    input_D = keras.layers.Input(shape=(IM_DIM, IM_DIM, 3))
           
    X = keras.layers.Conv2D(filters=DEPTH, kernel_size=K_SIZE, strides=2, padding='same')(input_D)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
                        
    X = keras.layers.Conv2D(filters=DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(input_D)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Conv2D(filters=DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2D(filters=DEPTH*8, kernel_size=K_SIZE, padding='same')(X)
    inner_output = keras.layers.Flatten()(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
                                                                   
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(DEPTH*8)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
                                                                                        
    output = keras.layers.Dense(1)(X)    
                                                                     
    return keras.models.Model(input_D, [output, inner_output])

def sampling(args):
    mean, logsigma = args
    epsilon = K.random_normal(shape=K.shape(mean))
    return mean + tf.exp(logsigma / 2)*epsilon


class VAEGAN():
    def __init__(self):
        self.models = self._build()
        self.full_model = self.models[0]
        self.encoder = self.models[1]
        self.generator = self.models[2]
        self.discriminator = self.models[3]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE

    def _build(self):
        #### MODELS
#        vaegan_encoder = Model(vaegan_x, [vaegan_z_mean, vaegan_z_log_var, vaegan_z], name = 'encoder')
#        vaegan_generator = Model(vaegan_z_input, vaegan_g5, name = 'generator')
#        vaegan_discriminator = Model(vaegan_disc_input, vaegan_d4, name = 'discriminator')
#        vaegan_full = VAEModel(vae_encoder, vae_decoder, 10000)

        ### OPTIMIZER
        vaegan_E = encoder(self)
        vaegan_G = generator(self)
        vaegan_D = discriminator(self)
        opti = Adam(lr=LEARNING_RATE)

        vaegan_full = VAEGANModel(vaegan_E, vaegan_G, vaegan_D, 10000, opti)
        vaegan_full.compile(optimizer = opti, loss = 'mse')

        return (vaegan_full, vaegan_E, vaegan_G, vaegan_D)

    def set_weights(self, filepath):
        self.full_model.load_weights(filepath)

    def train(self, data):
        self.full_model.fit(data, data,
                shuffle=True,
                epochs=1,
                batch_size=BATCH_SIZE,
                )
        
    def save_weights(self, filepath):
        self.full_model.save_weights(filepath)

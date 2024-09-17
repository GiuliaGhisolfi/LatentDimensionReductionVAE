import os

import tensorflow as tf
from keras.initializers import RandomNormal, Zeros
from tensorflow.keras.layers import (Conv1D, Conv1DTranspose, Conv2D, Dense,
                                     Dropout, Flatten, Input, Layer, LeakyReLU,
                                     Permute, Reshape)
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization

from src.autoencoders.ae import AutoEncoder

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)

class ReparameterizationLayer(Layer):
    def call(self, inputs, random_seed=RANDOM_SEED):
        '''Reparameterization trick to sample z from Gaussian distribution.'''
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Sample epsilon from a standard normal distribution
        epsilon = tf.random.normal(shape=(batch, dim), mean=0.0, stddev=1.0, seed=random_seed)
        
        # Reparameterization trick: z = mean + std * epsilon
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z

class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, input_dim, latent_dim=128, **kwargs):
        if 'encoder_name' not in kwargs:
            kwargs['encoder_name'] = 'vae_encoder'
        super().__init__(input_dim, latent_dim, **kwargs)

    def _build_encoder(self, name='vae_encoder'):
        input_layer = Input(shape=self.input_dim, dtype='float32', name='input_layer')
        x = input_layer
        if not self.input_type_int:
            x = Permute((2, 3, 1))(x) # from (channels, height, width) to (height, width, channels)
        else:
            x = Reshape((self.input_dim, 1))(input_layer)
        self.conv_shapes = []

        for i in range(len(self.n_filters)):
            if not self.input_type_int:
                x = Conv2D(
                    filters=self.n_filters[i],
                    kernel_size=self.kernel_size[i],
                    strides=self.stride[i],
                    padding=self.padding[i],
                    activation=self.activation,
                    kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                    bias_initializer=Zeros(),
                )(x)
                x = InstanceNormalization()(x)
            else:
                x = Conv1D(
                    filters=self.n_filters[i],
                    kernel_size=self.kernel_size[i],
                    strides=self.stride[i],
                    padding=self.padding[i],
                    activation=self.activation,
                    kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                    bias_initializer=Zeros(),
                )(x)
                x = InstanceNormalization()(x)
            self.conv_shapes.append(x.shape[1:])

        x = Flatten()(x)

        for dim in self.hidden_dims:
            x = Dense(
                dim,
                activation=self.activation,
                kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                bias_initializer=Zeros(),
                )(x)
            x = Dropout(self.dropout)(x)
            x = LeakyReLU(alpha=self.alpha)(x)
        x = Dense(
            self.latent_dim,
            activation=self.activation,
            kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
            bias_initializer=Zeros(),
            )(x)

        z_mean = Dense(
            self.latent_dim,
            activation=self.activation,
            kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
            bias_initializer=Zeros(),
            name='z_mean'
            )(x)
        z_log_var = Dense(
            self.latent_dim, 
            activation=self.activation,
            kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
            bias_initializer=Zeros(),
            name='z_log_var'
            )(x)
        z = ReparameterizationLayer(name='z')([z_mean, z_log_var], random_seed=self.random_seed) # reparameterization trick layer

        return Model(input_layer, [z_mean, z_log_var, z], name=name)
    
    def _build_ae(self):
        input_layer = Input(shape=self.input_dim, dtype='float32')
        z_mean, z_log_var, z = self.encoder(input_layer)
        output_layer = self.decoder(z)
        return Model(input_layer, output_layer, name='vae')
    
    def compute_latent_vector(self, X):
        if len(X.shape) == 3:
            X = X.reshape(1, *X.shape)
        _, _, z = self.encoder.predict(X)

        return z
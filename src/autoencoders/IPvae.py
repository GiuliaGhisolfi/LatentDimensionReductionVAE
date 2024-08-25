import os

import numpy as np
import tensorflow as tf
from keras.initializers import RandomNormal, Zeros
from tensorflow.keras.layers import (Conv1DTranspose, Conv2DTranspose, Dense,
                                     Dropout, Input, LeakyReLU, Permute,
                                     Reshape)
from tensorflow.keras.models import Model

from src.autoencoders.vae import VariationalAutoEncoder

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class IdentityPreservingVAE(VariationalAutoEncoder):
    def __init__(self, input_dim, conditions_dim, latent_dim=128, **kwargs):
        self.conditions_dim = conditions_dim
        super().__init__(input_dim, latent_dim, encoder_name='ip_vae_encoder', **kwargs)

    def _build_decoder(self):
        # define a decoder that takes the latent representation as input
        # and gives the original image and conditional information as output
        latent_input = Input(shape=(self.latent_dim,), dtype='float32')
        x = latent_input

        for dim in reversed(self.hidden_dims):
            x = Dense(
                dim,
                activation=self.activation,
                kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                bias_initializer=Zeros(),
                )(x)
            x = Dropout(self.dropout)(x)
            x = LeakyReLU(alpha=self.alpha)(x)

        x_image = Dense(
            np.prod(self.conv_shapes[-1]),
            activation=self.activation,
            kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
            bias_initializer=Zeros(),
            )(x)
        x_image = Reshape(self.conv_shapes[-1])(x_image)

        if not self.input_type_int:
            for i in reversed(range(len(self.conv_shapes))):
                x_image = Conv2DTranspose(
                    filters=self.n_filters[i],
                    kernel_size=self.kernel_size[i],
                    strides=self.stride[i],
                    padding=self.padding[i],
                    activation=self.activation,
                    kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                    bias_initializer=Zeros(),
                )(x_image)
            output_layer_image = Conv2DTranspose(
                filters=self.output_channels,
                kernel_size=1,
                padding='same',
                activation='sigmoid',
                kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                bias_initializer=Zeros(),
            )(x_image)
            output_layer_image = Permute((3, 1, 2))(output_layer_image) # (channels, height, width)
        else:
            for i in reversed(range(len(self.conv_shapes))):
                x_image = Conv1DTranspose(
                    filters=self.n_filters[i],
                    kernel_size=self.kernel_size[i],
                    strides=self.stride[i],
                    padding=self.padding[i],
                    activation=self.activation,
                    kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                    bias_initializer=Zeros(),
                )(x_image)
            output_layer_image = Conv1DTranspose(
                filters=self.output_channels,
                kernel_size=1,
                padding='same',
                activation='sigmoid',
                kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                bias_initializer=Zeros(),
            )(x_image)
            output_layer_image = Reshape((self.input_dim,))(output_layer_image)

        x_cond = Dense(
            self.conditions_dim,
            activation=self.activation,
            kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
            bias_initializer=Zeros(),
            )(x)
        output_layer_cond = Dense(
            self.conditions_dim,
            activation='sigmoid',
            kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
            bias_initializer=Zeros()
            )(x_cond)

        return Model(latent_input, [output_layer_image, output_layer_cond], name='ip_vae_decoder')

    def _build_ae(self, encoder_name='ip_vae_encoder'):
        self.encoder = self._build_encoder(name=encoder_name)
        self.decoder = self._build_decoder()

        input_layer = Input(shape=(self.input_dim,), dtype='float32')
        z_mean, z_log_var, z = self.encoder(input_layer)
        output_layer_image, output_layer_cond = self.decoder(z)

        return Model(input_layer, [output_layer_image, output_layer_cond], name='identity_preserving_vae')

    def _compile(self):
        if self.n_gpu >= 2:
            with self.strategy.scope():
                self.ae.compile(
                    optimizer=self.optimizer,
                    loss={'ip_vae_decoder': self.loss_function, 'ip_vae_decoder_1': 'mse'},
                    metrics=self.metrics
                )
        else:
            self.ae.compile(
                optimizer=self.optimizer,
                loss={'ip_vae_decoder': self.loss_function, 'ip_vae_decoder_1': 'mse'},
                metrics=self.metrics
            )

    def train(self, X_train, cX_train, X_val, cX_val, epochs=100, batch_size=32):
        self._compile()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )

        self.ae.fit(
            x=X_train,
            y=[X_train, cX_train],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, [X_val, cX_val]),
            shuffle=True,
            callbacks=[early_stopping],
        )

        if self.save_model:
            self.save()

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.initializers import RandomNormal, Zeros
from tensorflow.keras.layers import (Concatenate, Conv1D, Conv1DTranspose,
                                     Conv2D, Conv2DTranspose, Dense, Dropout,
                                     Flatten, Input, LeakyReLU, Permute,
                                     Reshape)
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization

from src.autoencoders.vae import (ReparameterizationLayer,
                                  VariationalAutoEncoder)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class ConditionalVAE(VariationalAutoEncoder):
    def __init__(self, input_dim, conditions_dim, latent_dim=128, **kwargs):
        self.conditions_dim = conditions_dim # dimension of the condition vector
        super().__init__(input_dim, latent_dim, **kwargs)

    def _build_encoder(self, name='cvae_encoder'):
        input_layer = Input(shape=self.input_dim, dtype='float32')
        condition_layer = Input(shape=self.conditions_dim, dtype='float32')
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
        x = Concatenate()([x, condition_layer])

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

        return Model([input_layer, condition_layer], [z_mean, z_log_var, z], name=name)
    
    def _build_decoder(self):
        latent_input = Input(shape=(self.latent_dim,), dtype='float32')
        condition_layer = Input(shape=self.conditions_dim, dtype='float32')
        x = Concatenate()([latent_input, condition_layer])

        for dim in reversed(self.hidden_dims):
            x = Dense(
                dim,
                activation=self.activation,
                kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                bias_initializer=Zeros(),
                )(x)
            x = Dropout(self.dropout)(x)
            x = LeakyReLU(alpha=self.alpha)(x)

        x = Dense(
            np.prod(self.conv_shapes[-1]),
            activation=self.activation,
            kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
            bias_initializer=Zeros(),
            )(x)
        x = Reshape(self.conv_shapes[-1])(x)

        if not self.input_type_int:
            for i in reversed(range(len(self.conv_shapes))):
                x = Conv2DTranspose(
                    filters=self.n_filters[i],
                    kernel_size=self.kernel_size[i],
                    strides=self.stride[i],
                    padding=self.padding[i],
                    activation=self.activation,
                    kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                    bias_initializer=Zeros(),
                )(x)
                x = InstanceNormalization()(x)
            output_layer = Conv2DTranspose(
                filters=self.output_channels,
                kernel_size=1,
                padding='same',
                activation='sigmoid',
                kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                bias_initializer=Zeros(),
            )(x)
            output_layer = Permute((3, 1, 2))(output_layer) # (channels, height, width)
        else:
            for i in reversed(range(len(self.conv_shapes))):
                x = Conv1DTranspose(
                    filters=self.n_filters[i],
                    kernel_size=self.kernel_size[i],
                    strides=self.stride[i],
                    padding=self.padding[i],
                    activation=self.activation,
                    kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                    bias_initializer=Zeros(),
                )(x)
                x = InstanceNormalization()(x)
            output_layer = Conv1DTranspose(
                filters=self.output_channels,
                kernel_size=1,
                padding='same',
                activation='sigmoid',
                kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                bias_initializer=Zeros(),
            )(x)
            output_layer = Reshape((self.input_dim,))(output_layer)

        return Model([latent_input, condition_layer], output_layer, name='cvae_decoder')

    def _build_ae(self):
        input_layer = Input(shape=(self.input_dim,), dtype='float32')
        condition_layer = Input(shape=self.conditions_dim, dtype='float32')
        z_mean, z_log_var, z = self.encoder([input_layer, condition_layer])
        output_layer = self.decoder([z, condition_layer])
        return Model([input_layer, condition_layer], output_layer, name='cvae')

    def train(self, X_train, cX_train, X_val, cX_val, epochs=100, batch_size=32):
        self._compile()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )

        self.ae.fit(
            x=[X_train, cX_train],
            y=X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([X_val, cX_val], X_val),
            shuffle=True,
            callbacks=[early_stopping],
        )

        if self.save_model:
            self.save()
    
    def compute_latent_vector(self, X, cX):
        if len(X.shape) == 3:
            X = X.reshape(1, *X.shape)
            cX = cX.reshape(1, *cX.shape)
        _, _, z = self.encoder.predict(X, cX)

        return z

    def recostruct(self, X, cX):
        X = X.reshape(1, *X.shape)
        cX = cX.reshape(1, *cX.shape)

        return self.ae.predict([X, cX], verbose=0)

    def visualize_recostruction(self, X, cX):
        recostruction = self.recostruct(X, cX)

        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        if X.shape[0] == 3:
            original_image = np.moveaxis(X, 0, -1)
            recostruction_image = recostruction.reshape(recostruction.shape[1], recostruction.shape[2], recostruction.shape[3])
            recostruction_image = np.moveaxis(recostruction_image, 0, -1)
            cmap=None
        else:
            original_image = X.reshape(X.shape[1], X.shape[2]).T
            recostruction_image = recostruction.reshape(recostruction.shape[2], recostruction.shape[3]).T
            cmap='gray'

        axs[0].imshow(original_image, cmap=cmap)
        axs[0].set_title('Original')
        axs[0].axis('off')
        axs[1].imshow(recostruction_image, cmap=cmap)
        axs[1].set_title('Recostruction')
        axs[1].axis('off')


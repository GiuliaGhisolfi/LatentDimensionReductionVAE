import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.initializers import RandomNormal, Zeros
from keras.utils import plot_model
from tensorflow.keras.layers import (Conv1D, Conv1DTranspose, Conv2D,
                                     Conv2DTranspose, Dense, Dropout, Flatten,
                                     Input, LeakyReLU, Permute, Reshape)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class AutoEncoder():
    def __init__(
        self,
        input_dim,
        latent_dim=128,
        n_filters=[32, 64, 64, 32],
        kernel_size=[3, 3, 3, 3],
        stride=[1, 2, 2, 1],
        padding=['same', 'same', 'same', 'same'],
        hidden_dims=[512, 256],
        activation='relu',
        learning_rate=0.001,
        dropout=0.3,
        alpha=0.2, # LeakyReLU alpha
        loss_function='mse',
        patience=10,
        metrics=['accuracy'],
        n_gpu=1,
        random_seed=RANDOM_SEED,
        save_model=False,
        save_path=None,
        encoder_name='ae_encoder',
    ):

        # Model parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.dropout = dropout
        self.alpha = alpha
        self.output_dim = input_dim
        self.input_type_int = type(input_dim)==int
        self.output_channels = 1 if self.input_type_int else input_dim[0]
        self.loss_function = loss_function
        self.patience = patience
        self.metrics = metrics

        # Save model parameters
        self.save_model = save_model
        if save_model and save_path is None:
            self.save_path = 'autoencoder.h5'
        self.save_path = save_path

        self.n_gpu = n_gpu
        self.random_seed = random_seed

        # Build model
        self._build_model(encoder_name=encoder_name)

    def _build_model(self, encoder_name):
        if self.n_gpu >= 2:
            # devices_list=["/gpu:0", "/gpu:1"]
            devices_list=["/gpu:{}".format(n) for n in range(self.n_gpu)]
            self.strategy = tf.distribute.MirroredStrategy(
                devices=devices_list,
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
            )
            print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
            with self.strategy.scope():
                # get multi-gpu train model
                self.encoder = self._build_encoder(name=encoder_name)
                self.decoder = self._build_decoder()
                self.ae = self._build_ae()
        else:
            # get normal train model
            self.encoder = self._build_encoder(name=encoder_name)
            self.decoder = self._build_decoder()
            self.ae = self._build_ae()

    def _build_encoder(self, name):
        input_layer = Input(shape=(self.input_dim,), dtype='float32')
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
        latent_output = Dense(
            self.latent_dim,
            activation=self.activation,
            kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                bias_initializer=Zeros(),
            )(x)

        return Model(input_layer, latent_output, name=name)

    def _build_decoder(self):
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

        return Model(latent_input, output_layer, name='ae_decoder')
    
    def _build_ae(self):
        input_layer = Input(shape=self.input_dim, dtype='float32')
        z = self.encoder(input_layer)
        output = self.decoder(z)

        ae = Model(input_layer, output, name='autoencoder')

        return ae
    
    def _compile(self):
        if self.n_gpu >= 2:
            with self.strategy.scope():
                self.ae.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        else:
            self.ae.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)

    def train(self, X_train, X_val, epochs=100, batch_size=32):
        self._compile()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )

        self.ae.fit(
            x=X_train,
            y=X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            shuffle=True,
            callbacks=[early_stopping],
        )

        if self.save_model:
            self.save()

    def summarize(self):
        self.encoder.summary()
        self.decoder.summary()
        self.ae.summary()

    def encode(self, X):
        return self.encoder.predict(X)
    
    def decode(self, z):
        return self.decoder.predict(z)
    
    def generate(self, z):
        return self.ae.predict(z)

    def visualize_loss(self):
        plt.plot(self.ae.history.history['loss'], label='loss')
        plt.plot(self.ae.history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def compute_latent_vector(self, X):
        if len(X.shape) == 3:
            X = X.reshape(1, *X.shape)
        z = self.encoder.predict(X)

        return z
    
    def recostruct(self, X):
        X = X.reshape(1, *X.shape)
        return self.ae.predict(X, verbose=0)

    def visualize_recostruction(self, X):
        recostruction = self.recostruct(X)

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

    def save(self):
        self.ae.save('saved_models/' + self.save_path)

    def load(self, path):
        self.ae = load_model(path)
        self.encoder = self.ae.get_layer('encoder')
        self.decoder = self.ae.get_layer('decoder')

    def plot_model(self, to_file='model.png', **kwargs):
        plot_model(
            model=self.ae,
            to_file=to_file,
            show_shapes=False,
            show_dtype=False,
            show_layer_names=False,
            rankdir="TB",
            expand_nested=False,
            dpi=200,
            show_layer_activations=True,
            show_trainable=False,
            **kwargs
        )
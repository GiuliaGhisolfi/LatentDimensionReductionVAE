import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.initializers import RandomNormal, Zeros
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Concatenate, Conv2DTranspose, Dense,
                                     Dropout, Input, LeakyReLU, Permute,
                                     Reshape)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class Generator:
    def __init__(
        self,
        input_dim,
        conditions_dim,
        output_dim,
        n_filters=[32, 64, 64, 32],
        kernel_size=[3, 3, 3, 3],
        stride=[1, 2, 2, 1],
        padding=['same', 'same', 'same', 'same'],
        hidden_dims=[256, 512],
        activation='relu',
        learning_rate=0.001,
        dropout=0.3,
        alpha=0.3,
        loss_function='mse',
        patience=10,
        metrics=['accuracy'],
        n_gpu=1,
        random_seed=RANDOM_SEED,
        save_model=False,
        save_path=None,
        name='generator'
    ):
        # Model parameters
        self.input_dim = input_dim # latent rappresentation
        self.conditions_dim = conditions_dim
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.optimizer = Adam(learning_rate=learning_rate)
        self.dropout = dropout
        self.alpha = alpha
        self.output_dim = output_dim # output image
        self.input_type_int = type(input_dim)==int
        self.output_channels = output_dim[0]
        self.loss_function = loss_function
        self.patience = patience
        self.metrics = metrics

        # Save model parameters
        self.save_model = save_model
        if save_model and save_path is None:
            self.save_path = 'generator.h5'
        self.save_path = save_path

        self.n_gpu = n_gpu
        self.random_seed = random_seed

        # Build model
        self._build_model(name)

    def _build_model(self, name='generator'):
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
                self.generator = self._build_generator(name)
        else:
            self.generator = self._build_generator(name)
    
    def _build_generator(self, name='generator'):
        latent_input = Input(shape=(self.input_dim,), dtype='float32')
        condition_layer = Input(shape=self.conditions_dim, dtype='float32')
        x = Concatenate()([latent_input, condition_layer])

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
            np.prod(self.n_filters[0] * 7 * 7),
            activation=self.activation,
            kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
            bias_initializer=Zeros(),
            )(x)
        #x = LeakyReLU(alpha=self.alpha)(x)
        x = Reshape((7, 7, self.n_filters[0]))(x) # dimension of the first layer of the generator, chosen arbitrarily: 7

        for i in range(len(self.n_filters)):
            x = Conv2DTranspose(
                filters=self.n_filters[i],
                kernel_size=self.kernel_size[i],
                strides=self.stride[i],
                padding=self.padding[i],
                activation=self.activation,
                kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
                bias_initializer=Zeros(),
            )(x)
        last_kernel_size = (self.output_dim[1] - (x.shape[1] - 1) * 1 + 2 * 0) #(output_dim - (input_dim - 1) * stride + 2 * padding)
        output_layer = Conv2DTranspose(
            filters=self.output_channels,
            kernel_size=last_kernel_size,
            strides=1,
            padding='valid', # valid maeans no padding
            activation='sigmoid',
            kernel_initializer=RandomNormal(stddev=0.01, seed=self.random_seed),
            bias_initializer=Zeros(),
        )(x)
        output_layer = Permute((3, 1, 2))(output_layer) # (channels, height, width)

        return Model([latent_input, condition_layer], output_layer, name=name)
    
    def _compile(self):
        if self.n_gpu >= 2:
            with self.strategy.scope():
                self.generator.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        else:
            self.generator.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)

    def train(self, z_train, X_train, cX_train, z_val, X_val, cX_val, batch_size=32, epochs=100):
        self._compile()

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )

        self.generator.fit(
            x=[z_train, cX_train],
            y=X_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([z_val, cX_val], X_val),
            shuffle=True,
            callbacks=[early_stopping],
        )

        if self.save_model:
            self.save()
    
    def summarize(self):
        self.generator.summary()
    
    def generate(self, z, cX):
        return self.generator.predict([z, cX])
    
    def visualize_loss(self):
        plt.plot(self.generator.history.history['loss'], '--o', label='loss')
        plt.plot(self.generator.history.history['val_loss'], '--o', label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save(self):
        self.generator.save('saved_models/' + self.save_path)

    def load_model(self, path):
        self.generator = load_model(path)
    
    def plot_model(self, to_file='generator.png', **kwargs):
        plot_model(
            model=self.generator,
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

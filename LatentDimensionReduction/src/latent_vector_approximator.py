import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.autoencoders.IPvae import IdentityPreservingVAE
from src.autoencoders.vae import VariationalAutoEncoder
from src.generator import Generator

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def convolutional_layers_params(n_conv_layers):
    kernel_size = [1] * n_conv_layers
    stride = [1] * n_conv_layers
    padding = ['same'] * n_conv_layers
    
    return kernel_size, stride, padding

class LatentVectorApproximator:
    def __init__(
        self,
        input_dim,
        conditions_dim,
        latent_dim=128,
        loss_function_image='mse',
        loss_function_latent='mse',
        patience=10,
        metrics=['accuracy', 'mae'],
        learning_rate_fine_tuning=0.001,
        dropout=0.3,
        alpha=0.2, # LeakyReLU alpha
        
        n_filters_vae=[32, 64, 64, 32],
        kernel_size_vae=[3, 3, 3, 3],
        stride_vae=[1, 2, 2, 1],
        padding_vae=['same', 'same', 'same', 'same'],
        hidden_dims_vae=[512, 256],
        activation_vae='relu',
        learning_rate_vae=0.001,

        n_filters_generator1=[1, 3, 3, 1],
        kernel_size_generator1=[3, 3, 3, 3],
        stride_generator1=[1, 2, 2, 1],
        padding_generator1=['same', 'same', 'same', 'same'],
        hidden_dims_generator1=[256, 512],
        activation_generator1='relu',
        learning_rate_generator1=0.001,

        n_filters_ipvae=[1, 3, 3, 1],
        n_conv_layers_ipvae=4,
        hidden_dims_ipvae=[512, 256],
        activation_ipvae='relu',
        learning_rate_ipvae=0.001,

        n_filters_generator2=[1, 3, 3, 1],
        kernel_size_generator2=[3, 3, 3, 3],
        stride_generator2=[1, 2, 2, 1],
        padding_generator2=['same', 'same', 'same', 'same'],
        hidden_dims_generator2=[256, 512],
        activation_generator2='relu',
        learning_rate_generator2=0.001,

        n_gpu=1,
        random_seed=RANDOM_SEED,
        save_model=False,
        save_path=None,
        ):
        # Model parameters
        self.input_dim = input_dim
        self.conditions_dim = conditions_dim
        self.latent_dim = latent_dim
        self.loss_function_image = loss_function_image
        self.loss_function_latent = loss_function_latent
        self.patience = patience
        self.metrics = metrics
        self.learning_rate = learning_rate_fine_tuning

        self.save_model = save_model
        if save_model and save_path is None:
            self.save_path = 'latent_vector_approximator.h5'
        self.save_path = save_path

        self.n_gpu = n_gpu
        self.random_seed = random_seed

        # VAE
        self.vae = self._init_vae(
            n_filters=n_filters_vae,
            kernel_size=kernel_size_vae,
            stride=stride_vae,
            padding=padding_vae,
            hidden_dims=hidden_dims_vae,
            activation=activation_vae,
            learning_rate=learning_rate_vae,
            dropout=dropout,
            alpha=alpha
        )

        # Generator1
        self.generator1 = self._init_generator(
            n_filters=n_filters_generator1,
            kernel_size=kernel_size_generator1,
            stride=stride_generator1,
            padding=padding_generator1,
            hidden_dims=hidden_dims_generator1,
            activation=activation_generator1,
            learning_rate=learning_rate_generator1,
            dropout=dropout,
            alpha=alpha,
            name='generator1'
        )

        # IPvae
        self.ipvae = self._init_ipvae(
            n_filters=n_filters_ipvae,
            n_conv_layers=n_conv_layers_ipvae,
            hidden_dims=hidden_dims_ipvae,
            activation=activation_ipvae,
            learning_rate=learning_rate_ipvae,
            dropout=dropout,
            alpha=alpha
        )

        # Generator2
        self.generator2 = self._init_generator(
            n_filters=n_filters_generator2,
            kernel_size=kernel_size_generator2,
            stride=stride_generator2,
            padding=padding_generator2,
            hidden_dims=hidden_dims_generator2,
            activation=activation_generator2,
            learning_rate=learning_rate_generator2,
            dropout=dropout,
            alpha=alpha,
            name='generator2'
        )

        print('LatentVectorApproxiator initialized')

    def _init_vae(self, n_filters, kernel_size, stride, padding, hidden_dims, activation, learning_rate, dropout, alpha):
        return VariationalAutoEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            hidden_dims=hidden_dims,
            activation=activation,
            learning_rate=learning_rate,
            dropout=dropout,
            alpha=alpha,
            loss_function=self.loss_function_image,
            patience=self.patience,
            metrics=self.metrics,
            n_gpu=self.n_gpu,
            random_seed=self.random_seed,
            save_model=self.save_model,
            save_path='vae_'+self.save_path if self.save_model else None
        )

    def _init_generator(self, n_filters, kernel_size, stride, padding, hidden_dims, activation, learning_rate,
        dropout, alpha, name):
        return Generator(
            input_dim=self.latent_dim,
            conditions_dim = self.conditions_dim,
            output_dim=self.input_dim,
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            hidden_dims=hidden_dims,
            activation=activation,
            learning_rate=learning_rate,
            dropout=dropout,
            alpha=alpha,
            loss_function=self.loss_function_image,
            patience=self.patience,
            metrics=self.metrics,
            n_gpu=self.n_gpu,
            random_seed=self.random_seed,
            save_model=self.save_model,
            save_path=name+'_'+self.save_path if self.save_model else None,
            name=name
        )

    def _init_ipvae(self, n_filters, n_conv_layers, hidden_dims, activation, learning_rate, dropout, alpha):
        kernel_size, stride, padding = convolutional_layers_params(n_conv_layers)
        return IdentityPreservingVAE(
            input_dim=self.latent_dim,
            conditions_dim = self.conditions_dim,
            latent_dim=self.latent_dim,
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            hidden_dims=hidden_dims,
            activation=activation,
            learning_rate=learning_rate,
            dropout=dropout,
            alpha=alpha,
            loss_function=self.loss_function_latent,
            patience=self.patience,
            metrics=self.metrics,
            n_gpu=self.n_gpu,
            random_seed=self.random_seed,
            save_model=self.save_model,
            save_path='ipvae_'+self.save_path if self.save_model else None
        )
    
    def train(self, X_train, cX_train, X_val, cX_val, pre_train_epochs=100, epochs=100, batch_size=32):
        # pre-training
        self._pre_train(X_train, cX_train, X_val, cX_val, epochs=pre_train_epochs, batch_size=batch_size)
        print('Pre-training done')

        # build model from pre-trained models
        print('Building final model')
        self.model = self._build_model()

        # fine-tuning
        self._fine_tune(X_train, cX_train, X_val, cX_val, epochs=epochs, batch_size=batch_size)
        print('Fine-tuning done')

    def _pre_train(self, X_train, cX_train, X_val, cX_val, epochs, batch_size):
        print('Training VAE')
        self.vae.train(X_train, X_val, epochs=epochs, batch_size=batch_size)
        _, _, z_vae_train = self.vae.encoder.predict(X_train)
        _, _, z_vae_val = self.vae.encoder.predict(X_val)

        print('Training first Generator')
        self.generator1.train(z_vae_train, X_train, cX_train, z_vae_val, X_val, cX_val, epochs=epochs, batch_size=batch_size)

        print('Training IPvae')
        self.ipvae.train(z_vae_train, cX_train, z_vae_val, cX_val, epochs=epochs, batch_size=batch_size)
        _, _, z_ipvae_train = self.ipvae.encoder.predict(z_vae_train)
        _, _, z_ipvae_val = self.ipvae.encoder.predict(z_vae_val)

        print('Training second Generator')
        self.generator2.train(z_ipvae_train,X_train, cX_train, z_ipvae_val, X_val, cX_val, epochs=epochs, batch_size=batch_size)

    def _build_model(self):

        input_layer = self.vae.encoder.input
        condition_layer = Input(shape=self.conditions_dim, dtype='float32', name='condition_layer')

        _, _, z = self.vae.encoder(input_layer)
        output_layer1 = self.generator1.generator([z, condition_layer])

        _, _, z = self.ipvae.encoder(z)
        output_layer2 = self.generator2.generator([z, condition_layer])

        model = Model(
            inputs=[input_layer, condition_layer],
            outputs=[output_layer1, output_layer2],
            name='latent_vector_approximator')
        #model.summary()

        return model
    
    def _compile(self):
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)

        if self.n_gpu >= 2:
            with self.strategy.scope():
                self.model.compile(optimizer=optimizer, loss=self.loss_function_image, metrics=self.metrics)
        else:
            self.model.compile(optimizer=optimizer, loss=self.loss_function_image, metrics=self.metrics)

    def _fine_tune(self, X_train, cX_train, X_val, cX_val, epochs, batch_size):
        # fine-tune
        print('Fine-tuning')
        self._compile()

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )

        self.model.fit(
            x=[X_train, cX_train],
            y=[X_train, X_train],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([X_val, cX_val], [X_val, X_val]),
            shuffle=True,
            callbacks=[early_stopping],
        )

    def compute_latent_vector(self, X):
        if len(X.shape) == 3:
            X = X.reshape(1, *X.shape)
        _, _, z1 = self.vae.encoder.predict(X)
        _, _, z = self.ipvae.encoder.predict(z1) # identity preserving VAE

        return z1, z # vae and identity preserving latent vectors

    def recostruct(self, X, cX):
        X = X.reshape(1, *X.shape)
        cX = cX.reshape(1, *cX.shape)

        recostruction1, recostruction2 = self.model.predict([X, cX])

        return recostruction1, recostruction2

    def summarize(self):
        self.vae.summarize()
        self.generator1.summarize()
        self.ipvae.summarize()
        self.generator2.summarize()

    def visualize_loss(self):
        plt.plot(self.model.history.history['loss'], label='loss')
        plt.plot(self.model.history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    
    def visualize_recostruction(self, X, cX):
        recostruction1, recostruction2 = self.recostruct(X, cX)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        if X.shape[0] == 3:
            original_image = np.moveaxis(X, 0, -1)
            recostruction1_image = recostruction1.reshape(recostruction1.shape[1], recostruction1.shape[2], recostruction1.shape[3])
            recostruction1_image = np.moveaxis(recostruction1_image, 0, -1)
            recostruction2_image = recostruction2.reshape(recostruction2.shape[1], recostruction2.shape[2], recostruction2.shape[3])
            recostruction2_image = np.moveaxis(recostruction2_image, 0, -1)
            cmap=None
        else:
            original_image = X.reshape(X.shape[1], X.shape[2]).T
            recostruction1_image = recostruction1.reshape(recostruction1.shape[2], recostruction1.shape[3]).T
            recostruction2_image = recostruction2.reshape(recostruction2.shape[2], recostruction2.shape[3]).T
            cmap='gray'

        axs[0].imshow(original_image, cmap=cmap)
        axs[0].set_title('Original')
        axs[0].axis('off')
        axs[1].imshow(recostruction1_image, cmap=cmap)
        axs[1].set_title('Recostruction first Generator')
        axs[1].axis('off')
        axs[2].imshow(recostruction2_image, cmap=cmap)
        axs[2].set_title('Recostruction second Generator')
        axs[2].axis('off')

        plt.show()

    def save(self):
        if self.save_path is None:
            self.save_path = 'latent_vector_approximator.h5'
        self.model.save('saved_models/' + self.save_path)

    def load_model(self, path):
        self.model = load_model(path)
import importlib
import json
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

from src.utils.dataloader import Dataset

DATA_PATH = 'data/'
RANDOM_STATE = 42
RANDOM_SEED = 42
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

####### DATA LOADING #######

def tensor_to_numpy(tensor):
    try:
        return tensor.numpy()
    except:
        return tensor.cpu().detach().numpy()

def get_training_and_validation_sets(scenario='no_obj', split_train_val=False):
    feature_set = Dataset(DATA_PATH, condition=scenario)
    X_train, Y_train, cX_train = feature_set.get_training_set()

    if split_train_val:
        X_train, X_val, Y_train, Y_val, cX_train, cX_val = train_test_split(X_train, Y_train, cX_train,
            test_size=0.2, random_state=RANDOM_STATE, shuffle=False)
        
        X_train, X_val, Y_train, Y_val, cX_train, cX_val = tensor_to_numpy(X_train), tensor_to_numpy(X_val
        ), tensor_to_numpy(Y_train), tensor_to_numpy(Y_val), tensor_to_numpy(cX_train), tensor_to_numpy(cX_val)
        
        return X_train, Y_train, cX_train, X_val, Y_val, cX_val
    
    else:
        X_train, Y_train, cX_train = tensor_to_numpy(X_train), tensor_to_numpy(Y_train), tensor_to_numpy(cX_train)
        return X_train, Y_train, cX_train

def get_test_set(scenario='no_obj'):
    feature_set = Dataset(DATA_PATH, condition=scenario)
    X_val, Y_val, cX_val = feature_set.get_validation_set()

    X_val, Y_val, cX_val = tensor_to_numpy(X_val), tensor_to_numpy(Y_val), tensor_to_numpy(cX_val)

    return X_val, Y_val, cX_val

####### GREY SCALE DATA #######

def rgb2gray(rgb):
    # rgb: (3, 64, 64), return: (1, 64, 64)
    return np.dot(rgb[...,:], [0.2989, 0.5870, 0.1140]).reshape(1, 64, 64)

def dataset_to_greyscale(dataset):
    return np.array([rgb2gray(img.T) for img in dataset])

def get_training_and_validation_sets_gray_scale(scenario='no_obj'):
    X_train, Y_train, cX_train, X_val, Y_val, cX_val = get_training_and_validation_sets(scenario)

    X_train_gray_scale, Y_train_gray_scale, X_val_gray_scale, Y_val_gray_scale = dataset_to_greyscale(X_train
    ), dataset_to_greyscale(Y_train), dataset_to_greyscale(X_val), dataset_to_greyscale(Y_val)

    return X_train_gray_scale, Y_train_gray_scale, cX_train, X_val_gray_scale, Y_val_gray_scale, cX_val

def get_test_set_gray_scale(scenario='no_obj'):
    X_val, Y_val, cX_val = get_test_set(scenario)

    X_val_gray_scale, Y_val_gray_scale = dataset_to_greyscale(X_val), dataset_to_greyscale(Y_val)

    return X_val_gray_scale, Y_val_gray_scale, cX_val

####### DATA VISUALIZATION #######

def visualize_dataset(X_train, from_index=0, to_index=10):
    n_samples = to_index - from_index
    n_rows = int(np.ceil(n_samples/5))

    fig, axs = plt.subplots(n_rows, 5, figsize=(15, 3*n_rows))
    fig.subplots_adjust(hspace =.5, wspace=.001)
    axs = axs.ravel()

    for i in range(n_samples):
        axs[i].axis('off')
        image = cv2.cvtColor(tensor_to_numpy(X_train[from_index+i]), cv2.COLOR_BGR2RGB)
        axs[i].imshow(image)
    plt.show()

####### GETTING MODEL CONFIGURATION #######

def import_from_string(dotted_path):
    module_path, _, function_name = dotted_path.rpartition('.')
    module = importlib.import_module(module_path)
    return getattr(module, function_name)

def get_config(config_filename):
    with open(config_filename, 'r') as config_file:
        config = json.load(config_file)
    
    if config['loss_function_image'] == 'src.utils.utils.loss_function':
        loss_function_image = import_from_string(config['loss_function_image'])
    else:
        loss_function_image = config['loss_function_image']
    config.pop('loss_function_image')

    return config, loss_function_image

####### SAVE AND LOAD LATENT VECTORS #######

def save_latent_vectors(X_z1, X_z, filename):
    with open(filename, 'wb') as f:
        pickle.dump((X_z1, X_z), f)

def read_latent_vectors(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

####### LOSS FUNCTIONS #######

def percentage_background(img, max_val, min_val):
    area = tf.cast(tf.size(img, out_type=tf.int32), tf.float32)
    return (tf.reduce_sum(tf.cast(img >= max_val, tf.float32)) + tf.reduce_sum(tf.cast(img <= min_val, tf.float32))) / area

def background_threshold(img, threshold):
    max_val = tf.reduce_max(img)
    min_val = tf.reduce_min(img)

    def condition(max_val, min_val):
        return percentage_background(img, max_val, min_val) < threshold

    def body(max_val, min_val):
        return max_val - 0.01, min_val + 0.01

    max_val, min_val = tf.while_loop(condition, body, [max_val, min_val])
    return max_val, min_val

def background_mask(img, threshold=0.8):
    max_val, min_val = background_threshold(img, threshold)

    mask = tf.ones_like(img)  # 1 means foreground
    mask = tf.where((img >= max_val) | (img <= min_val), tf.zeros_like(mask), mask)  # 0 means background

    return mask

def loss_function(y_true, y_pred, threshold=0.8):
    mask = background_mask(y_true, threshold)

    # MSE for foreground pixels
    foreground = tf.boolean_mask(y_true, mask)
    foreground_pred = tf.boolean_mask(y_pred, mask)
    foreground_loss = tf.reduce_mean(tf.square(foreground - foreground_pred))

    # error for background pixels
    inverse_mask = tf.logical_not(tf.cast(mask, tf.bool))
    background = tf.boolean_mask(y_true, inverse_mask)
    background_pred = tf.boolean_mask(y_pred, inverse_mask)

    div = tf.reduce_sum(tf.cast(background != background_pred, tf.float32))
    if div == 0:
        return foreground_loss
    else:
        background_loss = tf.reduce_sum(tf.abs(background - background_pred)) / div

        return foreground_loss + background_loss

    
####### More LOSS FUNCTIONS and METRICS #######

def bce_loss(y_true, y_pred):
    # Flatten the images to compare pixel by pixel
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    # Calculate binary cross-entropy loss
    bce_loss = binary_crossentropy(y_true, y_pred)
    
    # Return the average BCE loss across all pixels
    return K.mean(bce_loss)

def dice_coef(y_true, y_pred, smooth=1):
    # Flatten the images to compare pixel by pixel
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    # Calculate the intersection and union of the predicted and true masks
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    
    # Calculate the Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def pixel_error_loss(y_true, y_pred, threshold=0.5):
    # Calculate the absolute difference between the true and predicted masks
    diff = K.abs(y_true - y_pred)

    # Take into account only the pixels with a value greater than the threshold
    incorrect_pixels = K.cast(diff > threshold, dtype=tf.float32)

    # Calculate the mean number of incorrect pixels and normalize it by the total number of pixels
    num_incorrect_pixels = K.sum(incorrect_pixels)
    mean_incorrect_pixels = num_incorrect_pixels / K.cast(K.prod(K.shape(y_true)[1:]), dtype=tf.float32)
    
    return mean_incorrect_pixels



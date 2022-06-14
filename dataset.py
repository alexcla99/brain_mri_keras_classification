from utils import load_params, expand_dims, info

import tensorflow as tf
import numpy as np
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_dataset(src:str) -> (tf.data.Dataset, tf.data.Dataset):
    """Instanciate both train and validation data loaders."""
    x_train = np.load(os.path.join(src, "x_train.npy"))
    y_train = np.load(os.path.join(src, "y_train.npy"))
    x_val = np.load(os.path.join(src, "x_val.npy"))
    y_val = np.load(os.path.join(src, "y_val.npy"))
    # Create both data loaders
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    batch_size = load_params()["dataset"]["batch_size"]
    # Return both datasets
    train_dataset = (
        train_loader.map(expand_dims) # TODO map(rotate) for train dataset
        .batch(batch_size)
        .prefetch(buffer_size=AUTOTUNE)
    )
    val_dataset = (
        val_loader.map(expand_dims)
        .batch(batch_size)
        .prefetch(buffer_size=AUTOTUNE)
    )
    info("Train dataset shape: %s" % str(train_dataset.get_shape().as_list()))
    info("Validation dataset shape: %s" % str(val_dataset.get_shape().as_list()))
    return train_dataset, val_dataset

def get_test_dataset(src:str) -> tf.data.Dataset:
    """Instanciate the test data loader."""
    x_test = np.load(os.path.join(src, "x_test.npy"))
    y_test = np.load(os.path.join(src, "y_test.npy"))
    # Create the dataloader
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    batch_size = load_params()["dataset"]["batch_size"]
    # return the dataset
    test_dataset = (
        test_loader.map(expand_dims)
        .batch(batch_size)
        .prefetch(buffer_size=AUTOTUNE)
    )
    info("Test dataset shape: %s" % str(test_dataset.get_shape().as_list()))
    return test_dataset

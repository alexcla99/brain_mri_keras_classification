from utils import load_params, process_scan, info, balance_train_dataset, train_preprocessing, val_test_preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

# DATASETS LOADING ##################################################################################################
def load_dataset(
    src:str,
    augment:bool=False,
    norm_type:str=None,
    img_size:str=None,
    balance:bool=False,
    healthy_reduction:int=None) -> (tf.data.Dataset, tf.data.Dataset, np.array, np.array): # tf.data.Dataset):
    """Instanciate both train and validation data loaders."""
    normal_data_path = [
        os.path.join(os.getcwd(), src, "normal", x)
        for x in os.listdir(os.path.join(src, "normal"))
    ]
    abnormal_data_path = [
        os.path.join(os.getcwd(), src, "abnormal", x) 
        for x in os.listdir(os.path.join(src, "abnormal"))
    ]
    # Loading params and datasets building
    settings = load_params()
    data_rep = settings["preprocessing"]["data_rep"]
    random_seed = settings["preprocessing"]["random_seed"]
    batch_size = settings["dataset"]["batch_size"]
    np.random.seed(random_seed)
    normal_data = np.array(
        [process_scan(path, norm_type=norm_type, img_size=img_size) for path in normal_data_path]
    )
    abnormal_data = np.array(abnormal_data_path)
    # Reducing abnormal dataset according to a given number of samples
    if healthy_reduction is not None:
        abnormal_data = abnormal_data[:healthy_reduction]
    # Balancing both normal and abnormal datasets
    if balance == True:
        normal_data = balance_train_dataset(
            normal_data,
            abnormal_data,
            normal_data_path,
            norm_type=norm_type,
            img_size=img_size
        )
    else:
        abnormal_data = np.array(
            [process_scan(path, norm_type=norm_type, img_size=img_size) for path in abnormal_data_path]
        )
    normal_labels = np.array([0. for _ in range(len(normal_data))])
    abnormal_labels = np.array([1. for _ in range(len(abnormal_data))])
    assert len(abnormal_data) + len(normal_data) == len(abnormal_labels) + len(normal_labels)
    # Split them into abnormal and control train / val / test datasets
    # Data are not shuffled yet in order to get the same MRIs in the different subsets
    # So the dataloader can be called several times (in differents tasks)
    abnormal_x_train, abnormal_x_test, abnormal_y_train, abnormal_y_test = train_test_split(
        abnormal_data,
        abnormal_labels,
        test_size = 1 - data_rep[0],
        shuffle = False
    )
    abnormal_x_val, abnormal_x_test, abnormal_y_val, abnormal_y_test = train_test_split(
        abnormal_x_test,
        abnormal_y_test,
        test_size = data_rep[1]/(data_rep[1]+data_rep[2]),
        shuffle = False
    )
    normal_x_train, normal_x_test, normal_y_train, normal_y_test = train_test_split(
        normal_data,
        normal_labels,
        test_size = 1 - data_rep[0],
        shuffle = False
    )
    normal_x_val, normal_x_test, normal_y_val, normal_y_test = train_test_split(
        normal_x_test,
        normal_y_test,
        test_size = data_rep[1]/(data_rep[1]+data_rep[2]),
        shuffle = False
    )
    # Create the final train / val / test datasets and shuffle them
    x_train = np.concatenate((abnormal_x_train, normal_x_train), axis=0)
    x_val = np.concatenate((abnormal_x_val, normal_x_val), axis=0)
    x_test = np.concatenate((abnormal_x_test, normal_x_test), axis=0)
    y_train = np.concatenate((abnormal_y_train, normal_y_train), axis=0)
    y_val = np.concatenate((abnormal_y_val, normal_y_val), axis=0)
    y_test = np.concatenate((abnormal_y_test, normal_y_test), axis=0)
    # Shuffle indexes of the train/validation/test dataset
    train_indexes = np.random.permutation(len(x_train))
    val_indexes = np.random.permutation(len(x_val))
    test_indexes = np.random.permutation(len(x_test))
    # Shuffle datasets together
    x_train = x_train[train_indexes]
    y_train = y_train[train_indexes]
    x_val = x_val[val_indexes]
    y_val = y_val[val_indexes]
    x_test = x_test[test_indexes]
    y_test = y_test[test_indexes]
    info("Train MRIs: %s" % str(len(x_train)))
    info("%d positive samples" % np.count_nonzero(y_train))
    info("%d negative samples" % (len(y_train) - np.count_nonzero(y_train)))
    info("Test MRIs: %s" % str(len(x_test)))
    info("%d positive samples" % np.count_nonzero(y_test))
    info("%d negative samples" % (len(y_test) - np.count_nonzero(y_test)))
    info("Validation MRIs: %s" % str(len(x_val)))
    info("%d positive samples" % np.count_nonzero(y_val))
    info("%d negative samples" % (len(y_val) - np.count_nonzero(y_val)))
    # Create data loaders
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    # test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # Augment the on the fly during training
    if augment == True:
        train_dataset = train_loader.map(train_preprocessing)
    else:
        train_dataset = train_loader.map(val_test_preprocessing)
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .batch(batch_size)
        .prefetch(buffer_size=AUTOTUNE)
    )
    # Only rescale
    val_dataset = (
        val_loader.shuffle(len(x_val))
        .map(val_test_preprocessing)
        .batch(batch_size)
        .prefetch(buffer_size=AUTOTUNE)
    )
    # test_dataset = (
    #     test_loader.shuffle(len(x_val))
    #     .map(val_test_preprocessing)
    #     .batch(batch_size)
    #     .prefetch(buffer_size=AUTOTUNE)
    # )
    x_test, y_test = val_test_preprocessing(x_test, y_test)
    return train_dataset, val_dataset, x_test, y_test # test_dataset

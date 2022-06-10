from utils import load_params, info, load_nii

from sklearn.model_selection import train_test_split
from glob import glob
import numpy as np
import os

if __name__ == "__main__":
    """Load data and build the train / val / test subsets."""
    info("Starting preprocessing")
    settings = load_params("settings.json")
    data_dir = settings["metadata"]["train_data_dir"]
    # normalization_size = tuple(settings["metadata"]["normalization_size"])
    abnormal_data = [load_nii(e) for e in glob(os.path.join(data_dir, "abnormal", "*.nii"))] # TODO normalization_size
    control_data = [load_nii(e) for e in glob(os.path.join(data_dir, "control", "*.nii"))] # TODO normalization_size
    abnormal_labels = [1. for _ in range(len(abnormal_data))]
    control_labels = [0. for _ in range(len(control_data))]
    assert len(abnormal_data) + len(control_data) == len(abnormal_labels) + len(control_labels)
    # Load settings
    data_rep = settings["preprocessing"]["data_rep"]
    random_seed = settings["preprocessing"]["random_seed"]
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
    control_x_train, control_x_test, control_y_train, control_y_test = train_test_split(
        control_data,
        control_labels,
        test_size = 1 - data_rep[0],
        shuffle = False
    )
    control_x_val, control_x_test, control_y_val, control_y_test = train_test_split(
        control_x_test,
        control_y_test,
        test_size = data_rep[1]/(data_rep[1]+data_rep[2]),
        shuffle = False
    )
    # Create the final train / val / test datasets and shuffle them
    x_train = np.concatenate((abnormal_x_train, control_x_train), axis=0)
    x_val = np.concatenate((abnormal_x_val, control_x_val), axis=0)
    x_test = np.concatenate((abnormal_x_test, control_x_test), axis=0)
    y_train = np.concatenate((abnormal_y_train, control_y_train), axis=0)
    y_val = np.concatenate((abnormal_y_val, control_y_val), axis=0)
    y_test = np.concatenate((abnormal_y_test, control_y_test), axis=0)
    np.random.seed(random_seed)
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
    info("Train MRIs: " + str(len(x_train)))
    info("Test MRIs: " + str(len(x_test)))
    info("Validation MRIs: " + str(len(x_val)))
    # Saving data
    np.save(os.path.join(data_dir, "x_train.npy"), x_train, allow_pickle=False)
    np.save(os.path.join(data_dir, "x_val.npy"), x_val, allow_pickle=False)
    np.save(os.path.join(data_dir, "x_test.npy"), x_test, allow_pickle=False)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train, allow_pickle=False)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val, allow_pickle=False)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test, allow_pickle=False)
    info("Preprocessing done")

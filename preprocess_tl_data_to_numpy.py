from utils import load_params, info, load_nii

from sklearn.model_selection import train_test_split
from glob import glob
import numpy as np
import os

if __name__ == "__main__":
    """Load data and build the train / val / test subsets."""
    info("Starting preprocessing")
    settings = load_params("settings.json")
    data_dir = settings["metadata"]["fine_tune_data_dir"]
    coma_data = [load_nii(e) for e in glob(os.path.join(data_dir, "coma", "*.nii"))]
    control_data = [load_nii(e) for e in glob(os.path.join(data_dir, "control", "*.nii"))]
    coma_labels = [1. for _ in range(len(coma_data))]
    control_labels = [0. for _ in range(len(control_data))]
    assert len(coma_data) + len(control_data) == len(coma_labels) + len(control_labels)
    # Load settings
    settings = load_params("settings.json")["preprocessing"]
    data_rep = settings["data_rep"]
    random_seed = settings["random_seed"]
    # Split them into coma and control train / val / test datasets
    # Data are not shuffled yet in order to get the same MRIs in the different subsets
    # So the dataloader can be called several times (in differents tasks)
    coma_x_train, coma_x_test, coma_y_train, coma_y_test = train_test_split(
        coma_data,
        coma_labels,
        test_size = 1 - data_rep[0],
        shuffle = False
    )
    coma_x_val, coma_x_test, coma_y_val, coma_y_test = train_test_split(
        coma_x_test,
        coma_y_test,
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
    x_train = np.concatenate((coma_x_train, control_x_train), axis=0)
    x_val = np.concatenate((coma_x_val, control_x_val), axis=0)
    x_test = np.concatenate((coma_x_test, control_x_test), axis=0)
    y_train = np.concatenate((coma_y_train, control_y_train), axis=0)
    y_val = np.concatenate((coma_y_val, control_y_val), axis=0)
    y_test = np.concatenate((coma_y_test, control_y_test), axis=0)
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
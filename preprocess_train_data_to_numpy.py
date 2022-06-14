from utils import load_params, info, load_nii

from sklearn.model_selection import train_test_split
from glob import glob
import numpy as np
import os

if __name__ == "__main__":
    """Load data and build the train / val / test subsets."""
    info("Starting preprocessing")
    # Load settings
    settings = load_params()
    data_dir = settings["metadata"]["train_data_dir"]
    data_rep = settings["preprocessing"]["data_rep"]
    random_seed = settings["preprocessing"]["random_seed"]
    normalization_size = tuple(settings["metadata"]["normalization_size"])
    del settings
    # Loading data
    abnormal_data = [
        load_nii(e, norm_type="ct-threshold", img_size=normalization_size, rotate_axes=[1, 2])
        for e in glob(os.path.join(data_dir, "abnormal", "*.nii"))
    ]
    control_data = [
        load_nii(e, norm_type="ct-threshold", img_size=normalization_size, rotate_axes=[1, 2])
        for e in glob(os.path.join(data_dir, "control", "*.nii"))
    ]
    del normalization_size
    abnormal_labels = [1. for _ in range(len(abnormal_data))]
    control_labels = [0. for _ in range(len(control_data))]
    assert len(abnormal_data) + len(control_data) == len(abnormal_labels) + len(control_labels)
    # Split them into abnormal and control train / val / test datasets
    # Data are not shuffled yet in order to get the same MRIs in the different subsets
    # So the dataloader can be called several times (in differents tasks)
    abnormal_x_train, abnormal_x_test, abnormal_y_train, abnormal_y_test = train_test_split(
        abnormal_data,
        abnormal_labels,
        test_size = 1 - data_rep[0],
        shuffle = False
    )
    del abnormal_labels
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
    del control_labels
    control_x_val, control_x_test, control_y_val, control_y_test = train_test_split(
        control_x_test,
        control_y_test,
        test_size = data_rep[1]/(data_rep[1]+data_rep[2]),
        shuffle = False
    )
    del data_rep
    # Create the final train / val / test datasets and shuffle them
    x_train = np.concatenate((abnormal_x_train, control_x_train), axis=0)
    x_val = np.concatenate((abnormal_x_val, control_x_val), axis=0)
    x_test = np.concatenate((abnormal_x_test, control_x_test), axis=0)
    y_train = np.concatenate((abnormal_y_train, control_y_train), axis=0)
    y_val = np.concatenate((abnormal_y_val, control_y_val), axis=0)
    y_test = np.concatenate((abnormal_y_test, control_y_test), axis=0)
    del abnormal_x_train
    del control_x_train
    del abnormal_x_val
    del control_x_val
    del abnormal_x_test
    del control_x_test
    del abnormal_y_train
    del control_y_train
    del abnormal_y_val
    del control_y_val
    del abnormal_y_test
    del control_y_test
    np.random.seed(random_seed)
    # Shuffle indexes of the train/validation/test dataset
    train_indexes = np.random.permutation(len(x_train))
    val_indexes = np.random.permutation(len(x_val))
    test_indexes = np.random.permutation(len(x_test))
    del random_seed
    # Shuffle datasets together
    x_train = x_train[train_indexes]
    y_train = y_train[train_indexes]
    del train_indexes
    x_val = x_val[val_indexes]
    y_val = y_val[val_indexes]
    del val_indexes
    x_test = x_test[test_indexes]
    y_test = y_test[test_indexes]
    del test_indexes
    info("Train MRIs: %s" % str(len(x_train)))
    info("- %d positive samples" % np.count_nonzero(x_train).shape[0])
    info("- %d negative samples" % len(x_train) - np.count_nonzero(x_train).shape[0])
    info("Test MRIs: %s" % str(len(x_test)))
    info("- %d positive samples" % np.count_nonzero(x_test).shape[0])
    info("- %d negative samples" % len(x_test) - np.count_nonzero(x_test).shape[0])
    info("Validation MRIs: %s" % str(len(x_val)))
    info("- %d positive samples" % np.count_nonzero(x_val).shape[0])
    info("- %d negative samples" % len(x_val) - np.count_nonzero(x_val).shape[0])
    # Saving data
    np.save(os.path.join(data_dir, "x_train.npy"), x_train, allow_pickle=False)
    np.save(os.path.join(data_dir, "x_val.npy"), x_val, allow_pickle=False)
    np.save(os.path.join(data_dir, "x_test.npy"), x_test, allow_pickle=False)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train, allow_pickle=False)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val, allow_pickle=False)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test, allow_pickle=False)
    info("Preprocessing done")

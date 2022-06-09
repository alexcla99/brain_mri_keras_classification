from math import sqrt
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import nibabel as nib
import json

SETTINGS_FILE = "settings.json"

def info(i:str, status:int=0) -> None:
    """Display a string as an information / error message / debug message."""
    if status == 0:
        print("[INFO] %s" % i)
    elif status == 1:
        print("[ERROR] %s" % i)
    elif status == 2:
        print("[DEBUG] %s" % i)
    else:
        print(i)

def load_params(src:str=SETTINGS_FILE) -> dict:
    """Load the project's settings contained in a JSON file."""
    with open(src, "r") as handle:
        params = json.loads(handle.read())
        handle.close()
    return params

def load_nii(path:str, new_size:tuple=None) -> np.ndarray:
    """Load a nifti file into a numpy array."""
    data = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    # Normalize data size if specified
    if new_size is not None:
        data = np.resize(data, new_size) # TODO: resize_volume
        data = np.rot90(data, axes=[1, 2])
    # Min-max normalize data
    data = (data - data.min()) / (data.max() - data.min())
    # Expand data dimensions
    data = np.expand_dims(data, axis=-1)
    return data

# def resize_volume(img):
#     """Resize across z-axis"""
#     # Set the desired depth
#     desired_depth = 64
#     desired_width = 128
#     desired_height = 128
#     # Get current depth
#     current_depth = img.shape[-1]
#     current_width = img.shape[0]
#     current_height = img.shape[1]
#     # Compute depth factor
#     depth = current_depth / desired_depth
#     width = current_width / desired_width
#     height = current_height / desired_height
#     depth_factor = 1 / depth
#     width_factor = 1 / width
#     height_factor = 1 / height
#     # Rotate
#     img = ndimage.rotate(img, 90, reshape=False)
#     # Resize across z-axis
#     img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
#     return img

# TODO: data augmentation

# Thanks to: stackoverflow.com/questions/39895742/matthews-correlation-coefficient-with-keras
def mcc(y_true, y_pred):
    """Compute the Matthews Correlation Coefficient."""
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())

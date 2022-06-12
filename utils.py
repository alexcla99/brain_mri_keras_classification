from scipy import ndimage
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

def load_nii(path:str, normalisation_size:tuple=None) -> np.ndarray:
    """Load a nifti file into a numpy array."""
    data = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    # Min-max normalize data
    data = (data - data.min()) / (data.max() - data.min())
    # Size normalization
    if normalisation_size is not None:
        data = resize_volume(data, normalisation_size)
    # Expand data dimensions
    # data = np.expand_dims(data, axis=-1)
    return data

def resize_volume(img:np.array, new_size:tuple) -> np.array:
    """Resize across z-axis"""
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / new_size[-1] # Desired depth
    width = current_width / new_size[0] # Desired width
    height = current_height / new_size[1] # Desired height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def expand_dims(volume:tf.Tensor) -> tf.Tensor:
    return tf.expand_dims(volume, axis=3)

# TODO: data augmentation

# Thanks to: stackoverflow.com/questions/39895742/matthews-correlation-coefficient-with-keras
def mcc(y_true:tf.Tensor, y_pred:tf.Tensor) -> tf.Tensor:
    """Compute the Matthews Correlation Coefficient."""
    y_pred = tf.where(y_pred >= .5, 1., 0.)
    y_pred_pos = K.round(K.clip(y_pred, 0., 1.))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0., 1.))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_value = numerator / (denominator + K.epsilon())
    return mcc_value

from scipy import ndimage
import tensorflow as tf
import numpy as np
import nibabel as nib
import json

SETTINGS_FILE = "settings.json"

# CUSTOM UTILS ######################################################################################################
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

# DATA PREPROCESSING ################################################################################################
def load_nii(path:str) -> np.ndarray:
    """Load a nifti file into an array."""
    data = nib.load(path).get_fdata()
    return data

def normalize(data:np.ndarray, method:str="minmax") -> np.ndarray:
    """Normalize a 3D image according to the specified method."""
    if method == "minmax":
        data = (data - data.min()) / (data.max() - data.min())
    elif method == "threshold":
        min_value = -1000
        max_value = 400
        data[data < min_value] = min_value
        data[data > max_value] = max_value
        data = (data - min_value) / (max_value - min_value)
    data = data.astype(np.float32)
    return data

def resize_volume(img:np.array, new_size:tuple) -> np.array:
    """Resize across z-axis"""
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

def process_scan(path:str, norm_type:str=None, img_size:tuple=None) -> np.ndarray:
    """Process a 3D image according to the wanted output."""
    data = load_nii(path)
    # Voxels normalization
    if norm_type is not None:
        data = normalize(data, method=norm_type)
    # Size normalization
    if img_size is not None:
        data = resize_volume(data, new_size=img_size)
    return data

# DATASETS PREPROCESSING ############################################################################################
@tf.function
def rotate(volume:tf.Tensor) -> tf.Tensor:
    """Rotate the volume by a fex degrees."""
    def scipy_rotate(volume):
        # Define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # Randomly select one
        angle = np.random.choice(angles)
        # Rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume
    # Calling the above function
    random_seed = load_params()["preprocessing"]["random_seed"]
    np.random.seed(random_seed)
    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

def train_preprocessing(volume:tf.Tensor, label:tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Preprocessing done for the train dataset."""
    volume = rotate(volume)
    volume = volume = tf.expand_dims(volume, axis=-1)
    return volume, label

def val_test_preprocessing(volume:tf.Tensor, label:tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Preprocessing done for both val / test datasets."""
    volume = volume = tf.expand_dims(volume, axis=-1)
    return volume, label

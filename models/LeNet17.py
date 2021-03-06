from utils import load_params

from tensorflow import keras
from tensorflow.keras import layers

# LENET17 MODEL BUILDING ############################################################################################
def get_model(width:int, height:int, depth:int) -> keras.Model:
    """Instanciate a LeNet-like 3D CNN architecture with 17 layers."""
    params = load_params()
    dropout = params["LeNet17"]["dropout"]
    num_classes = params["dataset"]["num_classes"]
    # Input layer
    inputs = keras.Input((width, height, depth, 1))
    # First convolutional block
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # Second convolutional block
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # Third convolutional block
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # Fourth convolutional block
    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # Classification block
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x) # TODO (units=num_classes, activation="softmax")
    # Define and return the model
    model = keras.Model(inputs, outputs, name="LeNet17")
    return model

from utils import load_params

from tensorflow import keras
from tensorflow.keras import layers

def get_model(img_size:list) -> keras.Model:
    """Instanciate a LeNet-like 3D CNN architecture with 17 layers."""
    params = load_params()
    dropout = params["model"]["dropout"]
    num_classes = params["dataset"]["num_classes"]
    # Input layer
    inputs = keras.Input(tuple(img_size))
    # First convolutional block
    x = layers.Conv3D(filters=64, kernel_size=3, activatio="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # Second convolutional block
    x = layers.Conv3D(filters=64, kernel_size=3, activatio="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # Third convolutional block
    x = layers.Conv3D(filters=128, kernel_size=3, activatio="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # Fourth convolutional block
    x = layers.Conv3D(filters=256, kernel_size=3, activatio="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # Classification block
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(units=num_classes, acivation="sigmoid")(x)
    # Define and return the model
    model = keras.Model(inputs, outputs, name="LeNet17")
    return model

from utils import info, load_params
from dataset import load_dataset
from tf_config import tf_configure

from tensorflow import keras
import tensorflow as tf
import os, sys, json, traceback

# MODEL FINE TUNING #################################################################################################
if __name__ == "__main__":
    """Main program to transfer learn any model."""
    metadata = load_params()["metadata"]
    available_models = metadata["available_models"]
    train_data_dir = metadata["ft_data_dir"]
    results_dir = metadata["results_dir"]
    img_size = metadata["img_size"]
    if len(sys.argv) != 2:
        print("Usage: python3 fine_tune.py <model:str>")
        print("Example: python3 fine_tune.py LeNet17")
        print("Available models:")
        for e in available_models:
            print("* %s" % e)
    else:
        model_name = str(sys.argv[1])
        try:
            # Starting a fresh session
            tf.keras.backend.clear_session()
            tf_configure()
            # Check if the selected model exists
            assert(model_name in available_models)
            # Load params
            info("Loading parameters")
            params = load_params()[model_name]
            # Build both train and validation datasets
            info("Building datasets")
            train_dataset, val_dataset, test_datast = load_dataset(train_data_dir)
            info("Using %d train samples and %d validation samples" % (
                len([_ for _ in train_dataset]),
                len([_ for _ in val_dataset])
            ))
            # Load the selected model
            info("Loading the selected model (%s)" % model_name)
            if model_name == available_models[0]:
                from models.LeNet17 import get_model
            # elif: # TODO others models
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = get_model(img_size[0], img_size[1], img_size[-1])
                model.load_weights(os.path.join(results_dir, model_name, "%s_train.h5" % model_name))
            info(model.summary())
            # Freeze the model except its classifier (the four last layers)
            info("Freezing the selected model except its classifier")
            model.trainable = False
            for i in range(3): # TODO different number of layers for others models?
                model.layers[-(1+i)].trainable = True
            for i in range(len(model.layers)):
                info("Trainable layer %d: %s" % (i, model.layers[i].trainable))
            # Compile the model
            info("Compiling the model")
            model.compile(
                loss=params["loss"],
                optimizer=keras.optimizers.Adam(learning_rate=params["ft_lr"]),
                metrics=params["metrics"]
            )
            # Define callbacks
            info("Defining callbacks")
            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                os.path.join(results_dir, model_name, "%s_ft.h5" % model_name),
                save_best_only=True,
                save_weights_only=True
            )
            early_stopping_cb = keras.callbacks.EarlyStopping(
                monitor="val_%s" % params["metrics"][0], # In settings.json, it should be "acc"
                patience=params["ft_patience"]
            )
            # Train the model
            info("Transfer learning the model")
            model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=params["epochs"],
                shuffle=True,
                verbose=2,
                callbacks=[checkpoint_cb, early_stopping_cb]
            )
            # Save model's history
            info("Saving model's history")
            history = {params["metrics"][0]: [], "val_%s" % params["metrics"][0]: [], "loss": [], "val_loss": []}
            for i, metric in enumerate([params["metrics"][0], "loss"]):
                history[metric].append(model.history.history[metric])
                history["val_%s" % metric].append(model.history.history["val_%s" % metric])
            with open(os.path.join(results_dir, model_name, "%s_ft_metrics.json" % model_name), "w+") as handle:
                handle.write(json.dumps(history))
                handle.close()
            # End of the program
            info("Fine tuning done")
        except:
            info(traceback.format_exc(), status=1)

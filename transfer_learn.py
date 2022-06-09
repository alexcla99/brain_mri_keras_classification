from utils import info, load_params, mcc
from dataset import load_dataset

from tensorflow import keras
import tensorflow as tf
import os, sys, json, traceback

if __name__ == "__main__":
    """Main program to transfer learn any model."""
    metadata = load_params()["metadata"]
    available_models = metadata["available_models"]
    train_data_dir = metadata["tl_data_dir"]
    results_dir = metadata["results_dir"]
    if len(sys.argv) != 2:
        print("Usage: python3 transfer_learn.py <model:str>")
        print("Example: python3 transfer_learn.py LeNet17")
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
            train_dataset, val_dataset = load_dataset(train_data_dir)
            # assert(train_dataset.get_single_element().shape == val_dataset.get_single_element().shape)
            info("Using %d train samples and %d validation samples" % (
                len([e for e in train_dataset]),
                len([e for e in val_dataset])
            ))
            # Load the selected model
            info("Loading the selected model (%s)" % model_name)
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = keras.models.load_model(os.path.join(results_dir, model_name, "%s_train.h5" % model_name))
            info(model.summary)
            # Freeze the model except its classifier (the four last layers)
            info("Freezing the selected model except its classifier")
            model.trainable = False
            for i in range(4): # TODO different number of layers for others models?
                model.layers[-(1+i)].trainable = True
            # Compile the model
            info("Compiling the model")
            model.compile(
                loss=params["loss"],
                optimizer=keras.optimizers.Adam(learning_rate=params["tl_lr"]),
                metrics=[mcc]
            )
            # Define callbacks
            info("Defining callbacks")
            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                os.path.join(results_dir, model_name, "%s_tl.h5" % model_name),
                save_best_only=True
            )
            early_stopping_cb = keras.callbacks.EarlyStopping(
                monitor="val_mcc",
                patience=params["tl_patience"]
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
            history = {"mcc": [], "val_mcc": [], "loss": [], "val_loss": []}
            for i, metric in enumerate(["mcc", "loss"]):
                history[metric].append(model.history.history[metric])
                history["val_%s" % metric].append(model.history.history["val_%s" % metric])
            with open(os.path.join(results_dir, model_name, "%s_metrics_tl.json" % model_name), "w+") as handle:
                handle.write(json.dumps(history))
                handle.close()
            # End of the program
            info("Transfer learning done")
        except:
            info(traceback.format_exc(), status=1)

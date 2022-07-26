from utils import load_params
from dataset import load_dataset

from tensorflow import keras
import tensorflow as tf
import os, json, traceback, sys

# MODEL FINE TUNING #################################################################################################
def run_fine_tune(model_name:str, settings:dict, res_folder:str, balance:bool):
    try:
        # Starting a fresh session
        tf.print("##### FINE TUNING PHASE #####", output_stream=sys.stdout)
        # Load params
        tf.print("Loading parameters", output_stream=sys.stdout)
        params = load_params()
        train_data_dir = params["metadata"]["ft_data_dir"]
        available_models = params["metadata"]["available_models"]
        batch_size=params["dataset"]["batch_size"]
        params = params[model_name]
        results_dir = os.path.join("results", model_name, res_folder)
        img_size = settings["normalization_size"]
        method = settings["method"]
        # Build both train and validation datasets
        tf.print("Building datasets", output_stream=sys.stdout)
        train_dataset, val_dataset, x_test, y_test = load_dataset(
            train_data_dir,
            augment=settings["data_augmentation"],
            norm_type=settings["normalization_type"],
            img_size=img_size,
            balance=balance
        )
        tf.print("Using %d train samples and %d validation samples" % (
            len([_ for _ in train_dataset]),
            len([_ for _ in val_dataset])
        ), output_stream=sys.stdout)
        # Load the selected model
        tf.print("Loading the selected model (%s)" % model_name)
        if model_name == available_models[0]:
            from models.LeNet17 import get_model
        # elif: # TODO others models
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        # with mirrored_strategy.scope():
        model = get_model(img_size[0], img_size[1], img_size[-1])
        model.load_weights(os.path.join(results_dir, "%s.h5" % res_folder))
        model.summary()
        # Freeze the model except its classifier (the four last layers)
        tf.print("Freezing the selected model except its classifier", output_stream=sys.stdout)
        if method == "tl":
            model.trainable = False
            for i in range(3): # TODO different number of layers for others models?
                model.layers[-(1+i)].trainable = True
        elif method == "ft":
            model.trainable = True
        else:
            raise Exception("Unknow method (not tl. nor ft.!)")
        for i in range(len(model.layers)):
            tf.print("Trainable layer %d: %s" %(i, str(model.layers[i].trainable)), output_stream=sys.stderr)
        # Compile the model
        tf.print("Compiling the model", output_stream=sys.stdout)
        model.compile(
            loss=params["loss"],
            optimizer=keras.optimizers.Adam(learning_rate=params["ft_lr"]),
            metrics=params["metrics"]
        )
        # Define callbacks
        tf.print("Defining callbacks", output_stream=sys.stdout)
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "%s_%s.h5" % (res_folder, method)),
            save_best_only=True,
            save_weights_only=True
        )
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor="val_%s" % params["metrics"][0], # In settings.json, it should be "acc"
            patience=params["ft_patience"]
        )
        # Train the model
        tf.print("Transfer learning the model", output_stream=sys.stdout)
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=params["epochs"],
            shuffle=True,
            verbose=2,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )
        # Test the model
        tf.print("Testing the transfer learned model", output_stream=sys.stdout)
        evaluation = model.evaluate(
            x_test,
            y_test,
            batch_size=batch_size,
            verbose=2,
            return_dict=True
        )
        # Save model's history
        tf.print("Saving model's history", output_stream=sys.stdout)
        full_history = {
            params["metrics"][0]: [],
            "val_%s" % params["metrics"][0]: [],
            "loss": [],
            "val_loss": [],
            "evaluation": evaluation
        }
        for i, metric in enumerate([params["metrics"][0], "loss"]):
            full_history[metric].append(history.history[metric])
            full_history["val_%s" % metric].append(history.history["val_%s" % metric])
        with open(os.path.join(results_dir, "%s_ft_metrics.json" % res_folder), "w+") as handle:
            handle.write(json.dumps(full_history))
            handle.close()
        # End of the program
        tf.print("Fine tuning done", output_stream=sys.stdout)
    except:
        tf.print(traceback.format_exc(), output_stream=sys.stderr)

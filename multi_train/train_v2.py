from utils import load_params
from dataset import load_dataset

from tensorflow import keras
import tensorflow as tf
import os, json, traceback, sys

# MODEL TRAINING ####################################################################################################
def run_train(model_name:str, settings:dict, res_folder:str, balance:bool, healthy_reduction:int) -> None:
    """Main program to train any model from scratch."""
    try:
        # Starting a fresh session
        tf.print("##### TRAIN PHASE #####", output_stream=sys.stdout)
        settings_keys = ", ".join([k for k in settings.keys()])
        settings_values = ", ".join([str(v) for v in settings.values()])
        tf.print(
            "Using the current settings:\n%s\n%s" % (str(settings_keys), str(settings_values)),
            output_stream=sys.stdout
        )
        # Load params
        tf.print("Loading parameters", output_stream=sys.stdout)
        params = load_params()
        train_data_dir = params["metadata"]["train_data_dir"]
        available_models = params["metadata"]["available_models"]
        batch_size=params["dataset"]["batch_size"]
        params = params[model_name]
        results_dir = os.path.join("results", model_name, res_folder)
        img_size = settings["normalization_size"]
        # Build both train and validation datasets
        tf.print("Building datasets", output_stream=sys.stdout)
        train_dataset, val_dataset, x_test, y_test = load_dataset(
            train_data_dir,
            augment=settings["data_augmentation"],
            norm_type=settings["normalization_type"],
            img_size=img_size,
            balance=balance,
            healthy_reduction=healthy_reduction
        )
        tf.print("Using %d train samples and %d validation samples" % (
            len([_ for _ in train_dataset]),
            len([_ for _ in val_dataset])
        ), output_stream=sys.stdout)
        # Load the selected model
        tf.print("Loading the selected model (%s)" % model_name, output_stream=sys.stdout)
        if model_name == available_models[0]:
            from models.LeNet17 import get_model
        # elif: # TODO others models
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        # with mirrored_strategy.scope():
        model = get_model(img_size[0], img_size[1], img_size[-1])
        model.summary(print_fn=tf.print(output_stream=sys.stdout))
        # Compile the model
        tf.print("Compiling the model", output_stream=sys.stdout)
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            params["initial_lr"],
            decay_steps=1e6,
            decay_rate=.96,
            staircase=True
        )
        model.compile(
            loss=params["loss"],
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=params["metrics"]
        )
        # Define callbacks
        tf.print("Defining callbacks", output_stream=sys.stdout)
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "%s.h5" % res_folder),
            save_best_only=True,
            save_weights_only=True
        )
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor="val_%s" % params["metrics"][0], # In settings.json, it should be "acc"
            patience=params["patience"]
        )
        # Train the model
        tf.print("Training the model", output_stream=sys.stdout)
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=params["epochs"],
            shuffle=True,
            verbose=2,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )
        # Test the model
        tf.print("Testing the trained model", output_stream=sys.stdout)
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
        with open(os.path.join(results_dir, "%s_metrics.json" % res_folder), "w+") as handle:
            handle.write(json.dumps(full_history))
            handle.close()
        # End of the program
        tf.print("Training done", output_stream=sys.stdout)
    except:
        tf.print(traceback.format_exc(), output_stream=sys.stderr)

from utils import info, load_params
from dataset import get_test_dataset
from tf_config import tf_configure

from tensorflow import keras
import tensorflow as tf
import numpy as np
import os, sys, traceback

if __name__ == "__main__":
    """Main program to test any trained model from scratch."""
    metadata = load_params()["metadata"]
    available_models = metadata["available_models"]
    tl_data_dir = metadata["tl_data_dir"]
    results_dir = metadata["results_dir"]
    if len(sys.argv) != 2:
        print("Usage: python3 test_transfer_learned_model.py <model:str>")
        print("Example: python3 test_transfer_learned_model.py LeNet17")
        print("Available models:")
        for e in available_models:
            print("* %s" % e)
    else:
        model_name = str(sys.argv[1])
        try:
            # Starting a fresh session
            tf.keras.backend.clear_session()
            tf_configure()
            # Load the test dataset
            info("Loading the test dataset")
            test_dataset = get_test_dataset(tl_data_dir)
            # Load the selected model
            info("Loading the selected model (%s)" % model_name)
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = keras.models.load_model(os.path.join(results_dir, model_name, "%s_tl.h5" % model_name))
            info(model.summary())
            # Make predictions
            info("Making predictions")
            predictions = list()
            ground_truths = list()
            for e in test_dataset:
                prediction = model.predict(e[0])
                for p in prediction:
                    predictions.append(p[0])
                ground_truth = e[1].numpy()
                for g in ground_truth:
                    ground_truths.append(g)
            # Save predictions and ground truths
            info("Saving predictions")
            np.save(
                os.path.join(results_dir, "test_tl_model_predictions.npy"),
                np.array(predictions, model_name, dtype=np.float32), allow_pickle=False
            )
            np.save(
                os.path.join(results_dir, "test_tl_model_ground_truths.npy"),
                np.array(ground_truths, model_name, dtype=np.float32), allow_pickle=False
            )
            # End of the program
            info("Test of the transfer learned model done")
        except:
            info(traceback.format_exc(), status=1)

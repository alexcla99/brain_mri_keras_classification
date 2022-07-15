# Imports
from tf_config import tf_configure
from train_v2 import run_train
from fine_tune_v2 import run_fine_tune
import tensorflow as tf
import os, sys

# Main class to cross-test multiple parameters
class ModelTrainer:

    # Builder
    def __init__(self, model_name:str) -> None:
	# Parameters to try
        self.__normalisation_sizes:list = [
            # (512, 512, 32), # Minimal shape computed in "Debug.py"
            # (512, 512, 72), # Maximal shape computed in "Debug.py"
            (128, 128, 64),
            (113, 113, 117)
        ]
        self.__normalization_types:list = ["minmax", "threshold"]
        self.__methods:list = ["tl", "ft"]
        self.__data_augmentation:list = [True, False]
        # The model to be tested
        self.__model_name:str = model_name
        # Current selected settings
        self.__current_settings:dict = None
        # Test counter
        self.__counter:int = 0
        # The folder containing results
        self.__res_folder:str = ""
        # The output configurations
        self.__output:str = ""

    # Tester
    def run_trains(self) -> None:
        # Looping over parameters
        for normalization_size in self.__normalisation_sizes:
            for normalization_type in self.__normalization_types:
                for method in self.__methods:
                    for data_augmentation in self.__data_augmentation:
                        # Gathering current settings
                        self.__current_settings = {
                            "normalization_size": normalization_size, # OK (for both TRAIN and FT)
                            "normalization_type": normalization_type, # OK (for both TRAIN and FT)
                            "data_augmentation": data_augmentation,   # OK
                            "method": method                          # OK
                        }
                        self.__res_folder = "train_%d" % self.__counter
                        # Running the whole phases with the selected settings
                        tf.print("##### ITERATION %d #################################################################################" % self.__counter, output_stream=sys.stdout)
                        os.mkdir(os.path.join("results", self.__model_name, self.__res_folder))
                        run_train(
                            self.__model_name,
                            self.__current_settings,
                            self.__res_folder
                        )
                        run_fine_tune(
                            self.__model_name,
                            self.__current_settings,
                            self.__res_folder
                        )
                        tf.print("####################################################################################################", output_stream=sys.stdout)
                        # Updating the output
                        self.__output += "Counter %d: %s\n" % (self.__counter, str(self.__current_settings))
                        # Increasing the counter
                        self.__counter += 1
        # Saving the output
        with open(os.path.join("results", self.__model_name, "train_all_output.txt"), "w+") as f:
            f.write(self.__output)
            f.close()
        tf.print("Done!")

# Main program
if __name__ == "__main__":
    # Starting a fresh session
    tf.keras.backend.clear_session()
    tf_configure()
    # Running on multiple GPUs
    # tf.debugging.set_log_device_placement(True)
    # gpus = tf.config.list_logical_devices('GPU')
    # strategy = tf.distribute.MirroredStrategy(gpus)
    # with strategy.scope():
    with tf.device("/device:GPU:0"):
        # Instanciating the trainer
        model_trainer = ModelTrainer(model_name="LeNet17")
        # Running multiple configurations
        model_trainer.run_trains()

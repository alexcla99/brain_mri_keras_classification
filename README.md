# Compairing performances across multiple Keras models by training them and applying transfer learning on 3D images datasets.

Author: alexcla99  
Version: 2.0.0

### Folder content:

```
+-+- multi_train              # A folder containing scripts to test a model with cross configurations
| +--- dataset.py             # An override of the original dataset.py file
| +--- fine_tune_v2.py        # A new version of fine_tune.py file
| +--- train_all.py           # A script to run both training and fine tuning with multiple configurations
| +--- train_v2.py            # A new version of train.py file
|
+-+- models/                  # The folder containing available models
| +--- LeNet17.py             # The LeNet17 model
|
+--- results/                 # The folder containing the train, transfer learning and tests results
+--- train_data/              # The folder containing the dataset for training from scratch
+--- ft_data/                 # The folder containing the dataset for fine tuning
+--- __init__.py              # An empty file to make this directory being a Python library
+--- dataset.py               # The dataset loader
+--- fine_tune.py             # A script to apply fine tuning on a model
+--- README.md                # This file
+--- requirements.txt         # The Python libraries to be installed in order to run the project
+--- settings.json            # The settings of the model and the train phase
+--- test_trained_model.py    # A script to test a trained model
+--- test_fine_tuned_model.py # A script to test a fine tuned model
+--- tf_config.py             # A script to configure TensorFlow
+--- train.py                 # A script to train from scratch a model
+--- utils.py                 # Some utils
```

### Usage:

This library has been implemented and used with Python>=3.8.0

Requirements:
```Shell
pip3 install -r requirements
```

Train a model:
```Shell
python3 train.py <model:str>
# Example: python3 train.py LeNet17
```
Data to be used are selected from the "train_data" folder and results are saved in the "results" folder.

Available networks:
See the `models` folder.

Fine tune a model:
```Shell
python3 fine_tune.py <model:str>
# Example: python3 fine_tune.py LeNet17
```
Data to be used are selected from the "ft_data" folder and results are saved in the "results" folder.

Test a trained model:
```Shell
python3 test_trained_model.py <model:str>
# Example: python3 test_trained_model.py LeNet17
```

Test a fine tuned model:
```Shell
python3 test_fine_tuned_model.py <model:str>
# Example: python3 test_fine_tuned_model.py LeNet17
```

### Many thanks to:

[1] H. Zunair., "3D images classification from CT scans.", [keras.io](https://keras.io/examples/vision/3D_image_classification/), 2020.  
[2] Zunair et al., "Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction", [arXiv](https://arxiv.org/pdf/2007.13224.pdf), 2020.  

License: [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0).

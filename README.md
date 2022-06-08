# Compairing performances across multiple Keras models by training them and applying transfer learning on 3D images datasets.

Author: alexcla99
Version: 1.0.0

### Folder content:

```
+-+- models/                           # The folder containing available models
| +--- LeNet17.py                      # The LeNet17 model
|
+--- results/                          # The folder containing the train, transfer learning and tests results
+--- train_data/                       # The folder containing the dataset for training from scratch
+--- tl_data/                          # The folder containing the dataset for transfer learning
+--- __init__.py                       # An empty file to make this directory being a Python library
+--- dataset.py                        # The dataset loader
+--- preprocess_tl_data_to_numpy.py    # A script to preprocess the tl dataset and store it into numpy files
+--- preprocess_train_data_to_numpy.py # A script to preprocess the train dataset and store it into numpy files
+--- README.md                         # This file
+--- requirements.txt                  # The Python libraries to be installed in order to run the project
+--- settings.json                     # The settings of the model and the train phase
+--- test_trained_model.py             # A script to test a trained model
+--- test_transfer_learned_model.py    # A script to test a transfer learned model
+--- train.py                          # A script to train from scratch a model
+--- transfer_learn.py                 # A script to apply transfer learning on a model
+--- utils.py                          # Some utils
```

### Usage:

This library has been implemented and used with Python>=3.8.0

Requirements:
```Shell
pip3 install -r requirements
```

Preprocess data:
```Shell
python3 preprocess_<dataset>_to_numpy.py
```
Data are loaded from from train_data or tl_data (depending on which script you want to run) in order to store them in numpy files.

Train a model:
```Shell
python3 train.py <model:str>
# Example: python3 train.py LeNet17
```
Data to be used are selected from the "train_data" folder and results are saved in the "results" folder.

Available networks:
See the `models` folder.

Apply transfer learning on a model:
```Shell
python3 transfer_learn.py <model:str>
# Example: python3 transfer_learn.py LeNet17
```
Data to be used are selected from the "tl_data" folder and results are saved in the "results" folder.

Test a trained model:
```Shell
python3 test_trained_model.py <model:str>
# Example: python3 test_trained_model.py LeNet17
```

Test a transfer learned model:
```Shell
python3 test_transfer_learned_model.py <model:str>
# Example: python3 test_transfer_learned_model.py LeNet17
```

### Many thanks to:

[1] H. Zunair., "3D images classification from CT scans.". [Keras.io](https://keras.io/examples/vision/3D_image_classification/). 2020.  
[2] Zunair et al., "Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction". [arXiv](https://arxiv.org/pdf/2007.13224.pdf). 2020.  

License: [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0).

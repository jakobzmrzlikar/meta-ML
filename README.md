# meta-ML

## A meta-ML model for predicting the accuracy of deep neural networks on certain datasets. Formerly part of the [DataBench](https://www.databench.eu/) project.

### Dependencies:
*  [Keras](https://github.com/keras-team/keras)
*  [Tensorflow](https://www.tensorflow.org/)
*  [numpy](https://www.numpy.org/)
*  [scikit-learn](https://scikit-learn.org/stable/)
*  [psutil](https://pypi.org/project/psutil/)

### Usage
The _config_ directory contains JSON files. You must first specify your model's features and which dataset to use. Leave the _results_ section empty. After you input all the necessary parameters, simply run main.py and the _results_ section will be automatically updated. The metadataset is automatically updated as well each time a model is run on a dataset.

You can alternatively only specify certain model architecture parameters and run config_generator.py to generate new config files with different hyperparameter values.

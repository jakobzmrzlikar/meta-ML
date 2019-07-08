# DataBench

Dependencies:
*  [Keras](https://github.com/keras-team/keras)
*  [Tensorflow](https://www.tensorflow.org/)
*  [numpy](https://www.numpy.org/)
*  [scikit-learn](https://scikit-learn.org/stable/)
*  [psutil](https://pypi.org/project/psutil/)

# Usage
The "config" directory contains JSON files. You must first specify your model's features and which dataset to use. Leave the "results" section empty. After you input all the necessary parameters, simply run main.py and the "results" section will be automatically updated. The metadataset is automatically updated as well each time a model is run on a dataset.

You can alternatively only specify certain model architecture parameters and run config_generator to generate new config files with different hyperparameter values.

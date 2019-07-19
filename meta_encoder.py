import json
import numpy as np
from keras.utils import to_categorical

def bin_encode(param, dic):
    if param in dic:
        return 1
    return 0

def encode(path):
    with open(path, "r") as f:
        meta = json.load(f)

    with open("encoding.json", 'r') as e:
        encoding = json.load(e)

    dataset = meta["dataset"]
    conf = meta["model"]
    arch = conf["architecture"]
    build = conf["hyperparameters"]["compile"]
    fit = conf["hyperparameters"]["fit"]
    evaluate = conf["hyperparameters"]["evaluate"]
    results = meta["results"]
    data = []


    # encode dataset
    data += to_categorical(encoding["data_type"][dataset["type"]], num_classes=len(encoding["data_type"]), dtype='int')
    data.append(dataset["instances"])
    data.append(dataset["features"])


    # encode model type
    data.append(to_categorical(encoding["model_type"][conf["type"]], num_classes=len(encoding["model_type"]), dtype='int'))


    # encode keras architecture
    tmp = [0] * len(encoding["activation"])
    if bin_encode("Activation", arch):
        for a, num in zip(arch["Activation"], arch["Dense"]):
            tmp[encoding["activation"][a]] += num
    data += tmp


    # encode compile hyperparameters for Sequential
    if bin_encode("optimizer", build):
        data += to_categorical(encoding["optimizer"][build["optimizer"]], num_classes=len(encoding["optimizer"]), dtype='int')
    else: 
        data += [0] * len(encoding["optimizer"])
    if bin_encode("loss", build):
        data += to_categorical(encoding["loss"][build["loss"]],
        num_classes=len(encoding["loss"]), dtype='int')
    else:
        data += [0] * len(encoding["loss"])
    data.append(bin_encode("loss_weights", build))
    data.append(bin_encode("sample_weight_mode", build))
    data.append(bin_encode("wighted_metrics", build))
    data.append(bin_encode("target_tensors", build))


    # encode compile hyperparameters for SVC, NuSVC and LinearSVC
    if bin_encode("C", build):
        data.append(build["C"])
    else:
        data.append(1.0)
    if bin_encode("nu", build):
        data.append(build["nu"])
    else:
        data.append(0.5)
    if bin_encode("kernel", build):
        data += to_categorical(encoding["kernel"][build["kernel"]], num_classes=len(encoding["kernel"]), dtype='int')
    else:
        data += [0] * len(encoding["kernel"])
    if bin_encode("degree", build):
        data.append(build["degree"])
    else:
        data.append(3)
    if bin_encode("gamma", build):
        if build["gamma"] == 'scale':
            data.append(1/dataset["features"])
        elif build["gamma"] == 'auto':
            data.append(1/dataset["features"])
        else:
            data.append(build["gamma"])
    else:
        data.append(1/dataset["features"])
    if bin_encode("coef0", build):
        data.append(build["coef0"])
    else:
        data.append(0.0)
    if bin_encode("shrinking", build):
        data.append(int(build["shrinking"]))
    else:
        data.append(1)
    if bin_encode("probability", build):
        data.append(int(build["probability"]))
    else:
        data.append(1)
    if bin_encode("penalty", build):
        if build["penalty"] == 'l1':
            data += [1, 0]
    else:
        data += [0, 1]
    data.append(bin_encode("dual", build))
    if bin_encode("tol", build):
        data.append(build["tol"])
    else:
        data.append(1e-3)
    if bin_encode("cache_size", build):
        data.append(build["cache_size"])
    else:
        data.append(200)
    if bin_encode("max_iter", build):
        data.append(build["max_iter"])
    else:
        data.append(-1)
    if bin_encode("decision_function_shape", build):
        data += to_categorical(encoding["decision_function_shape"][build["decision_function_shape"]], num_classes=len(encoding["decision_function_shape"]), dtype='int')
    else:
        data += [1, 0]
    data.append(bin_encode("random_state", build))
    if bin_encode("multi_class", build):
        if build["multi_class"] == 'crammer_singer':
            data += [0, 1]
    else:
        data += [1, 0]
    if bin_encode("fit_intercept", build):
        data.append(int(build["fit_intercept"]))
    else:
        data.append(1)
    if bin_encode("intercept_scaling", build):
        data.append(build["intercept_scaling"])
    else:
        data.append(1.0)
    

    # encode compile hyperparameters for KNeighborsClassifier and RadiusNeighborsClassifier
    if bin_encode("n_neighbors", build):
        data.append(build["n_neighbors"])
    else:
        data.append(5)
    data.append(bin_encode("weights", build))
    if bin_encode("algorithm", build):
        data += to_categorical(encoding["knn_algorithm"][build["algorithm"]], num_classes=len(encoding["knn_algorithm"]), dtype='int')
    else:
        data += [0] * len(encoding["knn_algorithm"])
    if bin_encode("leaf_size", build):
        data.append(build["leaf_size"])
    else:
        data.append(30)
    if bin_encode("p", build):
        data.append(build["p"])
    else:
        data.append(2)
    if bin_encode("metric", build):
        data += to_categorical(encoding["metric"][build["metric"]],     num_classes=len(encoding["metric"]), dtype='int')
    else:
        data += to_categorical(encoding["metric"]["minkowski"],
        num_classes=len(encoding["metric"]), dtype='int')
    data.append(bin_encode("metric_params", build))
    if bin_encode("n_jobs", build):
        data.append(build["n_jobs"])
    else:
        data.append(1)
    if bin_encode("radius", build):
        data.append(build["radius"])
    else:
        data.append(1.0)
    data.append(bin_encode("outlier_label", build))


    # encode compile hyperparameters for GaussianNB, MultinomialNB, ComplementNB and BernoulliNB
    data.append(bin_encode("priors", build))
    if bin_encode("var_smoothing", build):
        data.append(build["var_smoothing"])
    else:
        data.append(1e-9)
    if bin_encode("alpha", build):
        data.append(build["alpha"])
    else:
        data.append(1.0)
    if bin_encode("fit_prior", build):
        data.append(int(build["fit_prior"]))
    else:
        data.append(1)
    data.append(bin_encode("class_prior", build))
    if bin_encode("norm", build):
        data.append(int(build["norm"]))
    else:
        data.append(0)
    if bin_encode("binarize", build):
        data.append(build["binarize"])
    else:
        data.append(0.0)

    # encode compile hyperparameters for DecisionTreeClassifier
    if bin_encode("criterion", build):
        if build["criterion"] == 'entropy':
            data += [0, 1]
    else:
        data += [1, 0]
    if bin_encode("splitter", build):
        if build["splitter"] == 'random':
            data += [0, 1]
    else:
        data += [1, 0]
    if bin_encode("max_depth", build):
        data.append(build["max_depth"])
    else:
        data.append(1e100)
    if bin_encode("min_samples_split", build):
        data.append(build["min_samples_split"])
    else:
        data.append(2)
    if bin_encode("min_samples_leaf", build):
        data.append(build["min_samples_leaf"])
    else:
        data.append(1)
    if bin_encode("min_weight_fraction_leaf", build):
        data.append(build["min_weight_fraction_leaf"])
    else:
        data.append(0.0)
    if bin_encode("max_features", build):
        if build["max_features"] in ['auto', 'sqrt']:
            data.append(np.sqrt(dataset["features"]))
        elif build["max_features"] == 'log2':
            data.append(np.log2(dataset["features"]))
        else:
            data.append(build["max_features"])
    else:
        data.append(dataset["features"])
    if bin_encode("max_leaf_nodes", build):
        data.append(build["max_leaf_nodes"])
    else:
        data.append(1e100)
    if bin_encode("min_impurity_decrease", build):
        data.append(build["min_impurity_decrease"])
    else:
        data.append(0.0)
    if bin_encode("min_impurity_split", build):
        data.append(build["min_impurity_split"])
    else:
        data.append(1e-7)
    if bin_encode("presort", build):
        data.append(int(build["presort"]))
    else:
        data.append(0)


    # encode fit hyperparameters
    if bin_encode("batch_size", fit):
        data.append(fit["batch_size"])
    else:
        data.append(32)
    if bin_encode("epochs", fit):
        data.append(fit["epochs"])
    else:
        data.append(1)
    data.append(bin_encode("callbacks", fit))
    if bin_encode("validation_split", fit):
        data.append(fit["validation_split"])
    else:
        data.append(0.0)
    data.append(bin_encode("validation_data", fit))
    if bin_encode("shuffle", fit):
        data.append(int(fit["Shuffle"]))
    else:
        data.append(1)
    data.append(bin_encode("class_weight", fit))
    data.append(bin_encode("sample_weight", fit))
    if bin_encode("initial_epoch", fit):
        data.append(fit["initial_epoch"])
    else:
        data.append(0)
    if bin_encode("steps_per_epoch", fit):
        data.append(fit["steps_per_epoch"])
    else:
        data.append(1)
    if bin_encode("validation_steps", fit):
        data.append(fit["validation_steps"])
    else:
        data.append(0)
    data.append(bin_encode("validation_freq", fit))
    

    # encode evaluation hyperparameters
    if bin_encode("batch_size", build):
        data.append(build["batch_size"])
    else:
        data.append(32)
    data.append(bin_encode("sample_weight", fit))
    if bin_encode("steps", evaluate):
        data.append(evaluate["steps"])
    else:
        data.append(0)
    data.append(bin_encode("callbacks", evaluate))


    # encode results
    data.append(round(100*results["acc"]))
    data.append(results["time"])
    return data

import json
import numpy as np


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert a list into a numpy array
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y


def get_dataset_path(config_file="config.json"):
    """Loads and returns the dataset path from the JSON config file."""
    with open(config_file, "r") as f:
        config = json.load(f)
    return config.get("DATASET_PATH", "")

import os
import yaml
from pathlib import Path
parent = os.path.dirname(os.path.abspath(__file__)) + '/../'
CONFIG_FILE = parent+os.sep+"config.yml"

with open(CONFIG_FILE) as f:
    parsed_yaml_file = yaml.load(f, Loader=yaml.FullLoader)

def get_source_max_length():
    algs = parsed_yaml_file['source_max_length']
    if algs is None:
        raise IndexError("source_max_length must not be empty in configuration file")

    return algs


def get_accumulate_batches():
    algs = parsed_yaml_file['accumulate_grad_batches']
    if algs is None:
        raise IndexError("accumulate_grad_batches must not be empty in configuration file")

    return algs


def get_target_max_length():
    algs = parsed_yaml_file['target_max_length']
    if algs is None:
        raise IndexError("source_max_length must not be empty in configuration file")

    return algs


def get_batch_size():
    algs = parsed_yaml_file['batch_size']
    if algs is None:
        raise IndexError("batch size must not be empty in configuration file")

    return algs


def get_learning_rate():
    algs = parsed_yaml_file['learning_rate']
    if algs is None:
        raise IndexError("learning rate must not be empty in configuration file")

    return algs


def get_max_epochs():
    algs = parsed_yaml_file['max_epochs']
    if algs is None:
        raise IndexError("source_max_length must not be empty in configuration file")

    return algs


def get_model_name():
    algs = parsed_yaml_file['model_name']
    if algs is None:
        raise IndexError("model name must not be empty in configuration file")

    return algs

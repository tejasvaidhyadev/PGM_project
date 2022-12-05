import logging
import json
import re
import pandas as pd
import pdb
# Logistic Regression
from sklearn.linear_model import LogisticRegression
import torch
import math

def logit(p):
    return torch.log(p / (1 - p))   


def platt_scale(outcome, probs):
    logits = logit(probs)
    logits = logits.reshape(-1, 1)
    log_reg = LogisticRegression(penalty='none', warm_start=True, solver='lbfgs')
    log_reg.fit(logits, outcome)
    return log_reg.predict_proba(logits)


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def make_bow_vector(ids, vocab_size, use_counts=False):
    """ Make a sparse BOW vector from a tensor of dense ids.
    Args:
        ids: torch.LongTensor [batch, features]. Dense tensor of ids.
        vocab_size: vocab size for this tensor.
        use_counts: if true, the outgoing BOW vector will contain
            feature counts. If false, will contain binary indicators.
    Returns:
        The sparse bag-of-words representation of ids.
    """
    vec = torch.zeros(ids.shape[0], vocab_size)
    ones = torch.ones_like(ids, dtype=torch.float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if CUDA:
        vec = vec.cuda()
        ones = ones.cuda()
        ids = ids.cuda()

    vec.scatter_add_(1, ids, ones)
    vec[:, 1] = 0.0  # zero out pad
    if not use_counts:
        vec = (vec != 0).float()
    return vec

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

# ToDo use the below function for saving params file 
class Params():
    """
    Class that loads hyperparameters from a pretty json file.
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
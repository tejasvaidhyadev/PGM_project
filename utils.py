import logging
import json
import re
import pandas as pd
import pdb
# Logistic Regression
from sklearn.linear_model import LogisticRegression
import torch
import math
CUDA = (torch.cuda.device_count() > 0)

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


def psi_q_only(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    ite_t = (q_t1 - q_t0)[t == 1]
    estimate = ite_t.mean()
    return estimate


def psi_plugin(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    ite_t = g*(q_t1 - q_t0)/prob_t
    estimate = ite_t.mean()
    return estimate


def psi_aiptw(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):
    # the robust ATT estimator described in eqn 3.9 of
    # https://www.econstor.eu/bitstream/10419/149795/1/869216953.pdf

    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)
    estimate = (t*(y-q_t0) - (1-t)*(g/(1-g))*(y-q_t0)).mean() / prob_t

    return estimate


def psi_very_naive(t, y):
    return y[t == 1].mean() - y[t == 0].mean()


def att_estimates(q0, q1, g, t, y, prob_t, truncate_level=0.05, deps=0.0001):

    one_step_tmle = make_one_step_tmle(prob_t, deps_default=deps)

    very_naive = psi_very_naive(t,y)
    q_only = psi_q_only(q0, q1, g, t, y, prob_t, truncate_level)
    plugin = psi_plugin(q0, q1, g, t, y, prob_t, truncate_level)
    aiptw = psi_aiptw(q0, q1, g, t, y, prob_t, truncate_level)
    one_step_tmle = one_step_tmle(q0, q1, g, t, y, truncate_level)  # note different signature

    estimates = {'very_naive': very_naive, 'q_only': q_only, 'plugin': plugin, 'one_step_tmle': one_step_tmle, 'aiptw': aiptw}

    return estimates

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
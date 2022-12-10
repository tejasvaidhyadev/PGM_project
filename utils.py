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
import pandas as pd


def att_estimates(q0, q1, g, t, y, prob_t, truncate_level=0.05, deps=0.0001):

    one_step_tmle = make_one_step_tmle(prob_t, deps_default=deps)

    very_naive = psi_very_naive(t,y)
    q_only = psi_q_only(q0, q1, g, t, y, prob_t, truncate_level)
    plugin = psi_plugin(q0, q1, g, t, y, prob_t, truncate_level)
    aiptw = psi_aiptw(q0, q1, g, t, y, prob_t, truncate_level)
    one_step_tmle = one_step_tmle(q0, q1, g, t, y, truncate_level)  # note different signature

    estimates = {'very_naive': very_naive, 'q_only': q_only, 'plugin': plugin, 'one_step_tmle': one_step_tmle, 'aiptw': aiptw}

    return estimates

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

def truncate_by_g(attribute, g, level=0.1):
    keep_these = np.logical_and(g >= level, g <= 1.-level)

    return attribute[keep_these]


def truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level=0.05):
    """
    Helper function to clean up nuisance parameter estimates.
    """
    # here q_t0 is 

    orig_g = np.copy(g)

    q_t0 = truncate_by_g(np.copy(q_t0), orig_g, truncate_level)
    q_t1 = truncate_by_g(np.copy(q_t1), orig_g, truncate_level)
    g = truncate_by_g(np.copy(g), orig_g, truncate_level)
    t = truncate_by_g(np.copy(t), orig_g, truncate_level)
    y = truncate_by_g(np.copy(y), orig_g, truncate_level)

    return q_t0, q_t1, g, t, y


def psi_very_naive(t, y):
    return y[t == 1].mean() - y[t == 0].mean()


def att_estimates(q0, q1, g, t, y, prob_t, truncate_level=0.05, deps=0.0001):

    one_step_tmle = make_one_step_tmle(prob_t, deps_default=deps)

    very_naive = psi_very_naive(t,y)
    q_only = psi_q_only(q0, q1, g, t, y, prob_t, truncate_level)
    plugin = psi_plugin(q0, q1, g, t, y, prob_t, truncate_level)
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
"""Utility functions for analyzing field lines and particle trajectories in plasma physics simulations.

This module is designed to work with the Poincare and Mesh classes.

Functions:
    identifyLCFS: Identifies the Last-Closed Flux Surface (LCFS).
    boris_solver2: Implements a Boris solver for collisionless particle motion in magnetic and electric fields.
"""
import logging
from time import perf_counter
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
#plt.rcParams.update({'figure.autolayout':True})

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
""""""
def identifyLCFS(LCFStype='inner', iconds=[0], t_maxs=[100], outputHandler=logging.getLogger(), num=11):
   pass
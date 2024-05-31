import warnings
import datetime
import json
from cycler import cycler

# basic plotting and data manipulation
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# For modeling
import sklearn
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report


# Update matplotlib defaults to something nicer
def init_notebook() -> None:
    """
    Initializes the notebook with updated matplotlib defaults.
    """
    warnings.filterwarnings('ignore')
    mpl_update = {
        'font.size': 16,
        'axes.prop_cycle': cycler('color', ['#0085ca', '#888b8d', '#00c389', '#f4364c', '#e56db1']),
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.figsize': [12.0, 8.0],
        'axes.labelsize': 20,
        'axes.labelcolor': '#677385',
        'axes.titlesize': 20,
        'lines.color': '#0055A7',
        'lines.linewidth': 3,
        'text.color': '#677385',
        'font.family': 'sans-serif',
        'font.sans-serif': 'Tahoma'
    }
    mpl.rcParams.update(mpl_update)


def prepend_date(string: str) -> str:
    """
    Prepends current date to a string.

    Args:
        string: The string to prepend the date to.

    Returns:
        The string with the current date prepended.
    """
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    return f'{now}-{string}'

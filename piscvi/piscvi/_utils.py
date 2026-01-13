import importlib.resources as pkg_resources
import random
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath("."))
import numpy as np

def get_resource_path(fname):
    """Get path to pkg resources by filename.

    Returns
    -------
    pathlib.PosixPath
        Path to file.
    """
    with pkg_resources.path("resources", fname) as f:
        data_file_path = f

    return Path(data_file_path)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


import os, os.path as osp, logging, re, time, json
from collections import OrderedDict
from contextlib import contextmanager
from scipy.ndimage import gaussian_filter
import requests
import numpy as np
import uuid, sys, time, argparse, os
import warnings
np.random.seed(1001)
#from plotting_func import mt_wind

def setup_logger(name='Chem compound thermo-stability'):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info('Logger %s is already defined', name)
    else:
        fmt = logging.Formatter(
            fmt = (
                f'\033[34m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s {os.uname()[1]}]\033[0m'
                + ' %(message)s'
                ),
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
logger = setup_logger()

def setup_logging(log_path="logs/test.txt", level=logging.INFO):
    """
    Configures logging and suppresses warnings.

    Args:
        log_path (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.ERROR).
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    warnings.filterwarnings("ignore", category=FutureWarning)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized.")
    return logger


def pull_arg(*args, **kwargs):
    """
    Pulls specific arguments out of sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(*args, **kwargs)
    args, other_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + other_args
    return args

def read_arg(*args, **kwargs):
    """
    Reads specific arguments from sys.argv but does not modify sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(*args, **kwargs)
    args, _ = parser.parse_known_args()
    return args


#from contextlib import contextmanager
# decorator as command dispatcher
#@contextmanager
class Scripter:
    def __init__(self):
        self.scripts = {}

    def __call__(self, fn):
        self.scripts[fn.__name__] = fn
        return fn

    def run(self):
        script = pull_arg('script', choices=list(self.scripts.keys())).script
        logger.info('Running %s', script)
        self.scripts[script]()


@contextmanager
def time_and_log(begin_msg, end_msg='Done'):
    try:
        t1 = time.time()
        logger.info(begin_msg)
        yield None
    finally:
        t2 = time.time()
        nsecs = t2-t1
        nmins = int(nsecs//60)
        nsecs %= 60
        logger.info(end_msg + f' (took {nmins:02d}m:{nsecs:.2f}s)')

def imgcat(path):
    """
    Only useful if you're using iTerm with imgcat on the $PATH:
    Display the image in the terminal.
    """
    os.system('imgcat ' + path)


def set_matplotlib_fontsizes(smaller=14,small=18, medium=22, large=26):
    import matplotlib.pyplot as plt
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=smaller)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=smaller)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small)    # legend fontsize
    plt.rc('figure', titlesize=large)   # fontsize of the figure title

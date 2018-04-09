'''
utils.py

Functions that don't fit anywhere else.

'''
from io import BytesIO
import os
import glob
import socket
import zipfile
import sys
import itertools
import subprocess

from PIL import Image
import numpy as np


'''
BINNING
functions to help converte between floating point numbers and categories.
'''

def linear_bin(a):
    a = a + 1
    b = round(a / (2/14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    b = np.argmax(arr)
    a = b *(2/14) - 1
    return a


def bin_Y(Y):
    d = []
    for y in Y:
        arr = np.zeros(15)
        arr[linear_bin(y)] = 1
        d.append(arr)
    return np.array(d) 

def unbin_Y(Y):
    d=[]
    for y in Y:
        v = linear_unbin(y)
        d.append(v)
    return np.array(d)

def map_range(x, X_min, X_max, Y_min, Y_max):
    '''
    Linear mapping between two ranges of values
    '''
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    XY_ratio = X_range/Y_range

    y = ((x-X_min) / XY_ratio + Y_min) // 1

    return int(y)



'''
NETWORKING
'''

def my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('192.0.0.8', 1027))
    return s.getsockname()[0]




'''
OTHER
'''
def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z



def param_gen(params):
    '''
    Accepts a dictionary of parameter options and returns 
    a list of dictionary with the permutations of the parameters.
    '''
    for p in itertools.product(*params.values()):
        yield dict(zip(params.keys(), p ))


def run_shell_command(cmd, cwd=None, timeout=15):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    out = []
    err = []

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill(proc.pid)

    for line in proc.stdout.readlines():
        out.append(line.decode())

    for line in proc.stderr.readlines():
        err.append(line)
    return out, err, proc.pid

'''
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()
'''
import signal

def kill(proc_id):
    os.kill(proc_id, signal.SIGINT)




def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)




def expand_path_mask(path):
    matches = []
    path = os.path.expanduser(path)
    for file in glob.glob(path):
        if os.path.isdir(file):
            matches.append(os.path.join(os.path.abspath(file)))
    return matches


def expand_path_arg(path_str):
    path_list = path_str.split(",")
    expanded_paths = []
    for path in path_list:
        paths = expand_path_mask(path)
        expanded_paths += paths
    return expanded_paths


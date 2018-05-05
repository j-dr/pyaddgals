from __future__ import print_function, division
import numpy as np

_eval_keys = ['']


def readCfg(filename):

    with open(filename, 'r') as fp:
        cfg = yaml.load(fp)

    return cfg

def evalKeys(cfg):

    for k in cfg.keys():
        if isinstance(cfg[k], dict):
            evalKeys(cfg[k])

        if k in _eval_keys:
            cfg[k] = eval(cfg[k])

def parseConfig(filename):

    cfg = readCfg(filename)
    cfg = evalKeys(cfg)

    return cfg

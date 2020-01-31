import numpy as np

def readHaloRdel(filepath):
    """
    Read text output from calcRnn code for halos
    into numpy array
    """

    dtype = np.dtype([('id', int), ('delta', float)])
    rdel = np.genfromtxt(filepath, dtype=dtype)
    rdel = rdel[rdel['id']!=0]
    return rdel['delta']

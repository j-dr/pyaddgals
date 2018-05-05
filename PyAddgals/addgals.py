from __future__ import print_function, division
from mpi4py import MPI
import numpy as np
import pyccl as ccl
import argparse

from .config import parseConfig
from .cosmology import Cosmology
from .domain import Domain
from .nBody import NBody


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('configfile', type=np.str, help='Config file')
    args = parser.parse_args()
    config_file = args.configfile

    config = parseConfig(config_file)
    comm = MPI.COMM_WORLD

    cc = config['Cosmology']
    nb_config = config['NBody']

    cosmo = Cosmology(**cc)

    domain = Domain(**nb_config.pop(['Domain']))
    domain.decomp(comm, comm.rank, comm.ntasks)

    nbody = NBody(cosmo, domain, **nb_config)

    nbody.read()
    nbody.galaxyCatalog.paintGalaxies(config['GalaxyModel'])

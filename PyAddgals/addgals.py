from __future__ import print_function, division
from mpi4py import MPI
import numpy as np
import argparse

from .config import parseConfig
from .domain import Domain
from .nBody import NBody


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('configfile', type=np.str, help='Config file')
    args = parser.parse_args()
    config_file = args.configfile

    config      = parseConfig(config_file)
    comm        = MPI.COMM_WORLD

    cc = config['Cosmology']
    nb_config = config['NBody']

    cosmo = ccl.Cosmology(Omega_c=cc['Omega_m']-cc['Omega_b'], h=cc['h'],
                            n_s=cc['n_s'], sigma8=cc['sigma8'], w=cc['w'])


    domain = Domain(**nb_config.pop(['Domain']))
    domain.decomp(comm, comm.rank, comm.ntasks)

    nbody = NBody(basepath, cosmo, domain, **nb_config)

    nbody.read()
    nbody.galaxyCatalog.paintGalaxies(config['GalaxyModel'])

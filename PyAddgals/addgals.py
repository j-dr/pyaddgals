from __future__ import print_function, division
from mpi4py import MPI
import numpy as np
import argparse

from .config import parseConfig
from .cosmology import Cosmology
from .domain import Domain
from .nBody import NBody


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=np.str, help='Config file')
    args = parser.parse_args()
    config_file = args.config_file

    config = parseConfig(config_file)
    comm = MPI.COMM_WORLD

    cc = config['Cosmology']
    nb_config = config['NBody']

    cosmo = Cosmology(**cc)

    domain = Domain(cosmo, **nb_config.pop('Domain'))
    domain.decomp(comm, comm.rank, comm.size)

    for d in domain.yieldDomains():
        nbody = NBody(cosmo, d, **nb_config)

        nbody.read()

        nbody.galaxyCatalog.paintGalaxies(config['GalaxyModel'])
        nbody.galaxyCatalog.write()

        nbody.delete()

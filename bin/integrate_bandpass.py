from PyAddgals.config import parseConfig, readConfig
from PyAddgals.cosmology import Cosmology
from PyAddgals.domain import Domain
from PyAddgals.nBody import NBody
from PyAddgals.addgalsModel import ADDGALSModel

from mpi4py import MPI
from glob import glob
import numpy as np
import fitsio
import sys


def main(config, filter_config, filepath, magflag):

    config = parseConfig(config)
    filter_config = parseConfig(filter_config)
    files = glob(filepath)

    comm = MPI.COMM_WORLD

    cc = config['Cosmology']
    nb_config = config['NBody']

    cosmo = Cosmology(**cc)

    domain = Domain(cosmo, **nb_config.pop('Domain'))
    domain.decomp(comm, comm.rank, comm.size)
    d = domain.dummyDomain()
    nbody = NBody(cosmo, d, **nb_config)

    model = ADDGALSModel(nbody, **config['GalaxyModel']['ADDGALSModel'])
    filters = config['GalaxyModel']['colorModelConfig']['filters']
    train = fitsio.read(config['GalaxyModel']['colorModelConfig']['trainingSetFile'])

    for f in files:
        g = fitsio.read(f, columns=['SEDID', 'Z', 'MAG_R', 'MU'])
        mags = np.zeros(len(g), dtype=np.dtype(['TMAG', 'AMAG', 'LMAG', 'OMAG',
                                                'OMAGERR', 'FLUX', 'IVAR', 'Z']))
        mags['Z'] = g['Z']
        mags['TMAG'], mags['AMAG'] = model.colorModel.computeMagnitudes(g['MAG_R'],
                                                                        g['Z'],
                                                                        train['COEFFS'][g['SEDID']],
                                                                        filters)

        pixnum = f.split('.')[-2]
        fname = '{}-{}.{}.fits'.format(filepath, magflag, pixnum)
        for i in range(len(filters)):
            mags['LMAG'][:, i] = mags['TMAG'][:, i] - 2.5 * np.log10(g['MU'])

        fitsio.write(fname, mags)


if __name__ == '__main__':

    config_file = sys.argv[1]
    filter_file = sys.argv[2]
    outpath = sys.argv[3]
    magflag = sys.argv[4]

    config = readConfig(config_file, filter_file, outpath, magflag)

    main(config)

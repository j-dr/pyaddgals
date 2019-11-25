from __future__ import print_function
from glob import glob
from mpi4py import MPI
import numpy as np
import sys
# import fnmatch
import nf
import fitsio

# parameters
nfilters = 4


def run_dnf_single_file(T, Terr, TZ, filename, outfile):

    GALAXY = fitsio.read(filename, columns=['FLUX_G', 'FLUX_R', 'FLUX_I', 'FLUX_Z',
                                            'MAGERR_G', 'MAGERR_R', 'MAGERR_I', 'MAGERR_Z',
                                            'MAG_I', 'ID'])

    GALAXY = GALAXY[GALAXY['MAG_I'] > 23]

    Ngalaxies = len(GALAXY)
    print('Photo galaxies=', Ngalaxies)

    G = np.zeros((Ngalaxies, nfilters), dtype='double')
    Gerr = np.zeros((Ngalaxies, nfilters), dtype='double')

    G[:, 0] = GALAXY['FLUX_G']
    G[:, 1] = GALAXY['FLUX_R']
    G[:, 2] = GALAXY['FLUX_I']
    G[:, 3] = GALAXY['FLUX_Z']

    Gerr[:, 0] = GALAXY['FLUX_G'] * GALAXY['MAGERR_G']
    Gerr[:, 1] = GALAXY['FLUX_R'] * GALAXY['MAGERR_R']
    Gerr[:, 2] = GALAXY['FLUX_I'] * GALAXY['MAGERR_I']
    Gerr[:, 3] = GALAXY['FLUX_Z'] * GALAXY['MAGERR_Z']

    # VALID=GALAXY
    V = G
    Verr = Gerr
    ####################################

    start = 0.0
    stop = 1.5
    step = 0.3

    zbins = np.arange(start, stop, step)

    # DNF'call
    z_photo, zerr_e, Vpdf, z1, nneighbors, closestDistance = nf.dnf(
        T, TZ, V, Verr, zbins, pdf=True, Nneighbors=80, bound=False, radius=2.0, magflux='flux')

    print("mean Nneighbors=", np.mean(nneighbors))

    out = np.zeros(Ngalaxies, dtype=np.dtype([('ID', np.int64), ('Z_MEAN', np.float), ('Z_MC', np.float), ('Z_SIGMA', np.float)]))
    out['ID'] = GALAXY['ID']
    out['Z_MEAN'] = z_photo
    out['Z_MC'] = z1
    out['Z_SIGMA'] = zerr_e

    fitsio.write(outfile)


if __name__ == '__main__':
    GALAXY = fitsio.read(sys.argv[1])
    fileglob = glob(sys.argv[2])
    if len(sys.argv) > 3:
        merge = True
    else:
        merge = False

    Ngalaxies = len(GALAXY)
    print('Train galaxies=', Ngalaxies)

    # GALAXY=np.random.permutation(GALAXY)
    G = np.zeros((Ngalaxies, nfilters), dtype='double')
    Gerr = np.zeros((Ngalaxies, nfilters), dtype='double')

    G[:, 0] = GALAXY['FLUX_G']
    G[:, 1] = GALAXY['FLUX_R']
    G[:, 2] = GALAXY['FLUX_I']
    G[:, 3] = GALAXY['FLUX_Z']

    Gerr[:, 0] = GALAXY['FLUX_G'] * GALAXY['MAGERR_G']
    Gerr[:, 1] = GALAXY['FLUX_R'] * GALAXY['MAGERR_R']
    Gerr[:, 2] = GALAXY['FLUX_I'] * GALAXY['MAGERR_I']
    Gerr[:, 3] = GALAXY['FLUX_Z'] * GALAXY['MAGERR_Z']

    Ntrain = Ngalaxies
    # TRAIN=GALAXY
    T = G
    Terr = Gerr
    TZ = GALAXY['Z']

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if not merge:
        for filename in fileglob[rank::size]:
            outfile = filename.replace('fits', 'dnf.fits')
            run_dnf_single_file(T, Terr, TZ, filename, outfile)
    else:
        for i, filename in enumerate(fileglob):
            if i == 0:
                outfile = filename.split('.')
                outfile[-1] = 'combined.dnf'
                outfile = '.'.join(outfile)

            infile = filename.replace('fits', 'dnf.fits')
            dnf_data = fitsio.read(infile)

            f = fitsio.FITS(outfile, 'rw')
            if len(f) > 1:
                f[-1].append(dnf_data)
            else:
                f.write(dnf_data)

    sys.exit()

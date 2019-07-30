from __future__ import print_function, division
from halotools.empirical_models import abunmatch
from copy import copy
from glob import glob
from mpi4py import MPI
from time import time
import numpy as np
import healpy as hp
import fitsio
import sys

from PyAddgals.config import parseConfig
from PyAddgals.cosmology import Cosmology
from PyAddgals.domain import Domain
from PyAddgals.nBody import NBody
from PyAddgals.addgalsModel import ADDGALSModel
from fast3tree import fast3tree


def load_model(cfg):

    config = parseConfig(cfg)

    cc = config['Cosmology']
    nb_config = config['NBody']

    cosmo = Cosmology(**cc)

    domain = Domain(cosmo, **nb_config.pop('Domain'))
    domain.decomp(None, 0, 1)

    for d in domain.yieldDomains():
        nbody = NBody(cosmo, d, **nb_config)
        break

    model = ADDGALSModel(nbody, **config['GalaxyModel']['ADDGALSModel'])
    filters = ['desy3/desy3std_g.par', 'desy3/desy3std_r.par', 'desy3/desy3std_i.par', 'desy3/desy3std_z.par', 'desy3/desy3_Y.par']

    return model, filters


def compute_distances(px, py, pz, hpx, hpy, hpz, hmass, mcut):
    idx = (hmass > 10**mcut)
    cpos = np.zeros((np.sum(idx), 3))

    pos = np.zeros((len(px), 3))
    pos[:, 0] = px
    pos[:, 1] = py
    pos[:, 2] = pz

    cpos[:, 0] = hpx[idx]
    cpos[:, 1] = hpy[idx]
    cpos[:, 2] = hpz[idx]

    rhalo = np.zeros(len(pos))

    with fast3tree(cpos) as tree:
        for i in range(len(pos)):
            d = tree.query_nearest_distance(pos[i, :])
            rhalo[i] = d

    return rhalo


def treefree_cam_rhaloscat(luminosity, x, y, z, hpx, hpy, hpz, mass, masslim,
                           cc, luminosity_train, gr_train, rhalo=None, rs=None, rank=None):

    if rhalo is None:
        start = time()
        rhalo = compute_distances(x, y, z, hpx, hpy, hpz,
                                  mass, masslim)
        end = time()
        print('[{}]: Finished computing rhalo. Took {}s'.format(rank, end - start))
        sys.stdout.flush()

    logrhalo = np.log10(rhalo)

    if rs is not None:
        logrhalo += (cc + rs * rhalo) * np.random.randn(len(rhalo))
    else:
        logrhalo += cc * np.random.randn(len(rhalo))

    start = time()
    idx_swap = abunmatch.conditional_abunmatch(
        luminosity, -logrhalo, luminosity_train, gr_train, 99, return_indexes=True)
    end = time()

    print('[{}]: Finished abundance matching SEDs. Took {}s.'.format(rank, end - start))
    sys.stdout.flush()
    return idx_swap, rhalo


def reassign_colors_cam(gals, halos, cfg, mhalo=12.466, scatter=0.749, rank=None):

    model, filters = load_model(cfg)

    gr = gals['AMAG'][:, 0] - gals['AMAG'][:, 1]

    idx_swap, rhalo = treefree_cam_rhaloscat(gals['MAG_R_EVOL'], gals['PX'], gals['PY'],
                                             gals['PZ'], halos['PX'],
                                             halos['PY'], halos['PZ'], halos['M200B'],
                                             mhalo, scatter, gals['MAG_R_EVOL'], gr,
                                             rank=rank)

    temp_sedid = gals['SEDID'][idx_swap]
    coeffs = model.colorModel.trainingSet[temp_sedid]['COEFFS']

    # make sure we don't have any negative redshifts
    z_a = copy(gals['Z'])
    z_a[z_a < 1e-6] = 1e-6
    mag = gals['MAG_R_EVOL']

    start = time()
    omag, amag = model.colorModel.computeMagnitudes(mag, z_a, coeffs, filters)
    end = time()

    print('[{}]: Done computing magnitudes. Took {}s'.format(rank, end - start))
    sys.stdout.flush()
#    g['SEDID'] = temp_sedid
#    g['AMAG'] = amag
#    g['TMAG'] = omag

    return omag, amag, temp_sedid


if __name__ == '__main__':

    filepath = sys.argv[1]
    hfilepath = sys.argv[2]
    cfg = sys.argv[3]
    mhalo = float(sys.argv[4])
    scatter = float(sys.argv[5])

    if len(sys.argv) > 5:
        lensmags = bool(sys.argv[6])

    files = glob(filepath)
    halofiles = glob(hfilepath.format('*'))
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    ud_map = hp.ud_grade(np.arange(12 * 2**2), 8, order_in='NESTED', order_out='NESTED')

    files = files[rank::size]

    for i, f in enumerate(halofiles):
        hi = fitsio.read(f, columns=['PX', 'PY', 'PZ', 'HOST_HALOID', 'M200B', 'Z_COS'])
        hi = hi[hi['HOST_HALOID'] == -1]

        if i == 0:
            h = hi
        else:
            h = np.hstack([h, hi])

    h = h[h['HOST_HALOID'] == -1]
    del hi

    hr = np.sqrt(h['PX']**2 + h['PY']**2 + h['PZ']**2)

    for i in range(len(files)):
        print(files[i])
        g = fitsio.read(files[i])
        r = np.sqrt(g['PX']**2 + g['PY']**2 + g['PZ']**2)

        config = parseConfig(cfg)
        cc = config['Cosmology']
        nb_config = config['NBody']
        nb_config['Domain']['pixlist'] = [0]

        cosmo = Cosmology(**cc)

        domain = Domain(cosmo, **nb_config.pop('Domain'))
        domain.decomp(None, 0, 1)

        for d in domain.yieldDomains():
            nbody = NBody(cosmo, d, **nb_config)
            print('[{}]: working on rbin {}'.format(rank, domain.rbins[d.boxnum][d.rbin]))
            sys.stdout.flush()
            idx = ((domain.rbins[d.boxnum][d.rbin] <= r) &
                   (r < domain.rbins[d.boxnum][d.rbin + 1]))

            hidx = (((domain.rbins[d.boxnum][d.rbin] - 100) <= hr) &
                    (hr < (domain.rbins[d.boxnum][d.rbin + 1] + 100.)))

#            gi = reassign_colors_cam(g[idx], h[hidx], cfg, mhalo=mhalo, scatter=scatter)
            omag, amag, sedid = reassign_colors_cam(g[idx], h[hidx], cfg,
                                                    mhalo=mhalo, scatter=scatter,
                                                    rank=rank)

            g['TMAG'][idx] = omag
            g['AMAG'][idx] = amag
            g['SEDID'][idx] = sedid

        if lensmags:
            for im in range(g['TMAG'].shape[1]):
                g['LMAG'][:, im] = g['TMAG'][:, im] - 2.5 * np.log10(g['MU'])

            ofile = files[i].replace('lensed', 'lensed_cam')
        else:
            fs = files[i].split('.')
            fs[0] = fs[0] + '_cam'
            ofile = '.'.join(fs)

        print('[{}]: Writing to {}'.format(rank, ofile))

        fitsio.write(ofile, g)

        del g

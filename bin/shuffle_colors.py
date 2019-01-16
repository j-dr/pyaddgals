from glob import glob
from mpi4py import MPI
import numpy as np
import healpy as hp
import fitsio
import sys
import os

from PyAddgals.config import parseConfig
from PyAddgals.cosmology import Cosmology
from PyAddgals.domain import Domain
from PyAddgals.nBody import NBody
from PyAddgals.addgalsModel import ADDGALSModel
from fast3tree import fast3tree


def rankRhalo(z, magnitude, rhalo, zwindow, magwindow):

    drhalo = np.max(rhalo) - np.min(rhalo)
    rankrhalo = np.zeros(len(z))

    pos = np.zeros((len(z), 3))
    pos[:, 0] = z
    pos[:, 1] = magnitude
    pos[:, 2] = rhalo

    max_distances = np.array([zwindow, magwindow, drhalo])
    neg_max_distances = -1.0 * max_distances

    tree_dsidx = np.random.choice(np.arange(len(z)), size=len(z) // 10)
    tree_pos = pos[tree_dsidx]

    with fast3tree(tree_pos) as tree:

        for i, p in enumerate(pos):

            tpos = tree.query_box(
                p + neg_max_distances, p + max_distances, output='pos')
            tpos = tpos[:, 2]
            tpos.sort()
            rankrhalo[i] = tpos.searchsorted(pos[i, 2]) / (len(tpos) + 1)

    return rankrhalo


def determineSwap(mag, rankrhalo, isred, mm, mr, magmin=-22, magmax=-19):

    bmag = np.zeros(len(mag))
    bmag[:] = mag
    bmag[bmag < magmin] = magmin
    bmag[bmag > magmax] = magmax

    def pofr(m, rankr):
        return -mr * (mm * (m + 22)) * (rankr - 0.5)

    pswap = pofr(bmag, rankrhalo)
    swap = np.random.rand(len(pswap)) < np.abs(pswap)
    swap[((swap) & (isred) & (pswap > 0)) | ((pswap < 0))] = False

    return swap, pswap


def shuffleColors(mag, z, swap, rankrhalo, ir, dm=0.1, dz=0.05, dr=0.05):

    pos = np.zeros((len(mag), 3))
    idx_swap = np.zeros(len(mag[swap]), dtype=np.int)
    swapped = np.zeros(len(z), dtype=np.bool)

    pos[:, 0] = mag
    pos[:, 1] = z
    pos[:, 2] = np.abs(rankrhalo - 0.5)

    max_distances = np.array([dm, dz, dr])
    neg_max_distances = np.array([-dm, -dz, -dr])

    with fast3tree(pos) as tree:

        for i, p in enumerate(pos[swap]):

            idx, tpos = tree.query_box(
                p + neg_max_distances, p + max_distances, output='both')
            isred_samp = ir[idx]
            swap_samp = swapped[idx]
            rankrhalo_samp = rankrhalo[idx]
            idx = idx[(isred_samp) & (~swap_samp) & (rankrhalo_samp > 0.5)]
            if len(idx) > 0:
                swapid = np.random.choice(idx)
                idx_swap[i] = swapid
                swapped[swapid] = True
            else:
                idx_swap[i] = -1

        print('Number of bad swaps: {}'.format(np.sum(idx_swap == -1)))
        max_distances *= 5
        neg_max_distances *= 5
        isbad, = np.where(idx_swap == -1)

        for i, p in enumerate(pos[swap][idx_swap == -1]):
            idx, tpos = tree.query_box(
                p + neg_max_distances, p + max_distances, output='both')
            isred_samp = ir[idx]
            swap_samp = swapped[idx]
            rankrhalo_samp = rankrhalo[idx]
            idx = idx[(isred_samp) & (~swap_samp) & (rankrhalo_samp > 0.5)]
            if len(idx) > 0:
                swapid = np.random.choice(idx)
                idx_swap[i] = swapid
                swapped[swapid] = True
            else:
                idx_swap[isbad[i]] = -1
        idx_swap[idx_swap == -1] = np.arange(len(idx_swap))[idx_swap == -1]

    return idx_swap


def load_model(cfg):

    config = parseConfig(cfg)

    comm = MPI.COMM_WORLD

    cc = config['Cosmology']
    nb_config = config['NBody']

    cosmo = Cosmology(**cc)

    domain = Domain(cosmo, **nb_config.pop('Domain'))
    domain.decomp(comm, comm.rank, comm.size)

    for d in domain.yieldDomains():
        nbody = NBody(cosmo, d, **nb_config)
        break

    model = ADDGALSModel(nbody, **config['GalaxyModel']['ADDGALSModel'])
    filters = ['desy3/desy3std_g.par', 'desy3/desy3std_r.par', 'desy3/desy3std_i.par', 'desy3/desy3std_z.par', 'desy3/desy3_Y.par']

    return model, filters


def reassign_colors(g, h, mr, mm, cfg, mhalo=6e12):

    model, filters = load_model(cfg)

    centrals = h[(h['HOST_HALOID'] == h['HALOID']) & (h['M200B'] > mhalo)]
    cpos = np.zeros((len(centrals), 3))

    pos = np.zeros((len(g), 3))
    pos[:, 0] = g['PX']
    pos[:, 1] = g['PY']
    pos[:, 2] = g['PZ']

    cpos[:, 0] = centrals['PX']
    cpos[:, 1] = centrals['PY']
    cpos[:, 2] = centrals['PZ']

    rhalo14 = np.zeros(len(pos))

    with fast3tree(cpos) as tree:
        for i in range(len(pos)):
            d = tree.query_nearest_distance(pos[i, :])
            rhalo14[i] = d

    gr = g['AMAG'][:, 0] - g['AMAG'][:, 1]
    isred = gr > (-0.22 - 0.05 * g['AMAG'][:, 1])

    rankrhalo = rankRhalo(g['Z_COS'], g['MAG_R_EVOL'], np.log10(rhalo14), 0.1, 0.3)
    swap, pswap = determineSwap(g['MAG_R_EVOL'], rankrhalo, isred, 1, 1.0)
    idx_swap = shuffleColors(g['MAG_R_EVOL'], g['Z_COS'], swap, rankrhalo, isred)

    from copy import copy
    sedid_swap = g['SEDID'][swap]
    sedid_swap1 = g['SEDID'][idx_swap]
    temp_sedid = copy(g['SEDID'])
    temp_sedid[swap] = sedid_swap1
    temp_sedid[idx_swap] = sedid_swap

    coeffs = model.colorModel.trainingSet[temp_sedid]['COEFFS']

    # make sure we don't have any negative redshifts
    z_a = copy(g['Z'])
    z_a[z_a < 1e-6] = 1e-6
    mag = g['MAG_R_EVOL']

    omag, amag = model.colorModel.computeMagnitudes(mag, z_a, coeffs, filters)

    g['SEDID'] = temp_sedid
    g['AMAG'] = amag
    g['TMAG'] = omag

    return g


if __name__ == '__main__':

    filepath = sys.argv[1]
    hfilepath = sys.argv[2]
    cfg = sys.argv[3]
    mr = sys.argv[4]
    mm = sys.argv[5]

    files = glob(filepath)
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    ud_map = hp.ud_grade(np.arange(12 * 2**2), 8, order_in='NESTED', order_out='NESTED')

    files = files[rank::size]

    for i in range(len(files)):
        g = fitsio.read(files[i])
        r = np.sqrt(g['PX']**2 + g['PY']**2 + g['PZ']**2)

        pix8 = int(files[i].split('.')[-2])
        pix = ud_map[pix8]

        h = fitsio.read(hfilepath.format(pix), columns=['PX', 'PY', 'PZ', 'HOST_HALOID', 'HALOID', 'M200B'])
        hr = np.sqrt(h['PX']**2 + h['PY']**2 + h['PZ']**2)
        config = parseConfig(cfg)

        cc = config['Cosmology']
        nb_config = config['NBody']

        cosmo = Cosmology(**cc)

        domain = Domain(cosmo, **nb_config.pop('Domain'))
        domain.decomp(None, 1, 1)

        for d in domain.yieldDomains():
            nbody = NBody(cosmo, d, **nb_config)
            idx = ((domain.rbins[d.boxnum][d.rbin] <= r) &
                   (r < domain.rbins[d.boxnum][d.rbin + 1]))

            hidx = (((domain.rbins[d.boxnum][d.rbin]-100) <= hr) &
                   (hr < (domain.rbins[d.boxnum][d.rbin + 1]+100)))

            gi = reassign_colors(g[idx], h[hidx], mr, mm, cfg, mhalo=6e12)
            ofile = files[i].replace('fits', 'swapped.fits')

            if os.path.exists(ofile):
                with fitsio.FITS(ofile, 'rw') as f:
                    f[-1].append(gi)
            else:
                fitsio.write(ofile, gi)

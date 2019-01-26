from __future__ import print_function, division
from halotools.empirical_models import abunmatch
from copy import copy
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


def reassign_colors_cam(g, h, cfg, mhalo=12.466, corr=0.749, alpham=0.0689):

    model, filters = load_model(cfg)

    centrals = h[(h['HOST_HALOID'] == -1) & (h['M200B'] > 10**mhalo)]
    cpos = np.zeros((len(centrals), 3))

    pos = np.zeros((len(g), 3))
    pos[:, 0] = g['PX']
    pos[:, 1] = g['PY']
    pos[:, 2] = g['PZ']

    cpos[:, 0] = centrals['PX']
    cpos[:, 1] = centrals['PY']
    cpos[:, 2] = centrals['PZ']

    rhalo = np.zeros(len(pos))

    with fast3tree(cpos) as tree:
        for i in range(len(pos)):
            d = tree.query_nearest_distance(pos[i, :])
            rhalo[i] = d

    mr = copy(g['MAG_R_EVOL'])
    mr[mr<-22] = -22
    mr[mr>-18] = -18
    gr = g['AMAG'][:, 0] - g['AMAG'][:, 1]

    idx = np.argsort(rhalo)
    rhalo_sorted = rhalo[idx]
    rank_rhalo = np.arange(len(rhalo))/len(rhalo)
    corr_coeff = corr * (mr + 22) ** (alpham)
    corr_coeff[corr_coeff > 1] = 1.
    noisy_rank_rhalo = abunmatch.noisy_percentile(rank_rhalo, corr_coeff)

    g = g[idx]
    gr = g['AMAG'][:,0] - g['AMAG'][:,1]

    idx_swap = abunmatch.conditional_abunmatch(g['MAG_R_EVOL'], noisy_rank_rhalo, g['MAG_R_EVOL'], -gr, 99, return_indexes=True)
    temp_sedid = g['SEDID'][idx_swap]

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
    mhalo = float(sys.argv[4])
    corr = float(sys.argv[5])
    alpham = float(sys.argv[6])

    files = glob(filepath)
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    ud_map = hp.ud_grade(np.arange(12 * 2**2), 8, order_in='NESTED', order_out='NESTED')

    files = files[rank::size]

    rbins = np.linspace(0, 4000, 11)

    for i in range(len(files)):
        print(files[i])
        g = fitsio.read(files[i])
        r = np.sqrt(g['PX']**2 + g['PY']**2 + g['PZ']**2)

        pix8 = int(files[i].split('.')[-2])
        pix = ud_map[pix8]
        h = fitsio.read(hfilepath.format(pix), columns=['PX', 'PY', 'PZ', 'HOST_HALOID', 'HALOID', 'M200B'])
        hr = np.sqrt(h['PX']**2 + h['PY']**2 + h['PZ']**2)
        hpix = hp.vec2pix(8, h['PX'], h['PY'], h['PZ'], nest=True)
        print('halo pix in gal pix: {}'.format(np.in1d(pix8, hpix)))

        config = parseConfig(cfg)
        cc = config['Cosmology']
        nb_config = config['NBody']

        cosmo = Cosmology(**cc)

        domain = Domain(cosmo, **nb_config.pop('Domain'))
        domain.decomp(None, 1, 1)

#        for j in range(len(rbins)-1):
#            idx = (rbins[j] < r) & (r < rbins[j+1])
#            hidx  = ((rbins[j] - 50) < hr) & (hr < (rbins[j+1] + 50))
#            gi = reassign_colors_cam(g[idx], h[hidx], cfg, mhalo=mhalo, corr=corr, alpham=alpham)

#            ofile = files[i].replace('fits', 'cam_allz.fits')
#            print(ofile)
#            if os.path.exists(ofile):
#                with fitsio.FITS(ofile, 'rw') as f:
#                    f[-1].append(gi)
#            else:
#                fitsio.write(ofile, gi)

        for d in domain.yieldDomains():
            nbody = NBody(cosmo, d, **nb_config)
            idx = ((domain.rbins[d.boxnum][d.rbin] <= r) &
                   (r < domain.rbins[d.boxnum][d.rbin + 1]))
            hidx = (((domain.rbins[d.boxnum][d.rbin]-100) <= hr) &
                   (hr < (domain.rbins[d.boxnum][d.rbin + 1]+100.)))

            gi = reassign_colors_cam(g[idx], h[hidx], cfg, mhalo=mhalo, corr=corr, alpham=alpham)
            ofile = files[i].replace('fits', 'cam.fits')

            if os.path.exists(ofile):
                with fitsio.FITS(ofile, 'rw') as f:
                    f[-1].append(gi)
            else:
                fitsio.write(ofile, gi)

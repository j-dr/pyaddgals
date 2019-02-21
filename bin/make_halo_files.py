from halotools.sim_manager import TabularAsciiReader
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


def load_model(cfg):

    config_file = cfg
    config = parseConfig(config_file)

    cc = config['Cosmology']
    nb_config = config['NBody']

    cosmo = Cosmology(**cc)
    d_config = nb_config.pop('Domain')

    domain = Domain(cosmo, **d_config)
    domain.decomp(None, 1, 1)

    for d in domain.yieldDomains():
        nbody = NBody(cosmo, d, **nb_config)
        break

    model = ADDGALSModel(nbody, **config['GalaxyModel']['ADDGALSModel'])

    return model, config, d_config


def write_halo_file(cfg, outpath):

    model, config, d_config = load_model(cfg)
    parentfiles = config['NBody']['halofile']
    fname = parentfiles[0].split('/')[-1]
    outfiles = [f.replace(fname, 'out_0.list') for f in parentfiles]

    cdict_parent = model.nbody.haloCatalog.getColumnDict('BCCLightcone')
    cdict_out = model.nbody.haloCatalog.getColumnDict('OutLightcone')

    for i in range(len(parentfiles)):

        reader = TabularAsciiReader(parentfiles[i], cdict_parent)
        parents = reader.read_ascii()

        reader = TabularAsciiReader(outfiles[i], cdict_out)
        out = reader.read_ascii()

        r = np.sqrt(out['PX']**2 + out['PY']**2 + out['PZ']**2)

        idx = np.in1d(out['HALOID'], parents['id']) & ((d_config['rmin'][i] <= r) &
                                                       (r < d_config['rmax'][i]))
        matched_out = out[idx]
        r = r[idx]

        vec = np.zeros((len(matched_out), 3))
        vec[:, 0] = matched_out['PX']
        vec[:, 1] = matched_out['PY']
        vec[:, 2] = matched_out['PZ']

        vel = np.zeros((len(matched_out), 3))
        vel[:, 0] = matched_out['VX']
        vel[:, 1] = matched_out['VY']
        vel[:, 2] = matched_out['VZ']

        matched_out['Z_COS'] = model.nbody.cosmo.zofR(r)
        matched_out['Z'] = matched_out['Z_COS'] + np.sum(vec * vel, axis=1) / \
            np.sqrt(np.sum(vec**2, axis=1)) / 299792.458

        del vel

        idx = parents['id'].argsort()
        parents = parents[idx]
        idx = parents['id'].searchsorted(matched_out['HALOID'])

        matched_out['HOST_HALOID'] = parents['pid'][idx]
        matched_out['TRA'], matched_out['TDEC'] = hp.vec2ang(vec, lonlat=True)
        matched_out['RA'] = 0.
        matched_out['DEC'] = 0.

        pix = hp.vec2pix(2, vec[:, 0], vec[:, 1], vec[:, 2], nest=True)
        upix = np.unique(pix)

        for p in upix:
            idx = pix == p
            opath = outpath.format(p)

            if os.path.exists(opath):
                with fitsio.FITS(opath, 'rw') as f:
                    f[-1].append(matched_out[idx])
            else:
                fitsio.write(opath, matched_out[idx])


if __name__ == '__main__':

    cfg = sys.argv[1]
    outpath = sys.argv[2]

    write_halo_file(cfg, outpath)

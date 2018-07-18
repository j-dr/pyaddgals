from __future__ import print_function, division
import numpy as np
import healpy as hp
import fitsio
import os
import sys

from .addgalsModel import ADDGALSModel

_available_models = ['ADDGALSModel']


class GalaxyCatalog(object):
    """
    Galaxy catalog class

    """

    def __init__(self, nbody):

        self.nbody = nbody
        self.catalog = {}

    def paintGalaxies(self, config):
        """Apply a galaxy model to the nbody sim

        Parameters
        ----------
        config : dict
            Galaxy model config file, must contain algorithm and
            relevant input information, e.g. LF, f_red(L,z), etc.

        Returns
        -------
        None
        """

        model_class = list(config.keys())[0]

        if not (model_class in _available_models):
            raise(ValueError("Model {} is not implemented".format(model_class)))

        if model_class == 'ADDGALSModel':
            model = ADDGALSModel(self.nbody, **config['ADDGALSModel'])

        print('Painting galaxies to domain with z_min, z_max, pix, nside: {}, {}, {}, {}'.format(self.nbody.domain.zmin,
                                                                                                 self.nbody.domain.zmax,
                                                                                                 self.nbody.domain.pix,
                                                                                                 self.nbody.domain.nside))

        model.paintGalaxies()

    def write(self, filename):
        """Write galaxy catalog to disk.

        Returns
        -------
        None
        """

        domain = self.nbody.domain
        if os.path.exists(filename):
            f = fitsio.FITS(filename)
            ngal = f[-1].read_header()['NAXIS2']
            f.close()

        self.catalog['ID'] = (domain.pix * 1e9 + np.arange(len(self.catalog['PX'])) + ngal).astype(np.int64)

        cdtype = np.dtype(list(zip(self.catalog.keys(),
                                   [(self.catalog[k].dtype.type,
                                    self.catalog[k].shape[1])
                                    if len(self.catalog[k].shape) > 1
                                    else self.catalog[k].dtype.type
                                    for k in self.catalog.keys()])))

        out = np.zeros(len(self.catalog[list(self.catalog.keys())[0]]),
                       dtype=cdtype)
        for k in self.catalog.keys():
            out[k] = self.catalog[k]


        r = np.sqrt(out['PX']**2 + out['PY']**2 + out['PZ']**2)
        pix = hp.vec2pix(domain.nside, out['PX'], out['PY'], out['PZ'],
                         nest=domain.nest)
        
        boxnum = domain.boxnum
        # cut off buffer region, make sure we only have the pixel we want
        print('Cutting catalog to {} <= z < {}'.format(self.nbody.cosmo.zofR(domain.rbins[boxnum][domain.rbin]),
                                                       self.nbody.cosmo.zofR(domain.rbins[boxnum][domain.rbin + 1])))
        
        sys.stdout.flush()
        idx = ((domain.rbins[boxnum][domain.rbin] <= r) &
               (r < domain.rbins[boxnum][domain.rbin + 1]) &
               (domain.pix == pix))

        if os.path.exists(filename):
            with fitsio.FITS(filename, 'rw') as f:
                f[-1].append(out[idx])
        else:
            fitsio.write(filename, out[idx])

    def delete(self):
        """Delete galaxy catalog

        Returns
        -------
        None

        """

        keys = list(self.catalog.keys())

        for k in keys:
            del self.catalog[k]

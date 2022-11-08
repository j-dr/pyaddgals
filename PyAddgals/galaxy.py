from __future__ import print_function, division
from numpy.lib.recfunctions import repack_fields
import numpy as np
import healpy as hp
import fitsio
import os
import sys

from .addgalsModel import ADDGALSModel
from .CLFModel import CLFModel

_available_models = ['ADDGALSModel', 'CLFModel']


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
            self.model = ADDGALSModel(self.nbody, **config['ADDGALSModel'])
            
        if model_class == 'CLFModel':
            self.model = CLFModel(self.nbody, **config['CLFModel'])

        print('Painting galaxies to domain with z_min, z_max, pix, nside: {}, {}, {}, {}'.format(self.nbody.domain.zmin,
                                                                                                 self.nbody.domain.zmax,
                                                                                                 self.nbody.domain.pix,
                                                                                                 self.nbody.domain.nside))

        self.model.paintGalaxies()

    def write(self, filename, nside_output, write_pos=False):
        """Write galaxy catalog to disk.

        Returns
        -------
        None
        """

        domain = self.nbody.domain

        if 'ID' not in list(self.catalog.keys()):
            self.catalog['ID'] = np.zeros(len(self.catalog['PX']))

        if 'TRA' not in list(self.catalog.keys()):
            self.catalog['TRA'], self.catalog['TDEC'] = hp.vec2ang(np.vstack([self.catalog['PX'],
                                                                   self.catalog['PY'],
                                                                   self.catalog['PZ']]).T,
                                                                   lonlat=True)

        self.catalog['EPSILON'] = np.zeros((len(self.catalog['PX']), 2))
        self.catalog['SIZE'] = np.zeros(len(self.catalog['PX']))
        self.catalog['KAPPA'] = np.zeros(len(self.catalog['PX']))
        self.catalog['MU'] = np.zeros(len(self.catalog['PX']))
        self.catalog['W'] = np.zeros(len(self.catalog['PX']))

        self.catalog['GAMMA1'] = np.zeros(len(self.catalog['PX']))
        self.catalog['GAMMA2'] = np.zeros(len(self.catalog['PX']))

        self.catalog['DEC'] = np.zeros(len(self.catalog['PX']))
        self.catalog['RA'] = np.zeros(len(self.catalog['PX']))

        self.catalog['LMAG'] = np.zeros_like(self.catalog['TMAG'])
        self.catalog['OMAG'] = np.zeros_like(self.catalog['TMAG'])
        self.catalog['OMAGERR'] = np.zeros_like(self.catalog['TMAG'])
        self.catalog['FLUX'] = np.zeros_like(self.catalog['TMAG'])
        self.catalog['IVAR'] = np.zeros_like(self.catalog['TMAG'])


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

        out = out[idx]
        del idx

        keys = list(self.catalog.keys())

        if len(keys) == 0:
            return

        for k in keys:
            del self.catalog[k]

        del self.catalog

        if nside_output != domain.nside:
            map_in = np.arange(12 * domain.nside**2)

            if domain.nest:
                order = 'NESTED'
            else:
                order = 'RING'

            map_out = hp.ud_grade(map_in, nside_output, order_in=order,
                                  order_out=order)
            pix, = np.where(map_out == domain.pix)

        else:
            pix = [domain.pix]

        for p in pix:
            fname = '{}.{}.fits'.format(filename, p)
            print('Writing to {}'.format(fname))

            if write_pos:
                pfname = '{}.{}.lens.fits'.format(filename, p)

            if os.path.exists(fname):
                f = fitsio.FITS(fname)
                ngal = f[-1].read_header()['NAXIS2']
                f.close()
            else:
                ngal = 0

            pix = hp.vec2pix(nside_output, out['PX'], out['PY'], out['PZ'],
                             nest=domain.nest)

            idx = pix == p
            if np.sum(idx) < 100:
                continue

            out['ID'][idx] = (p * 1e9 + np.arange(len(out['PX'][idx])) + ngal).astype(np.int64)

            if os.path.exists(fname):
                with fitsio.FITS(fname, 'rw') as f:
                    f[-1].append(out[idx])
            else:
                fitsio.write(fname, out[idx])

            if write_pos:
                if os.path.exists(pfname):
                    with fitsio.FITS(pfname, 'rw') as f:
                        f[-1].append(repack_fields(out[['ID', 'PX', 'PY', 'PZ']][idx]))
                else:
                    fitsio.write(pfname, repack_fields(out[['ID', 'PX', 'PY', 'PZ']][idx]))

        del out


    def writeSnapshot(self, filename):
        """Write galaxy catalog to disk.

        Returns
        -------
        None
        """

        domain = self.nbody.domain

        if 'ID' not in list(self.catalog.keys()):
            self.catalog['ID'] = np.zeros(len(self.catalog['PX']))

        if not self.model.colorModel.no_colors:
            self.catalog['LMAG'] = np.zeros_like(self.catalog['TMAG'])
            self.catalog['OMAG'] = np.zeros_like(self.catalog['TMAG'])
            self.catalog['OMAGERR'] = np.zeros_like(self.catalog['TMAG'])
            self.catalog['FLUX'] = np.zeros_like(self.catalog['TMAG'])
            self.catalog['IVAR'] = np.zeros_like(self.catalog['TMAG'])


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

        snapnum = domain.snapnum

        keys = list(self.catalog.keys())

        if len(keys) == 0:
            return

        for k in keys:
            del self.catalog[k]

        del self.catalog

        fname = '{}.{}.fits'.format(filename, snapnum)
        print('Writing to {}'.format(fname))

        if os.path.exists(fname):
            f = fitsio.FITS(fname)
            ngal = f[-1].read_header()['NAXIS2']
            f.close()
        else:
            ngal = 0

        out['ID'] = (np.arange(len(out['PX'])) + ngal).astype(np.int64)

        if os.path.exists(fname):
            with fitsio.FITS(fname, 'rw') as f:
                f[-1].append(out)
        else:
            fitsio.write(fname, out)

        del out

    def delete(self):
        """Delete galaxy catalog

        Returns
        -------
        None

        """

        if not hasattr(self, 'catalog'):
            return

        keys = list(self.catalog.keys())

        if len(keys) == 0:
            return

        for k in keys:
            del self.catalog[k]

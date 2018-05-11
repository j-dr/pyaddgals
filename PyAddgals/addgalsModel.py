from __future__ import print_function, division
from scipy.special import erf
import numpy as np

from .galaxyModel import GalaxyModel
from . import luminosityFunction


class ADDGALSModel(GalaxyModel):

    def __init__(self, nbody, luminosityFunctionConfig=None,
                 rdelModelConfig=None,
                 redFractionModelConfig=None, colorModelConfig=None):

        self.nbody = nbody

        if luminosityFunctionConfig is None:
            raise(ValueError('ADDGALS model must define luminosityFunctionConfig'))

        if rdelModelConfig is None:
            raise(ValueError('ADDGALS model must define rdelModelConfig'))

        if redFractionModelConfig is None:
            raise(ValueError('ADDGALS model must define redFractionModelConfig'))

        if colorModelConfig is None:
            raise(ValueError('ADDGALS model must define colorModelConfig'))

        lf_type = luminosityFunctionConfig.pop('modeltype')

        self.luminosityFunction = getattr(luminosityFunction, lf_type)
        self.luminosityFunction = self.luminosityFunction(
            nbody.cosmo, **luminosityFunctionConfig)

        self.rdelModel = RdelModel(self.luminosityFunction, **rdelModelConfig)
        self.redFractionModel = RedFractionModel(**redFractionModelConfig)
        self.colorModel = ColorModel(**colorModelConfig)

    def paintGalaxies(self):
        """Paint galaxies into nbody using ADDGALS

        Parameters
        ----------
        nbody : NBody
            The nbody object to paint galaxies into.

        Returns
        -------
        None
        """

        domain = self.nbody.domain

        n_gal = self.luminosityFunction.integrate(domain.zmin,
                                                  domain.zmax,
                                                  domain.getArea())

        z = self.drawRedshifts(n_gal)
        mag = self.luminosityFunction.sampleLuminosities(domain, z)
        dens = self.rdelModel.sampleDensities(z, mag)

        self.catalog = None


class RdelModel(object):

    def __init__(self, lf, modelfile=None, **kwargs):

        self.luminosityFunction = lf
        self.modelfile = modelfile

        if self.modelfile is None:

            for k in kwargs.keys():
                setattr(self, k, kwargs[k])

    def loadModelFile(self):
        """Load a model from file into self.model

        Returns
        -------
        None

        """

        assert(self.modelfile is not None)

        params = {'pmag': [], 'pz': [], 'mucmag': [], 'mufmag': [], 'mucz': [],
                  'mufz': [], 'sigmacmag': [],
                  'sigmacz': [], 'sigmafmag': [], 'sigmafz': []}
        mdtype = np.dtype([('param', 'S10'), ('value', np.float)])
        model = np.loadtxt(self.modelfile)

        for i in range(len(model['param'])):

            if model['param'][i][:2] == 'pz':
                params['pz'].append(model['value'][i])
            elif model['param'][i][:3] == 'cmz':
                params['mucz'].append(model['value'][i])
            elif model['param'][i][:3] == 'fmz':
                params['mufz'].append(model['value'][i])
            elif model['param'][i][:3] == 'csz':
                params['sigmacz'].append(model['value'][i])
            elif model['param'][i][:3] == 'fsz':
                params['sigmafz'].append(model['value'][i])
            elif model['param'][i][:1] == 'p':
                params['pmag'].append(model['value'][i])
            elif model['param'][i][:2] == 'cm':
                params['mucmag'].append(model['value'][i])
            elif model['param'][i][:2] == 'fm':
                params['mufmag'].append(model['value'][i])
            elif model['param'][i][:2] == 'cs':
                params['sigmacmag'].append(model['value'][i])
            elif model['param'][i][:2] == 'fs':
                params['sigmafmag'].append(model['value'][i])

        self.params = params

    def getParamsZL(self, z, mag):
        """Get the density pdf params for the given redshift and magnitude.

        Parameters
        ----------
        z : np.array
            Array of redshifts to get
        mag : type
            Description of parameter `mag`.

        Returns
        -------
        type
            Description of returned object.

        """

        pmag = np.sum(np.array(
            [self.params['pmag'][k] * mag**k for
             k in range(len(self.params['pmag']))]),
            axis=0)
        pz = np.sum(
            np.array([self.params['pz'][k] * z**k for
                      k in range(len(self.params['pz']))]),
            axis=0)
        p = pmag + pz

        mucmag = np.sum(np.array(
            [self.params['mucmag'][k] * mag**k for
             k in range(len(self.params['mucmag']))]),
            axis=0)
        mucz = np.sum(np.array(
            [self.params['mucz'][k] * z**k for
             k in range(len(self.params['mucz']))]),
            axis=0)
        muc = mucmag + mucz

        mufmag = np.sum(np.array(
            [self.params['mufmag'][k] * mag**k for
             k in range(len(self.params['mufmag']))]),
            axis=0)
        mufz = np.sum(np.array(
            [self.params['mufz'][k] * z**k for
             k in range(len(self.params['mufz']))]),
            axis=0)
        muf = mufmag + mufz

        sigmacmag = np.sum(np.array(
            [self.params['sigmacmag'][k] * mag**k for
             k in range(len(self.params['sigmacmag']))]),
            axis=0)
        sigmacz = np.sum(np.array(
            [self.params['sigmacz'][k] * z**k for
             k in range(len(self.params['sigmacz']))]),
            axis=0)
        sigmac = sigmacmag + sigmacz

        sigmafmag = np.sum(np.array(
            [self.params['sigmafmag'][k] * mag**k for
             k in range(len(self.params['sigmafmag']))]),
            axis=0)
        sigmafz = np.sum(np.array(
            [self.params['sigmafz'][k] * z**k for
             k in range(len(self.params['sigmafz']))]),
            axis=0)
        sigmaf = sigmafmag + sigmafz

        return p, muc, sigmac, muf, sigmaf

    def pofR(self, r, z, mag):

        dmag = 0.05
        weight1 = self.luminosityFunction.numberDensitySingleZL(z, mag+dmag)
        weight2 = self.luminosityFunction.numberDensitySingleZL(z, mag-dmag)

        pr1 = self.getParamsZL(z, mag+dmag)
        pr2 = self.getParamsZL(z, mag-dmag)

        # calculate the cululative distribution of r in a range of magnitudes
        # bracketing mag
        p1 = 0.5 * (1. - pr1[4]) * (1 + erf((np.log(r) - pr1[0]) /
                                    (pr1[1] * np.sqrt(2.0))))
        p2 = 0.5 * pr1[4] * (1 + erf((r - pr1[2]) / (pr1[3] * np.sqrt(2.0))))

        p3 = 0.5 * (1. - pr2[4]) * (1 + erf((np.log(r) - pr2[0]) /
                                    (pr2[1] * np.sqrt(2.0))))
        p4 = 0.5 * pr2[4] * (1 + erf((r - pr2[2]) / (pr2[3] * np.sqrt(2.0))))

        prob = weight1 * (p1 + p2) - weight2 * (p3 + p4)
        prob /= prob[-1]

        return prob


    def sampleDensity(self, domain, cosmo, z, mag):
        """Draw densities for galaxies at redshifts z and magnitudes m

        Parameters
        ----------
        z : np.array
            Redshifts of the galaxies we're adding
        mag : np.array
            Magnitudes of the galaxies we're adding

        Returns
        -------
        density : np.array
            Sampled densities

        """
        n_gal = z.size
        zbins = np.arange(domain.zmin, domain.zmax + 0.001, 0.001)
        zmean = zbins[1:] + zbins[:-1]

        magbins = np.arange(np.min(mag), np.max(mag) + 0.05, 0.05)
        magmean = (magbins[1:] + magbins[:-1]) / 2

        # sort galaxies by redshift
        zidx = z.argsort()
        z = z[zidx]
        mag = mag[zidx]

        deltabins = np.logspace(-3., np.log10(15.), 51)
        deltamean = (deltabins[1:] + deltabins[:-1]) / 2
        deltadelta = deltabins[1:] - deltabins[:-1]

        density = np.zeros(n_gal)
        count = 0

        for i in range(nzbins):
            zlidx = z.searchsorted(zbins[i])
            zhidx = z.searchsorted(zbins[i + 1])

            midx = mag[zlidx:zhidx].argsort()
            z[zlidx:zhidx] = z[zlidx:zhidx][midx]
            mag[zlidx:zhidx] = mag[zlidx:zhidx][midx]

            for j in range(nmagbins):
                mlidx = mag[zlidx:zhidx].searchsorted(magbins[j])
                mhidx = mag[zlidx:zhidx].searchsorted(magbins[j + 1])

                nij = mhidx - mlidx

                params = self.getParams(zmean[i], magmean[j])
                pdelta = self.pofDelta(deltamean, zmean[i], magmean[j])

                cdf_delta = np.cumsum(pdelta * deltadelta)
                cdf_delta /= cdf_delta[-1]

                rands = np.random.uniform(size=nij)
                density[count: count + nij] = density[cdf.searchsorted(rands)]
                count += nij

        return density


class RedFractionModel(object):

    def __init__(self, modelfile=None, **kwargs):
        pass


class ColorModel(object):

    def __init__(self, modelfile, **kwargs):
        pass

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
        pass

        

    def densityPDF(self, r, p, muc, sigmac, muf, sigmaf):

        return ((1 - p) * np.exp(-(np.log(rmean) - muc) ** 2
                / (2 * sigmac ** 2)) / ( rmean * np.sqrt(2 * np.pi )
                * sigmac ) + p * np.exp(-(rmean - muf) ** 2
                / (2 * sigmaf ** 2)) / (np.sqrt(2 * np.pi ) * sigmaf ))





class RedFractionModel(object):

    def __init__(self, modelfile=None, **kwargs):
        pass


class ColorModel(object):

    def __init__(self, modelfile, **kwargs):
        pass

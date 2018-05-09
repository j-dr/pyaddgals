from __future__ import print_function, division

from .galaxyModel import GalaxyModel
from . import luminosityFunction

class ADDGALSModel(GalaxyModel):

    def __init__(self, nbody, luminosityFunctionConfig=None,
                    rdelModelConfig=None,
                    fredModelConfig=None, colorModelConfig=None):

        self.nbody = nbody

        if luminosityFunctionConfig is None:
            raise(ValueError('ADDGALS model must define luminosityFunctionConfig'))

        if rdelModelConfig is None:
            raise(ValueError('ADDGALS model must define rdelModelConfig'))

        if fredModelConfig is None:
            raise(ValueError('ADDGALS model must define fredModelConfig'))

        if colorModelConfig is None:
            raise(ValueError('ADDGALS model must define colorModelConfig'))

        self.rdelModel = RdelModel(**rdelModelConfig)
        self.fredModel = FredModel(**fredModelConfig)
        self.colorModel = ColorModel(**colorModelConfig)

        lf_type = luminosityFunctionConfig['modeltype']

        self.luminosityFunction = getattr(luminosityFunction, lf_type)
        self.luminosityFunction = self.luminosityFunction(**luminosityFunctionConfig)


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

        galaxies = self.nbody.galaxyCatalog.catalog
        domain = self.nbody.domain
        cosmo = self.nbody.cosmo

        n_gal = self.luminosityFunction.integrate(cosmo, domain.zmin,
                                                    domain.zmax,
                                                    domain.getArea())
        galaxies['redshift_true'] = self.drawRedshifts(n_gal)

        self.luminosityFunction.drawLuminosities(cosmo, domain)


        self.catalog = None



class RdelModel(object):

    def __init__(self, modelfile=None, **kwargs):

        self.modelfile = modelfile

        if self.modelfile is None:

            for k in kwargs.keys():
                setattr(self, k) = kwargs[]

    def densityPDF(self, r, p, muc, sigmac, muf, sigmaf):

        return ((1 - p) * np.exp(-(np.log(rmean) - muc) ** 2
                / (2 * sigmac ** 2)) / ( rmean * np.sqrt(2 * np.pi )
                * sigmac ) + p * np.exp(-(rmean - muf) ** 2
                / (2 * sigmaf ** 2)) / (np.sqrt(2 * np.pi ) * sigmaf ))





class RedFractionModel(object):

    def __init__(self, modelfile=None, **kwargs):


class ColorModel(object):

    def __init__(self, modelfile, **kwargs):

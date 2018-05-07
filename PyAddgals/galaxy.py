from __future__ import print_function, division
import numpy as np

from . import luminosityFunction
from . import galaxyModel


class GalaxyCatalog(object):
    """
    Galaxy catalog class

    """

    def __init__(self, nbody):

        self.nbody = nbody

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

        model_class = config.pop('model')
        model = getattr(galaxyModel, model_class)
        model = model(**config)

        self.catalog = model.paintGalaxies(self.lightcone)

    def writeCatalog(self):
        """Write galaxy catalog to disk.

        Returns
        -------
        None
        """

        pass

    def delete(self):
        """Delete galaxy catalog

        Returns
        -------
        None

        """

        for k in self.catalog.keys():

            del self.catalog[k]

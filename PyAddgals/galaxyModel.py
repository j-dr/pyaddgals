from __future__ import print_function, division
from abc import ABCMeta, abstractmethod


class GalaxyModel(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def paintGalaxies(self, nbody):
        """Abstract method taking in an nbody object and
            returning a galaxy catalog.

        Parameters
        ----------
        nbody : NBody
            A Nbody object containing DM and halo information for the simulation
            to populate with galaxies

        Returns
        -------
        catalog : dict
            A dictionary whose keys are attributes of the galaxy catalog and
            whose values are arrays containing those attributes for each galaxy.

        """
        return catalog


class ADDGALSModel(GalaxyModel):

    def __init__(self, LuminosityFunction=None, RdelModel=None,
                    FredModel=None, ColorModel=None):
        pass

    def paintGalaxies(self, nbody):
        """Paint galaxies into nbody using ADDGALS

        Parameters
        ----------
        nbody : NBody
            The nbody object to paint galaxies into.

        Returns
        -------
        None
        """

        self.catalog = None

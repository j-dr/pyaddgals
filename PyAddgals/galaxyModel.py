from __future__ import print_function, division
from abc import ABCMeta, abstractmethod

class GalaxyModel(object):

    __metaclass__ = ABCMeta

    def __init__(self, nbody):
        self.nbody = nbody
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
        catalog = None
        return catalog

    def drawRedshifts(self, n_gal):
        """Draw redshifts proportional to r^3

        Parameters
        ----------
        n_gal : int
            number of galaxies to draw redshifts for

        Returns
        -------
        z : np.array
            Array of length n_gal containing redshifts of the galaxies.

        """

        rmin = self.nbody.domain.rmin
        rmax = self.nbody.domain.rmax

        rands = np.random.uniform(size=n_gal)
        z = ((rmax ** 3 - rmin ** 3) * rands + rmin ** 3) ** (1/3)

        return z

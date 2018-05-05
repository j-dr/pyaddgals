from __future__ import print_function, division
import pyccl as ccl


class NBody(object):
    """Object which stores all nbody data.
    """

    def __init__(self, cosmo, domain, partpath=None, denspath=None,
                 hinfopath=None, halopath=None):
        """Create NBody object.

        Parameters
        ----------
        cosmo : ccl.core.Cosmology
            Object containing cosmology information
        domain : Domain
            Domain object containing information about the piece of
            the simulation that a particular process is responsible for and
            necessary io information for that piece.
        partpath : str
            Path to particle data
        denspath : str
            Path to density data
        hinfopath : str
            Path to halo-particle data
        halofile : str
            Path to halo catalog

        Returns
        -------
        None
        """

        self.cosmo = cosmo
        self.domain = domain
        self.partpath = partpath
        self.denspath = denspath
        self.hinfopath = hinfopath
        self.halofile = halofile

        if not self.partpath:
            raise(ValueError("partpath, path to particle data must be defined for nbody"))

        if not self.denspath:
            raise(ValueError(
                "denspath, path to particle and halo density data must be defined for nbody"))

        if not self.hinfopath:
            raise(ValueError(
                "hinfopath, path to particle-halo data must be defined for nbody"))

        if not self.halofile:
            raise(ValueError(
                "halofile, path to input halo catalog must be defined for nbody"))

        self.particleCatalog = ParticleCatalog(self)
        self.haloCatalog = HaloCatalog(self)
        self.galaxyCatalog = GalaxyCatalog(self)

    def read(self):

        self.particleCatalog.read()
        self.haloCatalog.read()

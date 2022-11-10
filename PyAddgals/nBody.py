from __future__ import print_function, division

from .particle import ParticleCatalog
from .halo import HaloCatalog
from .galaxy import GalaxyCatalog


class NBody(object):
    """Object which stores all nbody data.
    """

    def __init__(self, cosmo, domain, partpath=None, denspath=None,
                 hinfopath=None, halofile=None, halodensfile=None,
                 n_blocks=None, f_downsample=1.):
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
        n_blocks : int/str or list
            Number of blocks in the snapshot

        Returns
        -------
        None
        """

        self.cosmo = cosmo
        self.domain = domain

        if not partpath:
            raise(ValueError("partpath, path to particle data must be defined for nbody"))

        if not denspath:
            raise(ValueError(
                "denspath, path to particle and halo density data must be defined for nbody"))

        if self.domain.fmt == 'BCCLightcone':
            if not hinfopath:
                raise(ValueError(
                    "hinfopath, path to particle-halo data must be defined for nbody"))

        if not halofile:
            raise(ValueError(
                "halofile, path to input halo catalog must be defined for nbody"))

        if not halodensfile:
            raise(ValueError(
                "halodensfile, path to density measurements for halo catalog must be defined for nbody"))

        if self.domain.fmt == 'Snapshot':
            if not n_blocks:
                raise(ValueError(
                    "n_blocks, number of blocks in snapshot must be defined for nbody snapshots"))

        # make all lists so we can deal with stitching together lightcones
        if isinstance(partpath, str):
            self.partpath = [partpath]
        else:
            self.partpath = partpath

        if isinstance(denspath, str):
            self.denspath = [denspath]
        else:
            self.denspath = denspath

        if isinstance(hinfopath, str):
            self.hinfopath = [hinfopath]
        else:
            self.hinfopath = hinfopath

        if isinstance(halofile, str):
            self.halofile = [halofile]
        else:
            self.halofile = halofile

        if isinstance(halodensfile, str):
            self.halodensfile = [halodensfile]
        else:
            self.halodensfile = halodensfile

        if self.domain.fmt == 'Snapshot':
            if isinstance(n_blocks, str) | isinstance(n_blocks, (int, float, complex)):
                self.n_blocks = [n_blocks]
            else:
                self.n_blocks = n_blocks

        if isinstance(f_downsample, str) | isinstance(f_downsample, (int, float, complex)):
            self.f_downsample = [f_downsample] * len(self.partpath)
        else:
            self.f_downsample = f_downsample

        self.particleCatalog = ParticleCatalog(self)
        self.haloCatalog = HaloCatalog(self)
        self.galaxyCatalog = GalaxyCatalog(self)

        self.boxnum = self.domain.boxnum

    def read(self):
        if (self.domain.fmt == 'BCCLightcone') | (self.domain.fmt == 'FastPMLightcone'):
            self.haloCatalog.read()
            self.particleCatalog.read()
        else:
            self.particleCatalog.read()
            self.haloCatalog.read()

    def delete(self):
        self.particleCatalog.delete()
        self.haloCatalog.delete()
        self.galaxyCatalog.delete()

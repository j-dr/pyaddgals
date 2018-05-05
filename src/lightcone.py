from __future__ import print_function, division
import pyccl as ccl


class Lightcone(object):
    """Object which stores all lightcone data.
    """
    def __init__(self, basepath, cosmo, fmt="BCC", pix=None,
                  nside=None,zmin=None, zmax=None, rnnpath=None,
                  nest=False, **kwargs):
        """Create lightcone object.

        Parameters
        ----------
        basepath : str
            Path to lightcone data.
        cosmo : ccl.core.Cosmology
            Object containing cosmology information
        fmat : str
            Lightcone format
        pix : int
            If fmat==BCC, healpix cell that we should read
        nside : int
            if fmat==BCC, nside of domain decomposition
        zmin : float
            if fmat==BCC, minimum redshift to be read
        zmax : float
            if fmat==BCC, maximum redshift to be read
        rnnpath : str
            if fmat==BCC, base path to rnn files

        Returns
        -------
        None

        """
        self.basepath = basepath
        self.cosmo    = cosmo
        self.fmat     = fmat
        self.pix      = pix
        self.nside    = nside
        self.zmin     = zmin
        self.zmax     = zmax
        self.rnnpath  = rnnpath
        self.nest     = nest

        #make sure all relevant info is defined for BCC catalog
        if self.fmat='BCC':
            if not self.pix:
                raise(ValueError("Pixel must be defined for BCC style catalog"))

            if not self.nside:
                raise(ValueError("Nside must be defined for BCC style catalog"))

            if not self.zmin:
                raise(ValueError("zmin must be defined for BCC style catalog"))

            if not self.zmax:
                raise(ValueError("zmax must be defined for BCC style catalog"))

            if not self.rnnpath:
                raise(ValueError("rnnpath must be defined for BCC style catalog"))

        self.rmin, self.rmin = self.getRadialLimits()

        self.particleCatalog = ParticleCatalog(self)
        self.haloCatalog     = HaloCatalog(self)
        self.galaxyCatalog   = GalaxyCatalog(self)


    def getRadialLimits(self):
        """Get the radial limits of the lightcone given the redshift limits

        Returns
        -------
        rmin : float
            Minimum radius

        rmax : float
            Maximum radius

        """

        rmin = ccl.comoving_radial_distance(cosmo, 1/(1+self.zmin))
        rmax = ccl.comoving_radial_distance(cosmo, 1/(1+self.zmax))

        return rmin, rmax

    def read(self):

        self.particleCatalog.read()
        self.haloCatalog.read()

    def addGalaxies(self):

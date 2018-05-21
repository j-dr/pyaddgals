from __future__ import print_function, division
from scipy.special import erf
from numba import jit, boolean
import numpy as np

from .galaxyModel import GalaxyModel
from . import luminosityFunction


@jit(nopython=True)
def assign(magnitude, redshift, density, z_part, density_part, dz=0.01):

    n_gal = magnitude.size
    n_part = density_part.size

    max_search_count = n_part // 1000
    max_search_d = 0.1

    idx_part = np.zeros(n_gal, dtype=np.int32)
    bad = np.zeros(n_gal, dtype=boolean)
    nassigned = np.ones(density_part.size, dtype=boolean)

    for i in range(n_gal):

        pidx = np.searchsorted(density_part, density[i])
        pidx -= 1
        pi = 0

        minz = redshift[i] - dz
        maxz = redshift[i] + dz
        delta_dens = 0.0
        assigned = False

        while ((not assigned) & (pi < max_search_count) &
                (delta_dens < (max_search_d))):

            if ((pidx - pi) >= 0) & ((pidx - pi) < n_part):
                if np.abs(density_part[pidx - pi] - density[i]) > delta_dens:
                    delta_dens = np.abs(density_part[pidx - pi] - density[i])

                if (nassigned[pidx - pi] & (minz < z_part[pidx - pi]) &
                        (z_part[pidx - pi] < maxz)):
                    idx_part[i] = pidx - pi
                    nassigned[pidx - pi] = False
                    assigned = True

            if ((pidx + pi) < n_part) & ((pidx + pi) >= 0):
                if (np.abs(density_part[pidx + pi] - density[i])) > delta_dens:
                    delta_dens = np.abs(density_part[pidx + pi] - density[i])

                if (nassigned[pidx + pi] & (minz < z_part[pidx + pi]) &
                        (z_part[pidx + pi] < maxz)):
                    idx_part[i] = pidx + pi
                    nassigned[pidx + pi] = False
                    assigned = True

            pi += 1

        if not assigned:
            bad[i] = True
            while (not assigned):
                if (pidx - pi) >= 0:
                    if nassigned[pidx - pi]:
                        idx_part[i] = pidx - pi
                        assigned = True

                if (pidx + pi) < n_part:
                    if nassigned[pidx + pi]:
                        idx_part[i] = pidx + pi
                        assigned = True

                pi += 1

    return idx_part


class ADDGALSModel(GalaxyModel):

    def __init__(self, nbody, luminosityFunctionConfig=None,
                 rdelModelConfig=None,
                 colorModelConfig=None):

        self.nbody = nbody

        if luminosityFunctionConfig is None:
            raise(ValueError('ADDGALS model must define luminosityFunctionConfig'))

        if rdelModelConfig is None:
            raise(ValueError('ADDGALS model must define rdelModelConfig'))

        if colorModelConfig is None:
            raise(ValueError('ADDGALS model must define colorModelConfig'))

        lf_type = luminosityFunctionConfig.pop('modeltype')

        self.luminosityFunction = getattr(luminosityFunction, lf_type)
        self.luminosityFunction = self.luminosityFunction(
            nbody.cosmo, **luminosityFunctionConfig)

        self.rdelModel = RdelModel(self.luminosityFunction, **rdelModelConfig)
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

        zidx = z.argsort()
        z = z[zidx]
        mag = mag[zidx]

        density, idx = self.rdelModel.sampleDensities(z, mag)

        z = z[idx]
        mag = mag[idx]

        pos, vel, z, density, mag, _ = self.assignParticles(z, mag, density)

        self.nbody.galacyCatalog.catalog['pos'] = pos
        self.nbody.galaxyCatalog.catalog['vel'] = vel
        self.nbody.galaxyCatalog.catalog['z'] = z
        self.nbody.galaxyCatalog.catalog['mag'] = mag
        self.nbody.galaxyCatalog.catalog['rnn'] = density

        id_train, coeff = self.colorModel.assignSEDs(mag, z, pos)

    def assignParticles(self, redshift, magnitude, density):
        """Assign galaxies to particles with the correct redshift
        and density.

        Parameters
        ----------
        z : np.array
            Array of redshifts of dimension (N)
        mag : np.array
            Array of magnitudes of dimension (N)
        density : np.array
            Array of densities of dimension (N)

        Returns
        -------
        pos : np.array
            Positions of galaxies
        vel : np.array
            Velocities of galaxies
        z : np.array
            Redshifts of galaxies
        density : np.array
            Densities of galaxies
        """

        midx = magnitude.argsort()
        redshift = redshift[midx]
        magnitude = magnitude[midx]
        density = density[midx]

        density_part = self.nbody.catalog['rnn']
        z_part = self.nbody.catalog['z']

        didx = density_part.argsort()
        density_part = density_part[didx]
        z_part = z_part[didx]

        idx = assign(magnitude, redshift, density, z_part, density_part)
        pos = self.nbody.particleCatalog.catalog['pos'][didx][idx]
        vel = self.nbody.particleCatalog.catalog['vel'][didx][idx]
        z_asn = z_part[idx]
        density_asn = density_part[idx]

        return pos, vel, z_asn, density_asn, redshift, magnitude


class RdelModel(object):

    def __init__(self, lf, modelfile=None, **kwargs):

        self.luminosityFunction = lf
        self.modelfile = modelfile

        if self.modelfile is None:
            for k in kwargs.keys():
                setattr(self, k, kwargs[k])
        else:
            self.loadModelFile()

    def loadModelFile(self):
        """Load a model from file into self.model

        Returns
        -------
        None

        """

        assert(self.modelfile is not None)

        mdtype = np.dtype([('param', 'S10'), ('value', np.float)])
        model = np.loadtxt(self.modelfile, dtype=mdtype)

        idx = model['param'].argsort()
        model = model[idx]

        self.params = {}

        self.params['muc'] = model['value'][:15]
        self.params['sigmac'] = model['value'][15:30]
        self.params['muf'] = model['value'][30:45]
        self.params['sigmaf'] = model['value'][45:60]
        self.params['p'] = model['value'][60:75]

    def makeVandermonde(self, z, mag, bmlim, fmlim, mag_ref):
        """Make a vandermonde matrix out of redshifts and luminosities

        Parameters
        ----------
        z : np.array
            Galaxy redshifts, dimension (n)
        mag : np.array
            Galaxy luminosities, dimension (m)
        bmlim : float
            Bright luminosity limit
        fmlim : float
            Faint luminosity limit
        mag_ref : float
            Reference magnitude

        Returns
        -------
        xvec : np.array
            vandermonde matrix of dimension (n,m)

        """

        bright_mag_lim = bmlim - mag_ref
        faint_mag_lim = fmlim - mag_ref

        x = np.meshgrid(mag, z)

        zv = 1 / (x[1].flatten() + 1) - 0.35
        mv = x[0].flatten()
        mv = mv - mag_ref
        mv[mv < bright_mag_lim] = bright_mag_lim
        mv[mv > faint_mag_lim] = faint_mag_lim

        o = np.ones(len(zv))

        # construct vandermonde matrix
        xvec = np.array([o, mv, mv * zv, mv * zv**2, mv * zv**3,
                         mv**2, mv**2 * zv, mv**2 * zv**2,
                         mv**3, mv**3 * zv, mv**4, zv,
                         zv**2, zv**3, zv**4])

        return xvec

    def getParamsZL(self, z, mag, magbright=-22.5, magfaint=-18., magref=-20.5):
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
        x = self.makeVandermonde(z, mag, magbright, magfaint, magref)

        p = np.dot(self.params['p'], x)
        muc = np.dot(self.params['muc'], x)
        muf = np.dot(self.params['muf'], x)
        sigmac = np.dot(self.params['sigmac'], x)
        sigmaf = np.dot(self.params['sigmaf'], x)

        return muc, sigmac, muf, sigmaf, p

    def pofR(self, r, z, mag, dmag=0.05):

        weight1 = self.luminosityFunction.cumulativeNumberDensity(
            z, mag + dmag)
        weight2 = self.luminosityFunction.cumulativeNumberDensity(
            z, mag - dmag)

        pr1 = self.getParamsZL(z, mag + dmag)
        pr2 = self.getParamsZL(z, mag - dmag)

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

        nzbins = zmean.size
        nmagbins = magmean.size

        # sort galaxies by redshift
        zidx = z.argsort()
        z = z[zidx]
        mag = mag[zidx]

        deltabins = np.logspace(-3., np.log10(15.), 51)
        deltamean = (deltabins[1:] + deltabins[:-1]) / 2

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

                cdf_r = self.pofR(deltamean, zmean[i], magmean[j])

                rands = np.random.uniform(size=nij)
                density[count: count +
                        nij] = deltamean[cdf_r.searchsorted(rands) - 1]
                count += nij

        return density


class RedFractionModel(object):

    def __init__(self, modelfile=None, **kwargs):
        pass


class ColorModel(object):

    def __init__(self, modelfile, **kwargs):
        pass

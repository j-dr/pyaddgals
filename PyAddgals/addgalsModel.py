from __future__ import print_function, division
from scipy.special import erf
from numba import jit, boolean
from fast3tree import fast3tree
from copy import copy
from time import time
import numpy as np
import healpy as hp
import fitsio
import sys

from .galaxyModel import GalaxyModel
from .kcorrect import KCorrect, k_reconstruct_maggies
from . import luminosityFunction
from . import shape


@jit(nopython=True)
def assign(magnitude, redshift, density, z_part, density_part, dz=0.01):

    n_gal = magnitude.size
    n_part = density_part.size

    max_search_count = n_part // 50
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
                    pi += 1
                    continue

            if ((pidx + pi) < n_part) & ((pidx + pi) >= 0):
                if (np.abs(density_part[pidx + pi] - density[i])) > delta_dens:
                    delta_dens = np.abs(density_part[pidx + pi] - density[i])

                if (nassigned[pidx + pi] & (minz < z_part[pidx + pi]) &
                        (z_part[pidx + pi] < maxz)):
                    idx_part[i] = pidx + pi
                    nassigned[pidx + pi] = False
                    assigned = True
                    pi += 1
                    continue

            pi += 1

        max_search_count = n_part

        if not assigned:
            bad[i] = True
            while (not assigned) & (pi < max_search_count):
                if (pidx - pi) >= 0:
                    if (nassigned[pidx - pi] & (minz < z_part[pidx - pi]) &
                            (z_part[pidx - pi] < maxz)):
                        idx_part[i] = pidx - pi
                        assigned = True
                        nassigned[pidx - pi] = False

                if (pidx + pi) < n_part:
                    if (nassigned[pidx + pi] & (minz < z_part[pidx + pi]) &
                            (z_part[pidx + pi] < maxz)):
                        idx_part[i] = pidx + pi
                        assigned = True
                        nassigned[pidx + pi] = False

                pi += 1

        if not assigned:
            pi = 0
            while (not assigned) & (pi < max_search_count):
                if (pidx - pi) >= 0:
                    if (nassigned[pidx - pi]):
                        idx_part[i] = pidx - pi
                        assigned = True
                        nassigned[pidx - pi] = False

                if (pidx + pi) < n_part:
                    if (nassigned[pidx + pi]):
                        idx_part[i] = pidx + pi
                        assigned = True
                        nassigned[pidx + pi] = False

                pi += 1

    return idx_part, bad


@jit(nopython=True)
def assignLcen(redshift, magnitude, density, mass_halo, density_halo, z_halo,
               params, scatter, dMr=0.015):

    n_halo = z_halo.size
    n_gal = redshift.size
    m0 = params[0]
    mc = params[1]
    a = params[2]
    b = params[3]
    k = params[4]

    mr0 = m0 - 2.5 * (a * np.log10(mass_halo / mc) - b *
                      np.log10(1. + (mass_halo / mc)**(k / b)))
    mr0 = mr0 + np.random.randn(n_halo) * (2.5 * scatter)

    bad = np.zeros(n_halo, dtype=boolean)

    # made this as large as possible to avoid not assigning halos
    max_search_count = n_gal

    assigned = np.zeros(n_gal, dtype=boolean)

    for i in range(n_halo):

        magmin = mr0[i] - dMr
        magmax = mr0[i] + dMr

        pidx = np.searchsorted(density, density_halo[i])
        pidx -= 1
        pi = 0

        halo_assigned = False

        while ((not halo_assigned) & (pi < max_search_count)):

            if ((pidx - pi) >= 0) & ((pidx - pi) < n_gal):

                if ((not assigned[pidx - pi]) & (magmin < magnitude[pidx - pi]) &
                        (magnitude[pidx - pi] < magmax)):
                    assigned[pidx - pi] = True
                    halo_assigned = True
                    pi += 1
                    continue

            if ((pidx + pi) < n_gal) & ((pidx + pi) >= 0):

                if ((not assigned[pidx + pi]) & (magmin < magnitude[pidx + pi]) &
                        (magnitude[pidx + pi] < magmax)):
                    assigned[pidx + pi] = True
                    halo_assigned = True
                    pi += 1
                    continue

            pi += 1

        # if not assigned with fiducial magniude window, make larger
        if not halo_assigned:
            bad[i] = True
            pi = 0
            magmin = mr0[i] - 3 * dMr
            magmax = mr0[i] + 3 * dMr

            while ((not halo_assigned) & (pi < max_search_count)):

                if ((pidx - pi) >= 0) & ((pidx - pi) < n_gal):

                    if ((not assigned[pidx - pi]) & (magmin < magnitude[pidx - pi]) &
                            (magnitude[pidx - pi] < magmax)):
                        assigned[pidx - pi] = True
                        halo_assigned = True
                        pi += 1
                        continue

                if ((pidx + pi) < n_gal) & ((pidx + pi) >= 0):

                    if ((not assigned[pidx + pi]) & (magmin < magnitude[pidx + pi]) &
                            (magnitude[pidx + pi] < magmax)):
                        assigned[pidx + pi] = True
                        halo_assigned = True
                        pi += 1
                        continue

                pi += 1

    return assigned, mr0, bad


class ADDGALSModel(GalaxyModel):

    def __init__(self, nbody, luminosityFunctionConfig=None,
                 rdelModelConfig=None,
                 colorModelConfig=None,
                 shapeModelConfig=None):

        self.nbody = nbody

        if luminosityFunctionConfig is None:
            raise(ValueError('ADDGALS model must define luminosityFunctionConfig'))

        if rdelModelConfig is None:
            raise(ValueError('ADDGALS model must define rdelModelConfig'))

        if colorModelConfig is None:
            raise(ValueError('ADDGALS model must define colorModelConfig'))

        lf_type = luminosityFunctionConfig['modeltype']

        self.luminosityFunction = getattr(luminosityFunction, lf_type)
        self.luminosityFunction = self.luminosityFunction(
            nbody.cosmo, **luminosityFunctionConfig)

        self.rdelModel = RdelModel(self.nbody, self.luminosityFunction, **rdelModelConfig)
        self.colorModel = ColorModel(self.nbody, **colorModelConfig)
        self.c = 3e5

        if shapeModelConfig is None:
            self.shapeModel = None
        else:
            shape_type = shapeModelConfig['modeltype']

            self.shapeModel = getattr(shape, shape_type)
            self.shapeModel = self.shapeModel(nbody.cosmo, **shapeModelConfig)

    def paintGalaxies(self):
        """Paint galaxy positions, luminosities and SEDs into nbody.
        Saves them in self.galaxyCatalog.catalog.

        Returns
        -------
        None
        """

        self.paintPositions()

        if not self.colorModel.no_colors:
            self.paintSEDs()
            self.paintShapes()

    def paintPositions(self):
        """Paint galaxy positions and luminosity in one band
        into nbody using ADDGALS method.

        Returns
        -------
        None
        """

        domain = self.nbody.domain

        print('[{}] : Painting galaxy positions'.format(self.nbody.domain.rank))
        sys.stdout.flush()
        start = time()
        overdens = self.nbody.particleCatalog.calculateOverdensity()

        print('[{}] : Domain has overdensity: {}'.format(self.nbody.domain.rank,
                                                         overdens))
        z = self.luminosityFunction.drawRedshifts(domain, overdens)
        z.sort()

        if domain.fmt == 'Snapshot':
            mag = self.luminosityFunction.sampleLuminositiesSnap(domain, z)
        else:
            mag = self.luminosityFunction.sampleLuminosities(domain, z)
        print('mag from lum')
        print(mag)
        zidx = z.argsort()
        z = z[zidx]
        mag = mag[zidx]
        del zidx

        if domain.fmt == 'Snapshot':
            density, mag = self.rdelModel.sampleDensitySnap(domain, mag)
        else:
            density, z, mag = self.rdelModel.sampleDensity(domain, z, mag)

        print(mag)
        print(density)
        end = time()

        print('[{}] Finished drawing mag, z, dens. {} galaxies in domain, took {}s'.format(self.nbody.domain.rank, len(z), end - start))
        sys.stdout.flush()

        # sort galaxies by density
        idx = density.argsort()
        z = z[idx]
        mag = mag[idx]
        density = density[idx]
        del idx

        start = time()
        mag_cen, assigned, bad_cen = self.assignHalos(z, mag, density)
        end = time()

        print('[{}] Finished assigning galaxies to halos. Took {}s'.format(self.nbody.domain.rank, end - start))
        sys.stdout.flush()

        start = time()
        pos, vel, z, density, mag, rhalo, halorad, haloid, halomass, bad = self.assignParticles(
            z[~assigned], mag[~assigned], density[~assigned])
        end = time()

        print('[{}] Finished assigning galaxies to particles. Took {}s'.format(self.nbody.domain.rank, end - start))
        sys.stdout.flush()

        n_halo = mag_cen.size

        pos = np.vstack([self.nbody.haloCatalog.catalog['pos'], pos])
        print('velocity halo min, halo max, part min, part max: {}, {}, {}, {}'.format(np.min(self.nbody.haloCatalog.catalog['vel']),
              np.max(self.nbody.haloCatalog.catalog['vel']), np.min(vel), np.max(vel)))
        vel = np.vstack([self.nbody.haloCatalog.catalog['vel'], vel])
        z = np.hstack([self.nbody.haloCatalog.catalog['z'], z])
        mag = np.hstack([mag_cen, mag])
        density = np.hstack([self.nbody.haloCatalog.catalog['rnn'], density])
        halomass = np.hstack(
            [self.nbody.haloCatalog.catalog['mass'], halomass])
        halorad = np.hstack([self.nbody.haloCatalog.catalog['radius'], halorad])
        rhalo = np.hstack([np.zeros(n_halo), rhalo])
        central = np.zeros(rhalo.size)
        central[:n_halo] = 1
        haloid = np.hstack([self.nbody.haloCatalog.catalog['id'], haloid])
        z_rsd = z + np.sum(pos * vel, axis=1) / np.sqrt(np.sum(pos**2, axis=1)) / 299792.458
        bad = np.hstack([bad_cen, bad])

        #done with halo catalog now
        self.nbody.haloCatalog.delete()

        self.nbody.galaxyCatalog.catalog['PX'] = pos[:, 0]
        self.nbody.galaxyCatalog.catalog['PY'] = pos[:, 1]
        self.nbody.galaxyCatalog.catalog['PZ'] = pos[:, 2]
        self.nbody.galaxyCatalog.catalog['VX'] = vel[:, 0]
        self.nbody.galaxyCatalog.catalog['VY'] = vel[:, 1]
        self.nbody.galaxyCatalog.catalog['VZ'] = vel[:, 2]
        self.nbody.galaxyCatalog.catalog['Z_COS'] = z
        self.nbody.galaxyCatalog.catalog['Z'] = z_rsd
        self.nbody.galaxyCatalog.catalog['MAG_R'] = mag
        self.nbody.galaxyCatalog.catalog['DIST8'] = density
        self.nbody.galaxyCatalog.catalog['M200'] = halomass
        self.nbody.galaxyCatalog.catalog['R200'] = halorad
        self.nbody.galaxyCatalog.catalog['RHALO'] = rhalo
        self.nbody.galaxyCatalog.catalog['HALOID'] = haloid
        self.nbody.galaxyCatalog.catalog['CENTRAL'] = central
        self.nbody.galaxyCatalog.catalog['BAD_ASSIGN'] = bad

    def paintSEDs(self):
        """Paint SEDs onto galaxies after positions and luminosities have
        already been assigned.

        Returns
        -------
        None

        """

        print('[{}] : Painting galaxy SEDs'.format(self.nbody.domain.rank))
        sys.stdout.flush()

        pos = np.vstack([self.nbody.galaxyCatalog.catalog['PX'],
                         self.nbody.galaxyCatalog.catalog['PY'],
                         self.nbody.galaxyCatalog.catalog['PZ']]).T
        mag = self.nbody.galaxyCatalog.catalog['MAG_R']
        z = self.nbody.galaxyCatalog.catalog['Z_COS']
        z_rsd = self.nbody.galaxyCatalog.catalog['Z']

        sigma5, ranksigma5, redfraction, \
            sed_idx, omag, amag, mag_evol = self.colorModel.assignSEDs(pos, mag, z, z_rsd)

        self.nbody.galaxyCatalog.catalog['SIGMA5'] = sigma5
        self.nbody.galaxyCatalog.catalog['PSIGMA5'] = ranksigma5
        self.nbody.galaxyCatalog.catalog['SEDID'] = sed_idx
        self.nbody.galaxyCatalog.catalog['MAG_R_EVOL'] = mag_evol
        self.nbody.galaxyCatalog.catalog['TMAG'] = omag
        self.nbody.galaxyCatalog.catalog['AMAG'] = amag

    def paintShapes(self):
        """Assign shapes to galaxies.

        Returns
        -------
        None
        """

        if self.shapeModel is None:
            return

        log_comoving_size, angular_size, epsilon = self.shapeModel.sampleShapes(self.nbody.galaxyCatalog.catalog)

        self.nbody.galaxyCatalog.catalog['TSIZE'] = angular_size
        self.nbody.galaxyCatalog.catalog['TE'] = epsilon
        self.nbody.galaxyCatalog.catalog['EPSILON_IA'] = np.zeros_like(epsilon)
        self.nbody.galaxyCatalog.catalog['COMOVING_SIZE'] = 10**log_comoving_size

    def assignHalos(self, z, mag, dens):
        """Assign central galaxies to resolved halos. Halo catalog
        will be cut to minimum mass and subhalos removed if they
        are not used. Assumes that inputs are sorted by density.

        Parameters
        ----------
        z : np.array
            Array of redshifts of dimension (N)
        mag : np.array
            Array of magnitudes of dimension (N)
        dens : np.array
            Array of densities of dimension (N)

        Returns
        -------
        lcen : np.array
            Magniutdes of galaxies assigned to halos
        assigned : np.array
            Whether or not a (magnitude, z, density) tuple has been assigned
            to a halo or not

        """

        # cut out low mass and subhalos if necessary
        idx = self.nbody.haloCatalog.catalog['mass'] >= self.rdelModel.lcenMassMin[self.nbody.domain.boxnum]
        if not self.rdelModel.useSubhalos:
            idx &= (self.nbody.haloCatalog.catalog['pid'] == -1)

        for k in self.nbody.haloCatalog.catalog.keys():
            self.nbody.haloCatalog.catalog[k] = self.nbody.haloCatalog.catalog[k][idx]

        # sort halos by density
        idx = self.nbody.haloCatalog.catalog['rnn'].argsort()

        for k in self.nbody.haloCatalog.catalog.keys():
            self.nbody.haloCatalog.catalog[k] = self.nbody.haloCatalog.catalog[k][idx]

        mass_halo = self.nbody.haloCatalog.catalog['mass']
        density_halo = self.nbody.haloCatalog.catalog['rnn']
        z_halo = self.nbody.haloCatalog.catalog['z']

        amean = 1 / (self.nbody.domain.zmean + 1)

        idx = self.rdelModel.lcenModel['scale'].searchsorted(amean)

        if idx < 0:
            idx = 0

        # numba doesn't like record arrays
        params = [self.rdelModel.lcenModel['M0'][idx],
                  self.rdelModel.lcenModel['Mc'][idx],
                  self.rdelModel.lcenModel['a'][idx],
                  self.rdelModel.lcenModel['b'][idx],
                  self.rdelModel.lcenModel['k'][idx]]

        assigned, lcen, bad = assignLcen(z, mag, dens, mass_halo, density_halo,
                                         z_halo, params, self.rdelModel.scatter)

        print('n_bad halos: {}'.format(np.sum(bad)))
        sys.stdout.flush()

        return lcen, assigned, bad

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
            Positions of particles assigned to galaxies
        vel : np.array
            Velocities of particles assigned to galaxies
        z_asn : np.array
            Redshifts of particles assigned to galaxies
        density_asn : np.array
            Densities of particles assigned to galaxies
        magnitude : np.array
            Sorted galaxy magnitudes
        rhalo : np.array
            Distance to nearest halo for particles assigned to galaxies
        haloid : np.array
            Haloid of nearest halo for particles assigned to galaxies
        halomass : np.array
            Mass of nearest halo for particles assigned to galaxies
        """

        midx = magnitude.argsort()
        redshift = redshift[midx]
        magnitude = magnitude[midx]
        density = density[midx]

        density_part = self.nbody.particleCatalog.catalog['rnn']
        z_part = self.nbody.particleCatalog.catalog['z']

        didx = density_part.argsort()
        density_part = density_part[didx]
        z_part = z_part[didx]

        idx, bad = assign(magnitude, redshift, density, z_part, density_part)
        pos = self.nbody.particleCatalog.catalog['pos'][didx][idx]
        vel = self.nbody.particleCatalog.catalog['vel'][didx][idx]
        rhalo = self.nbody.particleCatalog.catalog['rhalo'][didx][idx]
        haloradius = self.nbody.particleCatalog.catalog['radius'][didx][idx]
        haloid = self.nbody.particleCatalog.catalog['haloid'][didx][idx]
        halomass = self.nbody.particleCatalog.catalog['mass'][didx][idx]
        z_asn = z_part[idx]
        density_asn = density_part[idx]

        self.nbody.particleCatalog.delete()
        del z_part, density_part

        print('[{}] number of bad assignments: {}'.format(self.nbody.domain.rank, np.sum(bad)))
        sys.stdout.flush()

        return pos, vel, z_asn, density_asn, magnitude, rhalo, haloradius, haloid, halomass, bad


class RdelModel(object):

    def __init__(self, nbody, lf, rdelModelFile=None, lcenModelFile=None,
                 lcenMassMin=None, useSubhalos=False, scatter=None):

        if lcenModelFile is None:
            raise(ValueError('rdel model must define lcenModelFile'))

        if scatter is None:
            raise(ValueError('rdel model must define scatter'))

        if rdelModelFile is None:
            raise(ValueError('rdel model must define rdelModelFile'))

        self.nbody = nbody
        self.luminosityFunction = lf
        self.rdelModelFile = rdelModelFile
        self.lcenModelFile = lcenModelFile
        self.lcenMassMin = lcenMassMin
        self.useSubhalos = useSubhalos
        self.scatter = float(scatter)

        if isinstance(self.lcenMassMin, str):
            self.lcenMassMin = [float(self.lcenMassMin)]
        else:
            self.lcenMassMin = [float(lcm) for lcm in self.lcenMassMin]

        self.loadModelFile()

    def loadModelFile(self):
        """Load the rdel and lcen model files and parse them
        into a format usable by the rest of the code

        Returns
        -------
        None

        """

        mdtype = np.dtype([('param', 'S10'), ('value', np.float)])
        model = np.loadtxt(self.rdelModelFile, dtype=mdtype)

        idx = model['param'].argsort()
        model = model[idx]

        self.params = {}

        self.params['muc'] = model['value'][:15]
        self.params['sigmac'] = model['value'][15:30]
        self.params['muf'] = model['value'][30:45]
        self.params['sigmaf'] = model['value'][45:60]
        self.params['p'] = model['value'][60:75]

        self.lcenModel = fitsio.read(self.lcenModelFile)
        self.lcenModel['Mc'] = 10**self.lcenModel['Mc']

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

        p[p < 0] = 0.
        p[p > 1] = 1.
        muf[muf < 0] = 0.
        sigmac[sigmac < 0] = 0.0001
        sigmaf[sigmaf < 0] = 0.0001

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

    def sampleDensity(self, domain, z, mag, dz=0.005, dm=0.1,
                      n_dens_bins=1e5):
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
        zbins = np.arange(domain.zmin, domain.zmax + dz, dz)
        zmean = zbins[1:] + zbins[:-1]

        magbins = np.arange(np.min(mag), np.max(mag) + dm, dm)
        magmean = (magbins[1:] + magbins[:-1]) / 2

        nzbins = zmean.size
        nmagbins = magmean.size

        # sort galaxies by redshift
        zidx = z.argsort()
        z = z[zidx]
        mag = mag[zidx]

        deltabins = np.linspace(0.01, 15, n_dens_bins + 1)
        deltamean = (deltabins[1:] + deltabins[:-1]) / 2

        density = np.zeros(n_gal)
        count = 0

        for i in range(nzbins):
            zlidx = z.searchsorted(zbins[i])
            zhidx = z.searchsorted(zbins[i + 1])

            midx = mag[zlidx:zhidx].argsort()
            z[zlidx:zhidx] = z[zlidx:zhidx][midx]
            mag[zlidx:zhidx] = mag[zlidx:zhidx][midx]
            mi = mag[zlidx:zhidx]

            for j in range(nmagbins):
                mlidx = mi.searchsorted(magbins[j])
                mhidx = mi.searchsorted(magbins[j + 1])

                nij = mhidx - mlidx

                cdf_r = self.pofR(deltamean, zmean[i], magmean[j])

                rands = np.random.uniform(size=nij)
                density[count: count +
                        nij] = deltamean[cdf_r.searchsorted(rands) - 1]
                count += nij

        return density, z, mag


    def sampleDensitySnap(self, domain, mag, dz=0.005, dm=0.1,
                          n_dens_bins=1e5):
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

        n_gal = mag.size

        magbins = np.arange(np.min(mag), np.max(mag) + dm, dm)
        magmean = (magbins[1:] + magbins[:-1]) / 2

        nmagbins = magmean.size

        # sort galaxies by redshift
        mag = mag

        deltabins = np.linspace(0.01, 15, n_dens_bins + 1)
        deltamean = (deltabins[1:] + deltabins[:-1]) / 2

        density = np.zeros(n_gal)
        count = 0

        midx = mag.argsort()
        mag = mag[midx]
        mi = mag

        zmean = self.nbody.domain.zmean

        if zmean < 0.001:
            zmean = 0.001

        for j in range(nmagbins):
            mlidx = mi.searchsorted(magbins[j])
            mhidx = mi.searchsorted(magbins[j + 1])

            nij = mhidx - mlidx

            cdf_r = self.pofR(deltamean, zmean, magmean[j])

            rands = np.random.uniform(size=nij)
            density[count: count +
                    nij] = deltamean[cdf_r.searchsorted(rands) - 1]
            count += nij

        return density, mag


class ColorModel(object):

    def __init__(self, nbody, trainingSetFile=None, redFractionModelFile=None,
                 filters=None, band_shift=0.1, use_redfraction=True,
                 dm_rank=0.1, ds=0.05, dm_sed=0.1, rf_z=None, rf_m=None,
                 rf_zm=None, rf_b=None, Q=0.0, no_colors=False, **kwargs):

        if redFractionModelFile is None:
            raise(ValueError('ColorModel must define redFractionModelFile'))

        if trainingSetFile is None:
            raise(ValueError('ColorModel must define trainingSetFile'))

        if filters is None:
            raise(ValueError('ColorModel must define filters'))

        self.nbody = nbody
        self.redFractionModelFile = redFractionModelFile
        self.trainingSetFile = trainingSetFile
        self.filters = filters
        self.band_shift = band_shift
        self.use_redfraction = bool(use_redfraction)
        self.dm_rank = float(dm_rank)
        self.ds = float(ds)
        self.dm_sed = float(dm_sed)
        self.c = 3e5
        self.rf_m = rf_m
        self.rf_z = rf_z
        self.rf_zm = rf_zm
        self.rf_b = rf_b
        self.Q = Q
        self.no_colors = no_colors

        if isinstance(self.band_shift, str) | isinstance(self.band_shift, float):
            self.band_shift = [float(self.band_shift)]
        else:
            self.band_shift = [float(bs) for bs in self.band_shift]

        if not self.no_colors:
            self.loadModel()

    def loadModel(self):
        """Load color training model information.

        Returns
        -------
        None

        """

        self.trainingSet = fitsio.read(self.trainingSetFile)

        mdtype = np.dtype([('param', 'S10'), ('value', np.float)])
        model = np.loadtxt(self.redFractionModelFile, dtype=mdtype)

        idx = model['param'].argsort()
        model = model[idx]

        self.redFractionParams = model['value']

    def makeVandermondeRF(self, z, mag, bmlim, fmlim, mag_ref):
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

        x = np.meshgrid(z, mag)

        zv = 1 / (x[0].flatten() + 1) - 0.47
        mv = x[1].flatten()
        mv = mv - mag_ref
        mv[mv < bright_mag_lim] = bright_mag_lim
        mv[mv > faint_mag_lim] = faint_mag_lim

        o = np.ones(len(zv))

        # construct vandermonde matrix
        xvec = np.array([o, mv, mv * zv, mv * zv**2, mv**2,
                         mv**2 * zv, mv**3, zv,
                         zv**2, zv**3])

        return xvec

    def computeRedFraction(self, z, mag, dz=0.01, dm=0.1, bmlim=-22.,
                           fmlim=-18., mag_ref=-20):
        rf = np.zeros(z.size)

        zbins = np.arange(np.min(z), np.max(z) + dz, dz)
        magbins = np.arange(np.min(mag), np.max(mag) + dm, dm)

        nzbins = zbins.size - 1
        nmagbins = magbins.size - 1

        zmean = (zbins[1:] + zbins[:-1]) / 2
        magmean = (magbins[1:] + magbins[:-1]) / 2

        xvec = self.makeVandermondeRF(zmean, magmean, bmlim, fmlim, mag_ref)
        zm = 1/(xvec[7,:] + 0.47) - 1

        # calculate red fraction for mag, z bins
        rfgrid = np.dot(xvec.T, self.redFractionParams)
        if (self.rf_m is not None) & (self.rf_z is not None) & (self.rf_b is not None) & (self.rf_zm is not None):
            rfgrid *= (self.rf_b + self.rf_m * xvec[1,:] + self.rf_z * zm + self.rf_zm * zm * xvec[1,:])
        elif (self.rf_m is not None) & (self.rf_b is not None):
            rfgrid *= ((zmean * self.rf_m) + self.rf_b)
        elif (self.rf_b is not None):
            rfgrid *= self.rf_b

        rfgrid = rfgrid.reshape(nmagbins, nzbins)

        rfgrid[rfgrid > 1] = 1.
        rfgrid[rfgrid < 0] = 0

        # get red fraction for each galaxy, idx keeps track of
        # sorted galaxy positions
        idx = np.argsort(z)
        z = z[idx]
        mag = mag[idx]

        for i in range(nzbins):
            zlidx = z.searchsorted(zbins[i])
            zhidx = z.searchsorted(zbins[i + 1])

            midx = np.argsort(mag[zlidx:zhidx])
            mag[zlidx:zhidx] = mag[zlidx:zhidx][midx]
            idx[zlidx:zhidx] = idx[zlidx:zhidx][midx]

            for j in range(nmagbins):

                mlidx = mag[zlidx:zhidx].searchsorted(magbins[j])
                mhidx = mag[zlidx:zhidx].searchsorted(magbins[j + 1])

                rf[zlidx + mlidx: zlidx + mhidx] = rfgrid[j][i]

        # reorder red fractions to match order of inputs
        temp = np.zeros_like(rf)
        temp[idx] = rf
        rf = temp

        return rf

    def rankSigma5(self, z, magnitude, sigma5, zwindow, magwindow):

        dsigma5 = np.max(sigma5) - np.min(sigma5)
        ranksigma5 = np.zeros(len(z))

        pos = np.zeros((len(z), 3))
        pos[:, 0] = z
        pos[:, 1] = magnitude
        pos[:, 2] = sigma5

        max_distances = np.array([zwindow, magwindow, dsigma5])
        neg_max_distances = -1.0 * max_distances

        tree_dsidx = np.random.choice(np.arange(len(z)), size=len(z) // 10)
        tree_pos = pos[tree_dsidx]

        with fast3tree(tree_pos) as tree:

            for i, p in enumerate(pos):

                tpos = tree.query_box(
                    p + neg_max_distances, p + max_distances, output='pos')
                tpos = tpos[:, 2]
                tpos.sort()
                ranksigma5[i] = tpos.searchsorted(pos[i, 2]) / (len(tpos) + 1)

        return ranksigma5

    def computeSigma5(self, z, mag, pos_gals, dt=1.5):

        pos_bright_gals = pos_gals[mag < -19.8]

        max_distances = np.array([dt, dt, 1000])
        neg_max_distances = -1.0 * max_distances

        sigma5 = np.zeros(len(pos_gals))

        with fast3tree(pos_bright_gals) as tree:

            for i, p in enumerate(pos_gals):

                tpos = tree.query_box(
                    p + neg_max_distances, p + max_distances, output='pos')
                dtheta = np.abs(tpos - p)
                dtheta = 2 * np.arcsin(np.sqrt(np.sin(dtheta[:, 0] / 2)**2 + np.cos(
                    p[0] * np.cos(tpos[:, 0])) * np.sin(dtheta[:, 1] / 2)**2))
                dtheta.sort()
                try:
                    sigma5[i] = dtheta[4]
                except IndexError as e:
                    sigma5[i] = -1

        z_a = copy(z)
        z_a[z_a < 1e-6] = 1e-6
        try:
            da = self.nbody.cosmo.angularDiameterDistance(z_a)
        except RuntimeError:
            print(np.min(z_a))
            print(np.max(z_a))
            print(np.isfinite(z_a).all())
            print(z_a[~np.isfinite(z_a)])
            print(self.nbody.domain.zmin, self.nbody.domain.zmax, self.nbody.domain.nest)
            raise RuntimeError

        sigma5 = sigma5 * self.nbody.cosmo.angularDiameterDistance(z_a)

        return sigma5

    def computeRankSigma5(self, z, mag, pos_gals):
        """Calculate the ranked \Sigma_5 values for galaxies,
        where \Sigma_5 is the projected radius to the 5th nearest neighbor,
        and rank \Sigma_5 is the rank of \Sigma_5 in r-band magnitude and
        redshift bins.

        Parameters
        ----------
        z : np.array
            Redshifts of the galaxies (with RSD).
        mag : np.array
            Absolute r-band magnitudes of galaxies.
        pos_gals : np.array
            Angular/redshift space coordinates. Should be (N,3) shape, with
            first column being theta, second being phi (in radians) and third
            being z_rsd.
        Returns
        -------
        type
            Description of returned object.

        """

        start = time()
        dtheta = 100. / self.nbody.cosmo.angularDiameterDistance(self.nbody.domain.zmin)
        if dtheta > 1.5:
            dtheta = 1.5
        sigma5 = self.computeSigma5(z, mag, pos_gals, dt=dtheta)
        end = time()
        print('[{}] Finished computing sigma5. Took {}s'.format(self.nbody.domain.rank, end - start))

        start = time()
        ranksigma5 = self.rankSigma5(z, mag, sigma5, 0.01, self.dm_rank)
        end = time()
        print('[{}] Finished computing rank sigma5. Took {}s'.format(self.nbody.domain.rank, end - start))

        return sigma5, ranksigma5

    def matchTrainingSet(self, mag, ranksigma5, redfraction, dm=0.1, ds=0.05):

        mag_train = self.trainingSet['ABSMAG'][:, 2]
        ranksigma5_train = self.trainingSet['PSIGMA5']
        isred_train = self.trainingSet['ISRED']

        pos = np.zeros((mag.size, 3))
        pos_train = np.zeros((mag_train.size, 3))

        n_gal = mag.size
        rand = np.random.rand(n_gal)

        pos[:, 0] = mag
        pos[:, 1] = ranksigma5
        pos[:, 2] = redfraction

        pos[pos[:, 0] < np.min(mag_train), 0] = np.min(mag_train)
        pos[pos[:, 0] > np.max(mag_train), 0] = np.max(mag_train)

        pos_train[:, 0] = mag_train
        pos_train[:, 1] = ranksigma5_train
        pos_train[:, 2] = isred_train

        # max distance in isred direction large enough to select all
        # make search distance in mag direction larger as we go fainter
        # as there are fewer galaxies in the training set there

        def max_distances(m): return np.array([np.min([np.max([np.abs((22.5 + m)) * dm, 0.1]), 5]), ds, 1.1])

        def neg_max_distances(m): return -1. * \
            np.array([np.min([np.max([np.abs((22.5 + m)) * dm, 0.1]), 5]), ds, 1.1])

        sed_idx = np.zeros(n_gal, dtype=np.int)
        bad = np.zeros(n_gal, dtype=np.bool)

        with fast3tree(pos_train) as tree:

            for i, p in enumerate(pos):

                idx, tpos = tree.query_box(
                    p + neg_max_distances(p[0]), p + max_distances(p[0]), output='both')
                rf = np.sum(tpos[:, 2]) / len(tpos)
                isred = rand[i] < (rf * redfraction[i])
                idx = idx[tpos[:, 2] == int(isred)]
                tpos = tpos[tpos[:, 2] == int(isred)]
                tpos -= p
                dt = np.abs(np.sum(tpos**2, axis=1))
                try:
                    sed_idx[i] = idx[np.argmin(dt)]
                except Exception as e:
                    bad[i] = True

            isbad = np.where(bad)[0]
            print('Number of bad SED assignments: {}'.format(len(isbad)))

            def max_distances(m): return np.array(
                [np.max([10 * (22.5 + m)**2 * dm, dm]), ds, 1.1])

            def neg_max_distances(m): return -1. * \
                np.array([np.max([10 * (22.5 + m)**2 * dm, dm]), ds, 1.1])

            for i, p in enumerate(pos[bad]):
                idx, tpos = tree.query_box(
                    p + neg_max_distances(p[0]), p + max_distances(p[0]), output='both')

                rf = np.sum(tpos[:, 2]) / len(tpos)
                isred = rand[i] < (rf * redfraction[i])
                idx = idx[tpos[:, 2] == int(isred)]
                tpos = tpos[tpos[:, 2] == int(isred)]
                tpos -= p
                dt = np.abs(np.sum(tpos**2, axis=1))

                try:
                    sed_idx[isbad[i]] = idx[np.argmin(dt)]
                except Exception as e:
                    bad[i] = True

        return sed_idx, bad

    def computeMagnitudes(self, mag, z, coeffs, filters):
        """Compute observed and absolute magnitudes in the
        given filters for galaxies.

        Parameters
        ----------
        mag : np.array
            SDSS z=0.1 frame r-band absolute magnitudes.
        z : np.array
            Redshifts of galaxies.
        coeffs : np.array
            Kcorrect coefficients of all galaxies.
        filters : list
            List of filter files to calculate magnitudes for.

        Returns
        -------
        omag : np.array
            Array of observed magnitudes of shape (n_gal, nk) where
            nk is number of filters observed.
        amag : np.array
            Array of absolute magnitudes of shape (n_gal, nk) where
            nk is number of filters observed.

        """

        kcorr = KCorrect()

        # calculate sdss r band absolute magnitude in order
        # to renormalize the kcorrect coefficients to give
        # the correct absolute magnitudes for the simulated
        # galaxies
        sdss_r_name = ['sdss/sdss_r0.par']
        filter_lambda, filter_pass = kcorr.load_filters(sdss_r_name)

        rmatrix = kcorr.k_projection_table(filter_pass, filter_lambda, [0.1])
        rmatrix0 = kcorr.k_projection_table(filter_pass, filter_lambda, [0.0])

        amag = k_reconstruct_maggies(rmatrix.astype(np.float64),
                                     coeffs.astype(np.float64),
                                     np.zeros_like(z).astype(np.float64),
                                     kcorr.zvals.astype(np.float64))

        omag = k_reconstruct_maggies(rmatrix0.astype(np.float64),
                                     coeffs.astype(np.float64),
                                     z.astype(np.float64),
                                     kcorr.zvals.astype(np.float64))

        kc = 2.5 * np.log10(amag / omag)

        a = 1 / (1 + z)
        amax = 1 / (1 + 1e-7)
        a[a > amax] = amax

        omag = -2.5 * np.log10(omag)
        dm = self.nbody.cosmo.distanceModulus(1 / a - 1)
        amag = omag - dm.reshape(-1, 1) - kc

        # renormalize coeffs
        coeffs *= 10 ** ((mag.reshape(-1, 1) - amag) / -2.5)

        # Calculate observed and absolute magnitudes magnitudes
        filter_lambda, filter_pass = kcorr.load_filters(filters)

        rmatrix0 = kcorr.k_projection_table(filter_pass, filter_lambda,
                                            [0.0] * len(filters))
        rmatrix = kcorr.k_projection_table(filter_pass, filter_lambda,
                                           self.band_shift)
        amag = k_reconstruct_maggies(rmatrix,
                                     coeffs.astype(np.float64),
                                     np.zeros_like(z).astype(np.float64),
                                     kcorr.zvals)
        omag = k_reconstruct_maggies(rmatrix0,
                                     coeffs.astype(np.float64),
                                     z.astype(np.float64),
                                     kcorr.zvals)

        kc = 2.5 * np.log10(amag / omag)
        omag = -2.5 * np.log10(omag)
        amag = omag - dm.reshape(-1, 1) - kc

        return omag, amag

    def assignSEDs(self, pos, mag, z, z_rsd):

        start = time()

        theta, phi = hp.vec2ang(pos)
        rspos = np.vstack([theta, phi, self.c * z_rsd]).T
        print('[{}] checking redshifts... max(z), min(z), max(z_rsd), min(z_rsd): {}, {}, {}, {}'.format(self.nbody.domain.rank, np.max(z), np.min(z),np.max(z_rsd),np.min(z_rsd)))
        sigma5, ranksigma5 = self.computeRankSigma5(z_rsd, mag, rspos)
        end = time()
        print('[{}] Finished computing sigma5 and rank sigma5. Took {}s'.format(self.nbody.domain.rank, end - start))
        sys.stdout.flush()

        start = time()
        if self.use_redfraction:
            mag_evol = mag + self.Q * (1 / (1 + z) - 1 / 1.1)
            redfraction = self.computeRedFraction(z, mag_evol)
        else:
            redfraction = np.ones_like(z)

        sed_idx, bad = self.matchTrainingSet(mag, ranksigma5, redfraction, self.dm_sed, self.ds)
        coeffs = self.trainingSet[sed_idx]['COEFFS']
        end = time()

        print('[{}] Finished assigning SEDs. Took {}s'.format(self.nbody.domain.rank, end - start))
        sys.stdout.flush()

        # make sure we don't have any negative redshifts
        z_a = copy(z_rsd)
        z_a[z_a < 1e-6] = 1e-6
        mag = mag_evol

        start = time()
        omag, amag = self.computeMagnitudes(mag, z_a, coeffs, self.filters)
        end = time()

        print('[{}] Finished compiuting magnitudes from SEDs. Took {}s'.format(self.nbody.domain.rank, end - start))
        sys.stdout.flush()

        return sigma5, ranksigma5, redfraction, sed_idx, omag, amag, mag

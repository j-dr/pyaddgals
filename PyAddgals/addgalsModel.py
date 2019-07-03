from __future__ import print_function, division
from scipy.special import erf
from scipy.optimize import minimize
from numba import jit, boolean
from time import time
import numpy as np
import fitsio
import george
from george import kernels
from sklearn import preprocessing

import sys

from .galaxyModel import GalaxyModel
from .colorModel import ColorModel
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
               params, scatter, dMr=0.015, dz=0.02):

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

        zmin = z_halo[i] - dz
        zmax = z_halo[i] + dz

        pidx = np.searchsorted(density, density_halo[i])
        pidx -= 1
        pi = 0

        halo_assigned = False

        while ((not halo_assigned) & (pi < max_search_count)):

            if ((pidx - pi) >= 0) & ((pidx - pi) < n_gal):

                if ((not assigned[pidx - pi]) & (magmin < magnitude[pidx - pi]) &
                        (magnitude[pidx - pi] < magmax) &
                        (zmin < redshift[pidx - pi]) &
                        (redshift[pidx - pi] < zmax)):
                    assigned[pidx - pi] = True
                    halo_assigned = True
                    pi += 1
                    continue

            if ((pidx + pi) < n_gal) & ((pidx + pi) >= 0):

                if ((not assigned[pidx + pi]) & (magmin < magnitude[pidx + pi]) &
                        (magnitude[pidx + pi] < magmax) &
                        (zmin < redshift[pidx + pi]) &
                        (redshift[pidx + pi] < zmax)):
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

            zmin = z_halo[i] - dz
            zmax = z_halo[i] + dz

            while ((not halo_assigned) & (pi < max_search_count)):

                if ((pidx - pi) >= 0) & ((pidx - pi) < n_gal):

                    if ((not assigned[pidx - pi]) & (magmin < magnitude[pidx - pi]) &
                            (magnitude[pidx - pi] < magmax) &
                            (zmin < redshift[pidx - pi]) &
                            (redshift[pidx - pi] < zmax)):
                        assigned[pidx - pi] = True
                        halo_assigned = True
                        pi += 1
                        continue

                if ((pidx + pi) < n_gal) & ((pidx + pi) >= 0):

                    if ((not assigned[pidx + pi]) & (magmin < magnitude[pidx + pi]) &
                            (magnitude[pidx + pi] < magmax) &
                            (zmin < redshift[pidx + pi]) &
                            (redshift[pidx + pi] < zmax)):
                        assigned[pidx + pi] = True
                        halo_assigned = True
                        pi += 1
                        continue

                pi += 1

    return assigned, mr0, bad


@jit(nopython=True)
def assignLcenNodens(redshift, magnitude, density, mass_halo, density_halo, z_halo,
                     params, scatter, dMr=0.015, dz=0.01):

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

        zmin = z_halo[i] - dz
        zmax = z_halo[i] + dz

        pidx = np.searchsorted(magnitude, mr0[i])
        pidx -= 1
        pi = 0

        halo_assigned = False

        while ((not halo_assigned) & (pi < max_search_count)):

            if ((pidx - pi) >= 0) & ((pidx - pi) < n_gal):

                if ((not assigned[pidx - pi]) & (zmin < redshift[pidx - pi]) &
                        (redshift[pidx - pi] < zmax)):
                    assigned[pidx - pi] = True
                    halo_assigned = True
                    pi += 1
                    continue

            if ((pidx + pi) < n_gal) & ((pidx + pi) >= 0):

                if ((not assigned[pidx + pi]) & (zmin < redshift[pidx + pi]) &
                        (redshift[pidx + pi] < zmax)):
                    assigned[pidx + pi] = True
                    halo_assigned = True
                    pi += 1
                    continue

            pi += 1

        # if not assigned with fiducial magniude window, make larger
        if not halo_assigned:
            bad[i] = True
            pi = 0
            zmin = z_halo[i] - 3 * dz
            zmax = z_halo[i] + 3 * dz

            while ((not halo_assigned) & (pi < max_search_count)):

                if ((pidx - pi) >= 0) & ((pidx - pi) < n_gal):

                    if ((not assigned[pidx - pi]) & (zmin < redshift[pidx - pi]) &
                       (redshift[pidx - pi] < zmax)):
                        assigned[pidx - pi] = True
                        halo_assigned = True
                        pi += 1
                        continue

                if ((pidx + pi) < n_gal) & ((pidx + pi) >= 0):

                    if ((not assigned[pidx + pi]) & (zmin < redshift[pidx + pi]) &
                       (redshift[pidx + pi] < zmax)):
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

        self.rdelModel = RdelModel(
            self.nbody, self.luminosityFunction, **rdelModelConfig)
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

        zidx = z.argsort()
        z = z[zidx]
        mag = mag[zidx]
        del zidx

        if domain.fmt == 'Snapshot':
            density, mag = self.rdelModel.sampleDensitySnap(domain, mag)
        else:
            density, z, mag = self.rdelModel.sampleDensity(domain, z, mag)

        end = time()

        print('[{}] Finished drawing mag, z, dens. {} galaxies in domain, took {}s'.format(
            self.nbody.domain.rank, len(z), end - start))
        sys.stdout.flush()

        # sort galaxies by magnitude
        idx = mag.argsort()
        z = z[idx]
        mag = mag[idx]
        density = density[idx]
        del idx

        start = time()
        mag_cen, assigned, bad_cen = self.assignHalos(z, mag, density)
        end = time()

        print('[{}] Finished assigning galaxies to halos. Took {}s'.format(
            self.nbody.domain.rank, end - start))
        sys.stdout.flush()

        start = time()
        pos, vel, z, density, mag, rhalo, halorad, haloid, halomass, bad = self.assignParticles(
            z[~assigned], mag[~assigned], density[~assigned])
        end = time()

        print('[{}] Finished assigning galaxies to particles. Took {}s'.format(
            self.nbody.domain.rank, end - start))
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
        halorad = np.hstack(
            [self.nbody.haloCatalog.catalog['radius'], halorad])
        rhalo = np.hstack([np.zeros(n_halo), rhalo])
        central = np.zeros(rhalo.size)
        central[:n_halo] = 1
        haloid = np.hstack([self.nbody.haloCatalog.catalog['id'], haloid])
        z_rsd = z + np.sum(pos * vel, axis=1) / \
            np.sqrt(np.sum(pos**2, axis=1)) / 299792.458
        bad = np.hstack([bad_cen, bad])

        # done with halo catalog now
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
            sed_idx, omag, amag, mag_evol = self.colorModel.assignSEDs(
                pos, mag, z, z_rsd)

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

        log_comoving_size, angular_size, epsilon = self.shapeModel.sampleShapes(
            self.nbody.galaxyCatalog.catalog)

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

        assigned, lcen, bad = assignLcenNodens(z, mag, dens, mass_halo, density_halo,
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

        print('[{}] number of bad assignments: {}'.format(
            self.nbody.domain.rank, np.sum(bad)))
        sys.stdout.flush()

        return pos, vel, z_asn, density_asn, magnitude, rhalo, haloradius, haloid, halomass, bad


class RdelModel(object):

    def __init__(self, nbody, lf, rdelModelFile=None, lcenModelFile=None,
                 lcenMassMin=None, useSubhalos=False, scatter=None,
                 gaussian_process=False):

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
        self.gaussian_process = gaussian_process

        if isinstance(self.lcenMassMin, str):
            self.lcenMassMin = [float(self.lcenMassMin)]
        else:
            self.lcenMassMin = [float(lcm) for lcm in self.lcenMassMin]

        self.loadModelFile()

    def clean_data(self, X, y, yerr):
        medy = np.median(y)

        idx = (np.isfinite(y) & (np.abs((y - medy)) < 100 * np.abs(medy)) &
               (yerr > 0) & (yerr == yerr) & (np.abs(yerr) != np.inf))
        return X[idx], y[idx], yerr[idx]

    def fit_gp(self, X, y, ye):
        Xc, yc, yerrc = self.clean_data(X, y, ye)
        scaler = preprocessing.StandardScaler().fit(Xc)
        scaler_y = preprocessing.StandardScaler().fit(yc.reshape(-1, 1))
        nX = scaler.transform(Xc)
        ny = scaler_y.transform(yc.reshape(-1, 1))
        nye = yerrc * scaler_y.scale_

        kernel = np.var(ny.flatten()) * kernels.ExpSquaredKernel(0.5, ndim=Xc.shape[1])
        gp = george.GP(kernel, fit_white_noise=True)
        gp.compute(nX, np.sqrt(nye).flatten())

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(ny.flatten())

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(ny.flatten())

        result = minimize(neg_ln_like, [1., 1], jac=grad_neg_ln_like, method="L-BFGS-B")
        print('[{}]: fit {}'.format(self.nbody.domain.rank, result))
        gp.set_parameter_vector(result.x)

        return gp

    def pred_gp(self, gp, y, X, px, ye):
        Xc, yc, _ = self.clean_data(X, y, ye)
        scaler = preprocessing.StandardScaler().fit(Xc)
        scaler_y = preprocessing.StandardScaler().fit(yc.reshape(-1, 1))

        ny = scaler_y.transform(yc.reshape(-1, 1))
        npx = scaler.transform(px)
        pred, pred_var = gp.predict(ny.flatten(), npx, return_var=True)

        return scaler_y.inverse_transform(pred)

    def loadModelFile(self):
        """Load the rdel and lcen model files and parse them
        into a format usable by the rest of the code

        Returns
        -------
        None

        """

        if not self.gaussian_process:
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
        else:

            rdel_params = fitsio.read(self.rdelModelFile, ext=0)
            rdel_param_errors = fitsio.read(self.rdelModelFile, ext=1)
            self.X = fitsio.read(self.rdelModelFile, ext=2)
            self.T = fitsio.read(self.rdelModelFile, ext=3)

            self.rdel_params = np.dot(np.linalg.inv(self.T), rdel_params.T)
            self.rdel_param_errors = np.abs(np.dot(np.linalg.inv(self.T), rdel_param_errors.T))

            gpp = self.fit_gp(self.X, self.rdel_params[0, :],
                              self.rdel_param_errors[0, :])
            gpmuc = self.fit_gp(self.X, self.rdel_params[1, :],
                                self.rdel_param_errors[1, :])
            gpsigmac = self.fit_gp(self.X, self.rdel_params[2, :],
                                   self.rdel_param_errors[2, :])
            gpmuf = self.fit_gp(self.X, self.rdel_params[3, :],
                                self.rdel_param_errors[3, :])
            gpsigmaf = self.fit_gp(self.X, self.rdel_params[4, :],
                                   self.rdel_param_errors[4, :])

            self.gp_model = [gpp, gpmuc, gpsigmac, gpmuf, gpsigmaf]

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

        if not self.gaussian_process:
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
        else:
            zp = z
            mp = mag

            if mp > magfaint:
                mp = magfaint
            elif mp < magbright:
                mp = magbright

            if zp > np.max(self.X[:, 0]):
                zp = np.max(self.X[:, 0])
            elif zp < np.min(self.X[:, 0]):
                zp = np.min(self.X[:, 0])

            xp = np.atleast_2d([zp, mp])
            p = self.pred_gp(self.gp_model[0], self.rdel_params[0, :],
                             self.X, xp, self.rdel_param_errors[0, :])[0]
            muc = self.pred_gp(self.gp_model[1], self.rdel_params[1, :],
                               self.X, xp, self.rdel_param_errors[1, :])[0]
            sigmac = self.pred_gp(self.gp_model[2], self.rdel_params[2, :],
                                  self.X, xp, self.rdel_param_errors[2, :])[0]
            muf = self.pred_gp(self.gp_model[3], self.rdel_params[3, :],
                               self.X, xp, self.rdel_param_errors[3, :])[0]
            sigmaf = self.pred_gp(self.gp_model[4], self.rdel_params[4, :],
                                  self.X, xp, self.rdel_param_errors[4, :])[0]

            pars = [p, muc, sigmac, muf, sigmaf]
            pars = np.dot(self.T, pars)
            p, muc, sigmac, muf, sigmaf = pars

        return muc, sigmac, muf, sigmaf, p

    def pofR(self, r, z, mag, dmag=0.2):

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
                      n_dens_bins=1e5, dmcdf=0.2):
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

                cdf_r = self.pofR(deltamean, zmean[i], magmean[j], dmag=dmcdf)

                rands = np.random.uniform(size=nij)
                density[count: count +
                        nij] = deltamean[cdf_r.searchsorted(rands) - 1]
                count += nij

        return density, z, mag

    def sampleDensitySnap(self, domain, mag, dz=0.005, dm=0.1,
                          n_dens_bins=1e5, dmcdf=0.2):
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

            cdf_r = self.pofR(deltamean, zmean, magmean[j], dmag=dmcdf)

            rands = np.random.uniform(size=nij)
            density[count: count +
                    nij] = deltamean[cdf_r.searchsorted(rands) - 1]
            count += nij

        return density, mag

from __future__ import print_function, division
from fast3tree import fast3tree
from halotools.empirical_models import abunmatch
from copy import copy
from time import time

import healpy as hp
import numpy as np
import sys
import fitsio
import sys

from .kcorrect import KCorrect, k_reconstruct_maggies


class ColorModel(object):
    def __init__(
        self,
        nbody,
        trainingSetFile=None,
        redFractionModelFile=None,
        filters=None,
        band_shift=0.1,
        use_redfraction=True,
        dm_rank=0.1,
        ds=0.05,
        dm_sed=0.1,
        rf_z=None,
        rf_m=None,
        rf_zm=None,
        rf_b=None,
        Q=0.0,
        no_colors=False,
        piecewise_mag_evolution=False,
        match_magonly=False,
        third_order_mag_evolution=False,
        derived_quantities=False,
        **kwargs
    ):

        if redFractionModelFile is None:
            raise (ValueError("ColorModel must define redFractionModelFile"))

        if trainingSetFile is None:
            raise (ValueError("ColorModel must define trainingSetFile"))

        if filters is None:
            raise (ValueError("ColorModel must define filters"))

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
        self.piecewise_mag_evolution = piecewise_mag_evolution
        self.third_order_mag_evolution = third_order_mag_evolution
        self.no_colors = no_colors
        self.match_magonly = match_magonly
        self.derived_quantities = derived_quantities

        if self.match_magonly:
            self.ds = 0.01

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

        mdtype = np.dtype([("param", "S10"), ("value", np.float)])
        model = np.loadtxt(self.redFractionModelFile, dtype=mdtype)

        idx = model["param"].argsort()
        model = model[idx]

        self.redFractionParams = model["value"]

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
        xvec = np.array(
            [
                o,
                mv,
                mv * zv,
                mv * zv**2,
                mv**2,
                mv**2 * zv,
                mv**3,
                zv,
                zv**2,
                zv**3,
            ]
        )

        return xvec

    def computeRedFraction(
        self, z, mag, dz=0.01, dm=0.1, bmlim=-22.0, fmlim=-18.0, mag_ref=-20
    ):
        rf = np.zeros(z.size)

        zbins = np.arange(np.min(z), np.max(z) + dz, dz)
        magbins = np.arange(np.min(mag), np.max(mag) + dm, dm)

        nzbins = zbins.size - 1
        nmagbins = magbins.size - 1

        zmean = (zbins[1:] + zbins[:-1]) / 2
        magmean = (magbins[1:] + magbins[:-1]) / 2

        xvec = self.makeVandermondeRF(zmean, magmean, bmlim, fmlim, mag_ref)
        zm = 1 / (xvec[7, :] + 0.47) - 1

        # calculate red fraction for mag, z bins
        rfgrid = np.dot(xvec.T, self.redFractionParams)
        if (
            (self.rf_m is not None)
            & (self.rf_z is not None)
            & (self.rf_b is not None)
            & (self.rf_zm is not None)
        ):
            rfgrid *= (
                self.rf_b
                + self.rf_m * xvec[1, :]
                + self.rf_z * zm
                + self.rf_zm * zm * xvec[1, :]
            )
        elif (self.rf_m is not None) & (self.rf_b is not None):
            rfgrid *= (zmean * self.rf_m) + self.rf_b
        elif self.rf_b is not None:
            rfgrid *= self.rf_b

        rfgrid = rfgrid.reshape(nmagbins, nzbins)

        rfgrid[rfgrid > 1] = 1.0
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

                rf[zlidx + mlidx : zlidx + mhidx] = rfgrid[j][i]

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
                    p + neg_max_distances, p + max_distances, output="pos"
                )
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
                    p + neg_max_distances, p + max_distances, output="pos"
                )
                dtheta = np.abs(tpos - p)
                dtheta = 2 * np.arcsin(
                    np.sqrt(
                        np.sin(dtheta[:, 0] / 2) ** 2
                        + np.cos(p[0] * np.cos(tpos[:, 0]))
                        * np.sin(dtheta[:, 1] / 2) ** 2
                    )
                )
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
            print(
                self.nbody.domain.zmin, self.nbody.domain.zmax, self.nbody.domain.nest
            )
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
        dtheta = 100.0 / self.nbody.cosmo.angularDiameterDistance(
            self.nbody.domain.zmin
        )
        if dtheta > 1.5:
            dtheta = 1.5
        sigma5 = self.computeSigma5(z, mag, pos_gals, dt=dtheta)
        end = time()
        print(
            "[{}] Finished computing sigma5. Took {}s".format(
                self.nbody.domain.rank, end - start
            )
        )

        start = time()
        ranksigma5 = self.rankSigma5(z, mag, sigma5, 0.01, self.dm_rank)
        end = time()
        print(
            "[{}] Finished computing rank sigma5. Took {}s".format(
                self.nbody.domain.rank, end - start
            )
        )

        return sigma5, ranksigma5

    def matchTrainingSet(self, mag, ranksigma5, redfraction, dm=0.1, ds=0.05):

        mag_train = self.trainingSet["ABSMAG"][:, 2]

        if self.match_magonly:
            ranksigma5_train = np.random.rand(len(self.trainingSet["PSIGMA5"]))
        else:
            ranksigma5_train = self.trainingSet["PSIGMA5"]

        isred_train = self.trainingSet["ISRED"]

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

        def max_distances(m):
            return np.array(
                [np.min([np.max([np.abs((22.5 + m)) * dm, 0.1]), 5]), ds, 1.1]
            )

        def neg_max_distances(m):
            return -1.0 * np.array(
                [np.min([np.max([np.abs((22.5 + m)) * dm, 0.1]), 5]), ds, 1.1]
            )

        sed_idx = np.zeros(n_gal, dtype=np.int)
        bad = np.zeros(n_gal, dtype=np.bool)

        with fast3tree(pos_train) as tree:

            for i, p in enumerate(pos):

                idx, tpos = tree.query_box(
                    p + neg_max_distances(p[0]), p + max_distances(p[0]), output="both"
                )
                rf = np.sum(tpos[:, 2]) / len(tpos)
                isred = rand[i] < (rf * redfraction[i])
                idx = idx[tpos[:, 2] == int(isred)]
                tpos = tpos[tpos[:, 2] == int(isred)]
                tpos -= p

                if self.match_magonly:
                    dt = np.abs(tpos[:, 0])
                else:
                    dt = np.abs(np.sum(tpos**2, axis=1))

                try:
                    sed_idx[i] = idx[np.argmin(dt)]
                except Exception as e:
                    bad[i] = True

            isbad = np.where(bad)[0]
            print("Number of bad SED assignments: {}".format(len(isbad)))

            if self.match_magonly:

                def max_distances(m):
                    return np.array([np.max([(22.5 + m) ** 2 * dm, dm]), ds * 10, 1.1])

                def neg_max_distances(m):
                    return -1.0 * np.array(
                        [np.max([(22.5 + m) ** 2 * dm, dm]), ds * 10, 1.1]
                    )

            else:

                def max_distances(m):
                    return np.array([np.max([10 * (22.5 + m) ** 2 * dm, dm]), ds, 1.1])

                def neg_max_distances(m):
                    return -1.0 * np.array(
                        [np.max([10 * (22.5 + m) ** 2 * dm, dm]), ds, 1.1]
                    )

            for i, p in enumerate(pos[bad]):
                idx, tpos = tree.query_box(
                    p + neg_max_distances(p[0]), p + max_distances(p[0]), output="both"
                )

                rf = np.sum(tpos[:, 2]) / len(tpos)
                isred = rand[i] < (rf * redfraction[i])
                idx = idx[tpos[:, 2] == int(isred)]
                tpos = tpos[tpos[:, 2] == int(isred)]
                tpos -= p

                if self.match_magonly:
                    dt = np.abs(tpos[:, 0])
                else:
                    dt = np.abs(np.sum(tpos**2, axis=1))

                try:
                    sed_idx[isbad[i]] = idx[np.argmin(dt)]
                except Exception as e:
                    bad[i] = True

        return sed_idx, bad

    def computeMagnitudes(self, mag, z, coeffs, filters, return_coeff=False):
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
        return_coeff: bool
            If True, return rescaled kcorrect coefficients.

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
        sdss_r_name = ["sdss/sdss_r0.par"]
        filter_lambda, filter_pass = kcorr.load_filters(sdss_r_name)

        rmatrix = kcorr.k_projection_table(filter_pass, filter_lambda, [0.1])
        rmatrix0 = kcorr.k_projection_table(filter_pass, filter_lambda, [0.0])

        amag = k_reconstruct_maggies(
            rmatrix.astype(np.float64),
            coeffs.astype(np.float64),
            np.zeros_like(z).astype(np.float64),
            kcorr.zvals.astype(np.float64),
        )

        omag = k_reconstruct_maggies(
            rmatrix0.astype(np.float64),
            coeffs.astype(np.float64),
            z.astype(np.float64),
            kcorr.zvals.astype(np.float64),
        )

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
        #        print(filters)
        #        sys.stdout.flush()

        filter_lambda, filter_pass = kcorr.load_filters(filters)

        rmatrix0 = kcorr.k_projection_table(
            filter_pass, filter_lambda, [0.0] * len(filters)
        )
        rmatrix = kcorr.k_projection_table(filter_pass, filter_lambda, self.band_shift)
        amag = k_reconstruct_maggies(
            rmatrix,
            coeffs.astype(np.float64),
            np.zeros_like(z).astype(np.float64),
            kcorr.zvals,
        )
        omag = k_reconstruct_maggies(
            rmatrix0, coeffs.astype(np.float64), z.astype(np.float64), kcorr.zvals
        )

        kc = 2.5 * np.log10(amag / omag)
        omag = -2.5 * np.log10(omag)
        amag = omag - dm.reshape(-1, 1) - kc

        if return_coeff:
            return omag, amag, coeffs
        else:
            return omag, amag

    def reassign_colors_cam(
        self,
        px,
        py,
        pz,
        hpx,
        hpy,
        hpz,
        m200,
        mr,
        amag,
        mhalo=12.466,
        corr=0.749,
        alpham=0.0689,
    ):

        centrals = h[(h["HOST_HALOID"] == -1) & (h["M200B"] > 10**mhalo)]
        cpos = np.zeros((len(centrals), 3))

        pos = np.zeros((len(g), 3))
        pos[:, 0] = g["PX"]
        pos[:, 1] = g["PY"]
        pos[:, 2] = g["PZ"]

        cpos[:, 0] = centrals["PX"]
        cpos[:, 1] = centrals["PY"]
        cpos[:, 2] = centrals["PZ"]

        rhalo = np.zeros(len(pos))

        with fast3tree(cpos) as tree:
            for i in range(len(pos)):
                d = tree.query_nearest_distance(pos[i, :])
                rhalo[i] = d

        mr = copy(g["MAG_R_EVOL"])
        mr[mr < -22] = -22
        mr[mr > -18] = -18
        gr = amag[:, 0] - amag[:, 1]

        idx = np.argsort(rhalo)
        rhalo_sorted = rhalo[idx]
        rank_rhalo = np.arange(len(rhalo)) / len(rhalo)
        corr_coeff = corr * (mr + 22) ** (alpham)
        corr_coeff[corr_coeff > 1] = 1.0
        noisy_rank_rhalo = abunmatch.noisy_percentile(rank_rhalo, corr_coeff)

        g = g[idx]
        gr = g["AMAG"][:, 0] - g["AMAG"][:, 1]

        idx_swap = abunmatch.conditional_abunmatch(
            g["MAG_R_EVOL"],
            noisy_rank_rhalo,
            g["MAG_R_EVOL"],
            -gr,
            99,
            return_indexes=True,
        )
        temp_sedid = g["SEDID"][idx_swap]

        return temp_sedid

    def poly(self, z, m0, m1, m2, c, zhi, zlo):
        zlidx = z < zlo
        zr = z - zlo
        zhi = zhi - zlo
        zidx = zr > zhi
        ar = 1 / (zr + 1) - 1
        ahi = 1 / (zhi + 1) - 1

        dm = c + m0 * ar + m1 * ar**2 + m2 * ar**3
        dm[zlidx] = 0
        dm[zidx] = (
            c
            + m0 * (ahi)
            + m1 * (ahi) ** 2
            + m2 * (ahi) ** 3
            + (ar[zidx] - ahi) * (m0 + 2 * m1 * (ahi) + 3 * m2 * (ahi) ** 2)
        )

        return dm

    def assignSEDs(self, pos, mag, z, z_rsd):

        start = time()

        theta, phi = hp.vec2ang(pos)
        rspos = np.vstack([theta, phi, self.c * z_rsd]).T
        print(
            "[{}] checking redshifts... max(z), min(z), max(z_rsd), min(z_rsd): {}, {}, {}, {}".format(
                self.nbody.domain.rank,
                np.max(z),
                np.min(z),
                np.max(z_rsd),
                np.min(z_rsd),
            )
        )

        if not self.match_magonly:
            sigma5, ranksigma5 = self.computeRankSigma5(z_rsd, mag, rspos)
            end = time()
            print(
                "[{}] Finished computing sigma5 and rank sigma5. Took {}s".format(
                    self.nbody.domain.rank, end - start
                )
            )
            sys.stdout.flush()
        else:
            print("[{}] Only matching SEDs on magnitude".format(self.nbody.domain.rank))
            ranksigma5 = np.random.rand(len(z))
            sigma5 = ranksigma5

        start = time()
        if self.piecewise_mag_evolution:
            zidx = z > self.Q[2]
            mag_evol = mag + self.Q[0] * (1.0 / (1 + z) - 1.0 / (1 + 0.1)) + self.Q[3]
            mag_evol[zidx] = (
                mag[zidx]
                + self.Q[1] * (1.0 / (1 + z[zidx]) - 1.0 / (1 + 0.1))
                + self.Q[0] * (1.0 / (1 + self.Q[2]) - 1.0 / (1 + 0.1))
            ) + self.Q[3]
        elif self.third_order_mag_evolution:
            mag_evol = mag + self.poly(z, *self.Q)
        else:
            mag_evol = mag + self.Q * (1 / (1 + z) - 1 / 1.1)

        if self.use_redfraction:
            redfraction = self.computeRedFraction(z, mag_evol)
        else:
            redfraction = np.ones_like(z)

        sed_idx, bad = self.matchTrainingSet(
            mag, ranksigma5, redfraction, self.dm_sed, self.ds
        )
        coeffs = self.trainingSet[sed_idx]["COEFFS"]
        end = time()

        print(
            "[{}] Finished assigning SEDs. Took {}s".format(
                self.nbody.domain.rank, end - start
            )
        )
        sys.stdout.flush()

        # make sure we don't have any negative redshifts
        z_a = copy(z_rsd)
        z_a[z_a < 1e-6] = 1e-6
        mag = mag_evol

        start = time()
        if self.derived_quantities:
            omag, amag, coeffs = self.computeMagnitudes(
                mag, z_a, coeffs, self.filters, return_coeff=True
            )

        else:
            omag, amag = self.computeMagnitudes(
                mag, z_a, coeffs, self.filters, return_coeff=False
            )

        end = time()

        print(
            "[{}] Finished compiuting magnitudes from SEDs. Took {}s".format(
                self.nbody.domain.rank, end - start
            )
        )
        sys.stdout.flush()

        if not self.derived_quantities:
            return sigma5, ranksigma5, redfraction, sed_idx, omag, amag, mag
        else:
            kcorr = KCorrect()

            sfr, met, smass = kcorr.get_derived_quantities(
                self.nbody.cosmo, coeffs, z_a
            )
            return (
                sigma5,
                ranksigma5,
                redfraction,
                sed_idx,
                omag,
                amag,
                mag,
                sfr,
                met,
                smass,
            )

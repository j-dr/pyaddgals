from __future__ import print_function, division
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
from copy import copy
import numpy as np


class LuminosityFunction(object):

    def __init__(self, cosmo, params=None, name=None, magmin=25., magmax=10.,
                 m_min_of_z_snap=-18, m_max_of_z_snap=-25, **kwargs):
        """Initialize LuminosityFunction object.

        Parameters
        ----------
        params : array
            Array of parameters for the LF
        name : str
            Name of the LF
        magmin : float
            Faintest magniutude that we want to populate the simulation
            to.

        Returns
        -------
        type
            Description of returned object.

        """

        self.name = name
        self.params = params
        self.cosmo = cosmo
        self.magmin = float(magmin)
        self.magmax = float(magmax)

        self.m_min_of_z_snap = m_min_of_z_snap
        self.m_max_of_z_snap = m_max_of_z_snap

    def genLuminosityFunction(self, lums, zs):
        """Compute the luminosity function at a range of redshifts.

        Parameters
        ----------
        lums : np.array
            Array of luminosities at which to calculate the LF
        zs : np.array
            Array of redshifts at which to calculate the LF

        Returns
        -------
        None

        """

        self.lf = np.zeros((len(lums), len(zs)))

        for i, z in enumerate(zs):
            zp = self.evolveParams(z)
            self.lf[:, i] = self.numberDensity(zp, lums)

    def genLuminosityFunctionZ(self, lums, z):
        """Calculate the luminosity function at one redshift

        Parameters
        ----------
        lums : np.array
            Array of luminosities at which to calculate the LF
        z : float
            Redshift at which to calculate the LF

        Returns
        -------
        out : np.array
            A structured array, containing luminosities and number
            densities of the LF
        """

        zp = self.evolveParams(z)
        lf = self.numberDensity(zp, lums)
        out = np.zeros(len(lf[0]), dtype=np.dtype(
            [('mag', np.float), ('phi', np.float)]))
        out['mag'] = lums
        out['phi'] = lf

        return out

    def numberDensity(self, p, lums):
        """Return number density in units of Mpc^{-3} h^{3}

        Parameters
        ----------
        lums : np.array
            Array of luminosities at which to calculate the LF
        p : np.array
            Parameters of the LF

        Returns
        -------
        phi : np.array
            Array of number densities at lums
        """
        pass

    def numberDensitySingleZL(self, z, l):
        """Calculate the number density for a single redshift and luminosity.

        Parameters
        ----------
        z : float
            Redshift
        l : float
            Luminosity

        Returns
        -------
        nd : float
            number density

        """

        zp = self.evolveParams(z)
        nd = self.numberDensity(zp, l)

        return nd

    def cumulativeNumberDensity(self, z, L):
        """Cumulative number density at redshift z until luminosity L.

        Parameters
        ----------
        z : type
            Description of parameter `z`.
        L : type
            Description of parameter `L`.

        Returns
        -------
        nd - np.float
            The cumulative number density at L, z

        """

        nd = quad(self.numberDensityIntegrandZL,
                  self.m_max_of_z(z), L, args=z)[0]

        return nd

    def m_min_of_z(self, z):
        if (self.magmin - self.cosmo.distanceModulus(z)) < -5:
            return self.magmin - self.cosmo.distanceModulus(z)
        else:
            return -5.

    def m_max_of_z(self, z):

        return self.magmax - self.cosmo.distanceModulus(0.05)

    def evolveParams(self, z):
        """Evolve base parameters of LF to a redshift of z. Usually
        using the typical Q, P evolution parameters.

        Parameters
        ----------
        z : float
            Redshift to evolve parameters to

        Returns
        -------
        zp : list
            Parameters evolved to redshift z

        """
        pass

    def numberDensityIntegrandZL(self, l, z):

        return self.numberDensitySingleZL(z, l) * self.cosmo.dVdz(z)

    def integrateZL(self, z_min, z_max, area):
        """Integrate the luminosity function over a redshift and luminosity
        range to give a total number of galaxies in some volume.

        Parameters
        ----------
        z_min : float
            Low z bound of redshift integral
        z_max : float
            High z bound of redshift integral
        area : float
            If lightcone, the area the the volume subtends
        Returns
        -------
        n_gal : float
            Number of galaxies in this volume
        """

        n_gal = (area / 41253.) * dblquad(self.numberDensityIntegrandZL, z_min, z_max,
                                          self.m_max_of_z, self.m_min_of_z,
                                          epsabs=1e-2, epsrel=1e-2)[0]

        return int(n_gal)

    def integrateL(self, z, volume):
        """Integrate the luminosity function over a redshift and luminosity
        range to give a total number of galaxies in some volume.

        Parameters
        ----------
        z : float
            Redshift to evaluate the LF at
        volume : float
            If lightcone, the area the the volume subtends
        Returns
        -------
        n_gal : float
            Number of galaxies in this volume

        """

        n_gal = volume * quad(lambda l: self.numberDensitySingleZL(z, l),
                              self.m_max_of_z_snap, self.m_min_of_z_snap,
                              epsabs=1e-2, epsrel=1e-2)[0]

        return int(n_gal)

    def drawRedshifts(self, domain, overdens):

        if domain.fmt == 'BCCLightcone':
            z_min = domain.zmin
            z_max = domain.zmax

            z_bins, nd_cumu = self.redshiftCDF(z_min, z_max, domain)
            nd_spl = interp1d(z_bins, nd_cumu)
            z_fine = np.linspace(z_min, z_max, 10000)
            nd = nd_spl(z_fine) * overdens
            cdf = nd / nd[-1]
            rand = np.random.rand(int(nd[-1]))
            z_samp = np.linspace(z_min, z_max, 10000)[cdf.searchsorted(rand) - 1]

        elif domain.fmt == 'Snapshot':
            z_min = domain.zmean
            z_max = domain.zmean
            n_gal = self.integrateL(domain.zmean, domain.getVolume())
            z_samp = np.zeros(n_gal) + domain.zmean

        return z_samp

    def redshiftCDF(self, z_min, z_max, domain):

        zbins = np.linspace(z_min, z_max, 100)
        n_gal_cum = [self.integrateZL(z_min, zc, domain.getArea())
                     for zc in zbins]
        return zbins, np.array(n_gal_cum)

    def sampleLuminosities(self, domain, z):

        n_gal = z.size
        zbins = np.arange(domain.zmin, domain.zmax + 0.001, 0.001)

        zmean = (zbins[1:] + zbins[:-1]) / 2
        volume = domain.getVolume()

        nzbins = zmean.size
        lums_gal = np.zeros(n_gal)
        count = 0

        for i in range(nzbins):
            zlidx = z.searchsorted(zbins[i])
            zhidx = z.searchsorted(zbins[i + 1])
            n_gal_z = zhidx - zlidx

            # calculate faintest luminosity to use given
            # the apparent magnitude limit that we want to
            # populate to
            if self.m_min_of_z_snap is not None:
                lummin = self.m_min_of_z_snap
            else:
                lummin = self.m_min_of_z(zmean[i])

            lums = np.linspace(self.m_max_of_z(0.0), lummin, 100000)

            # get the parameters at this redshift
            params = self.evolveParams(zmean[i])

            number_density = self.numberDensity(params, lums)
            cdf_lum = np.cumsum(number_density * (lums[1] - lums[0]))
            cdf_lum /= cdf_lum[-1]

            # sample from CDF
            rands = np.random.uniform(size=n_gal_z)
            lums_gal[count:count + n_gal_z] = lums[cdf_lum.searchsorted(rands)]
            count += n_gal_z

        return lums_gal


    def sampleLuminositiesSnap(self, domain, z):

        n_gal = z.size
        zmean = domain.zmean
        volume = domain.getVolume()

        if self.m_min_of_z_snap is not None:
            lummin = self.m_min_of_z_snap
        else:
            lummin = self.m_min_of_z(zmean[i])

        lums = np.linspace(self.m_max_of_z(0.0), lummin, 100000)

        # get the parameters at this redshift
        params = self.evolveParams(zmean)

        number_density = self.numberDensity(params, lums)
        cdf_lum = np.cumsum(number_density * (lums[1] - lums[0]))
        cdf_lum /= cdf_lum[-1]

        # sample from CDF
        rands = np.random.uniform(size=n_gal)
        lums_gal = lums[cdf_lum.searchsorted(rands)]

        return lums_gal


class DSGLuminosityFunction(LuminosityFunction):

    def __init__(self, cosmo, params=None, name=None, **kwargs):

        if params is None:
            params = np.array([1.56000000e-02, -1.66000000e-01,
                               6.71000000e-03,
                               -1.52300000e+00, -2.00100000e+01,
                               3.08000000e-05,
                               -2.18500000e+01, 4.84000000e-01, -1, 0])

        LuminosityFunction.__init__(
            self, cosmo, params=params, name='DSG', **kwargs)
        self.unitmap = {'mag': 'magh', 'phi': 'hmpc3dex'}

    def evolveParams(self, z):
        zp = copy(self.params)

        zp[0] += self.params[-1] * (1 / (z + 1) - 1 / 1.1)
        zp[2] += self.params[-1] * (1 / (z + 1) - 1 / 1.1)
        zp[5] += self.params[-1] * (1 / (z + 1) - 1 / 1.1)

        zp[4] += self.params[-2] * (1 / (z + 1) - 1 / 1.1)
        zp[6] += self.params[-2] * (1 / (z + 1) - 1 / 1.1)

        return zp

    def numberDensity(self, p, lums):
        """
        Sum of a double schechter function and a gaussian.
        m -- magnitudes at which to calculate the number density
        p -- Function parameters. Order
             should be phi^{star}_{1}, M^{star}, \alpha_{1},
             phi^{star}_{2}, M^{star}, \alpha_{2}, \phi_{gauss},
             \M_{gauss}, \sigma_{gauss}
        """
        phi = 0.4 * np.log(10) * np.exp(-10**(-0.4 * (lums - p[4]))) * \
            (p[0] * 10 ** (-0.4 * (lums - p[4]) * (p[1] + 1)) +
             p[2] * 10 ** (-0.4 * (lums - p[4]) * (p[3] + 1))) + \
            p[5] / np.sqrt(2 * np.pi * p[7] ** 2) * \
            np.exp(-(lums - p[6]) ** 2 / (2 * p[7] ** 2))

        return phi


def read_tabulated_loglf(filename):

    data = np.loadtxt(filename)
    lf = data[:, :2]
    lf[:, 1] = 10**lf[:, 1]

    return lf


def read_tabulated_lf(filename):

    data = np.loadtxt(filename)
    lf = data[:, :2]

    return lf


def read_tabulated_bbgs_lf(filename):

    data = np.loadtxt(filename)
    lf = data

    return lf


class BBGSLuminosityFunction(LuminosityFunction):

    def __init__(self, Q, P, **kwargs):

        self.lf = read_tabulated_bbgs_lf(
            '/home/jderose/projects/l-addgals/src/training/rdel/LF_r_z0.1_bright_end_evol.txt')
        self.Q = Q
        self.P = P
        self.unitmap = {'mag': 'mag', 'phi': 'hmpc3dex'}

        LuminosityFunction.__init__(
            self, np.array([Q, P]), name='BBGS', **kwargs)

    def evolveParams(self, z):
        return self.params, z

    def numberDensity(self, p, lums):
        Q = p[0][0]
        P = p[0][1]
        z = p[1]

        if z > 0.45:
            pz = 0.45
        elif z <= 0.05:
            pz = 0.05
        else:
            pz = z

        phi = self.lf[:, 1] + (pz - 0.05) / 0.4 * self.lf[:, 2]

        mag = self.lf[:, 0] + Q * (1 / (z + 1.0) - 1 / 1.1)
        phi = phi * 10 ** (0.4 * P * (pz - 0.1))
        af = AbundanceFunction(mag, phi, ext_range=(-26, 10),
                               nbin=2000, faint_end_fit_points=6)

        return af(self.lf[:, 0])


class CapozziLuminosityFunction(LuminosityFunction):

    def __init__(self, params=None, **kwargs):

        if params is None:

            self.phi0 = 39.4e-4 / 0.7**3
            self.mstar0 = -21.63 - 5 * np.log10(0.7)
            self.Q = 2.9393
            self.P = [-0.00480474, -0.06140413]

    def evolveParams(self, z):
        zp = copy([self.phi0, -1.2, self.mstar0])

        zp[0] += self.P[0] * (1 / (z + 1) - 1 / 1.1) + \
            self.P[1] * (1 / (z + 1) - 1 / 1.1) ** 2
        zp[-1] += self.Q * (1 / (z + 1) - 1 / 1.1)

        return zp

    def numberDensity(self, p, lums):
        phi = (0.4 * np.log(10) * np.exp(-10**(-0.4 * (lums - p[2]))) *
               (p[0] * 10 ** (-0.4 * (lums - p[2]) * (p[1] + 1))))

        return phi


class BernardiLuminosityFunction(LuminosityFunction):

    def __init__(self, Q, **kwargs):

        self.lf = read_tabulated_loglf(
            '/nfs/slac/g/ki/ki23/des/jderose/amatch/bernardi-test/anc/LF_SerExp.dat')
        print(self.lf[:, 1])
        self.Q = Q
        self.unitmap = {'mag': 'mag', 'phi': 'mpc3dex'}

        LuminosityFunction.__init__(self, Q, name='Bernardi', **kwargs)

    def evolveParams(self, z):
        return self.Q, z

    def numberDensity(self, p, lums):
        """
        Shift the tabulated Bernardi 2013 luminosity function
        p -- Q, h  and z
        lums -- Null
        """
        Q = p[0]
        z = p[1]

        self.lf[:, 0] += Q * (1 / (1 + z) - 1 / 1.1)

        return self.lf[:, 1]


class ReddickLuminosityFunction(LuminosityFunction):

    def __init__(self, Q):

        self.lf = load_abundance_function(log_phi=False)
        self.Q = Q
        self.unitmap = {'mag': 'magh', 'phi': 'hmpc3dex'}

        LuminosityFunction.__init__(self, Q, name='Reddick', **kwargs)

    def evolveParams(self, z):
        return self.Q, z

    def numberDensity(self, p, lums):
        """
        Shift the tabulated Bernardi 2013 luminosity function
        p -- Q, h  and z
        lums -- Null
        """
        Q = p[0]
        z = p[1]

        self.lf[:, 0] += Q * (1 / (1 + z) - 1 / 1.1)

        return self.lf[:, 1]

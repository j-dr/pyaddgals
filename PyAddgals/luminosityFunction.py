from __future__ import print_function, division
from scipy.integrate import dblquad
import numpy as np


class LuminosityFunction(object):

    def __init__(self, params, name=None, magmin=25., magmax=10.):
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
        self.magmin = float(magmin)

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
            self.lf[:, i] = self.calcNumberDensity(zp, lums)

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
        lf = self.calcNumberDensity(zp, lums)
        out = np.zeros(len(lf[0]), dtype=np.dtype(
            [('mag', np.float), ('phi', np.float)]))
        out['mag'] = lums
        out['phi'] = lf

        return out

    def calcNumberDensity(self, p, lums):
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

    def calcNumberDensitySingleZL(self, z, l):
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
        nd = self.calcNumberDensity(zp, l)

        return nd

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

    def integrate(self, cosmo, z_min, z_max, area):
        """Integrate the luminosity function over a redshift and luminosity
        range to give a total number of galaxies in some volume.

        Parameters
        ----------
        cosmo : Cosmology
            Cosmology object
        m_min : float
            Faint end bound of luminosity integral
        m_max : float
            Bright end bound of luminosity integral
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

        m_min_of_z = lambda z : self.magmin - cosmo.distanceModulus(z)
        m_max_of_z = lambda z : self.magmax - cosmo.distanceModulus(z)

        f = lambda z, l : self.calcNumberDensitySingleZL(z, l) * cosmo.dVdz(z)

        n_gal = dblquad(f, z_min, z_max, m_min_of_z, m_max_of_z)

        return n_gal[0]



    def drawLuminosities(self, cosmo, domain):

        zmin = domain.zmin
        zmean = domain.zmean
        volume = domain.volume

        #calculate faintest luminosity to use given
        #the apparent magnitude limit that we want to
        #populate to
        lummin = self.magmin - cosmo.distanceModulus(z)
        lums = np.linspace(-25, lummin, 10000)

        #get the parameters at the
        params = self.evolveParams(z)
        number_density = self.calcNumberDensity(params, lums)
        cdf_lum = np.cumsum(number_density)
        cdf_lum /= p_of_lum[-1]

        n_gals = number_density[-1] * volume
        rands = np.random.uniform(size=n_gals)
        lum_gals = cdf_lum.searchsorted(rands)

        return lum_gals


class DSGLuminosityFunction(LuminosityFunction):

    def __init__(self, params=None, name=None):

        if params is None:
            params = np.array([1.56000000e-02,  -1.66000000e-01,
                                6.71000000e-03,
                               -1.52300000e+00,  -2.00100000e+01,
                               3.08000000e-05,
                               -2.18500000e+01,   4.84000000e-01, -1, 0])

        LuminosityFunction.__init__(self, params, name='DSG')
        self.unitmap = {'mag': 'magh', 'phi': 'hmpc3dex'}

    def evolveParams(self, z):
        zp = copy(self.params)

        zp[0] += self.params[-1] * (1 / (z + 1) - 1 / 1.1)
        zp[2] += self.params[-1] * (1 / (z + 1) - 1 / 1.1)
        zp[5] += self.params[-1] * (1 / (z + 1) - 1 / 1.1)

        zp[4] += self.params[-2] * (1 / (z + 1) - 1 / 1.1)
        zp[6] += self.params[-2] * (1 / (z + 1) - 1 / 1.1)

        return zp

    def calcNumberDensity(self, p, lums):
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

    def __init__(self, Q, P):

        self.lf = read_tabulated_bbgs_lf(
            '/home/jderose/projects/l-addgals/src/training/rdel/LF_r_z0.1_bright_end_evol.txt')
        self.Q = Q
        self.P = P
        self.unitmap = {'mag': 'mag', 'phi': 'hmpc3dex'}

        LuminosityFunction.__init__(self, np.array([Q, P]), name='BBGS')

    def evolveParams(self, z):
        return self.params, z

    def calcNumberDensity(self, p, lums):
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

    def __init__(self, params=None):

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

    def calcNumberDensity(self, p, lums):
        phi = (0.4 * np.log(10) * np.exp(-10**(-0.4 * (lums - p[2]))) *
               (p[0] * 10 ** (-0.4 * (lums - p[2]) * (p[1] + 1))))

        return phi


class BernardiLuminosityFunction(LuminosityFunction):

    def __init__(self, Q):

        self.lf = read_tabulated_loglf(
            '/nfs/slac/g/ki/ki23/des/jderose/amatch/bernardi-test/anc/LF_SerExp.dat')
        print(self.lf[:, 1])
        self.Q = Q
        self.unitmap = {'mag': 'mag', 'phi': 'mpc3dex'}

        LuminosityFunction.__init__(self, Q, name='Bernardi')

    def evolveParams(self, z):
        return self.Q, z

    def calcNumberDensity(self, p, lums):
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

        LuminosityFunction.__init__(self, Q, name='Reddick')

    def evolveParams(self, z):
        return self.Q, z

    def calcNumberDensity(self, p, lums):
        """
        Shift the tabulated Bernardi 2013 luminosity function
        p -- Q, h  and z
        lums -- Null
        """
        Q = p[0]
        z = p[1]

        self.lf[:, 0] += Q * (1 / (1 + z) - 1 / 1.1)

        return self.lf[:, 1]

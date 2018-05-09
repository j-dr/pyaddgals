from __future__ import print_function, division
import pyccl as ccl
from scipy.misc import derivative


class Cosmology(object):

    def __init__(self, omega_m=None, omega_b=None, n_s=None, h=None,
                    sigma8=None, a_s=None, w=-1.0, n_eff=3.046, n_nu_mass=0.0,
                    m_nu=0.0):

        if omega_m is None:
            raise(ValueError("Must define omega_m"))

        if omega_b is None:
            raise(ValueError("Must define omega_b"))

        if n_s is None:
            raise(ValueError("Must define n_s"))

        if h is None:
            raise(ValueError("Must define h"))

        if (sigma8 is None) & (a_s is None):
            raise(ValueError('Must define either sigma8 or A_s'))


        self.omega_m = float(omega_m)
        self.omega_b = float(omega_b)
        self.n_s = float(n_s)
        self.w = float(w)
        self.m_nu = float(m_nu)
        self.n_eff = float(n_eff)
        self.n_nu_mass = float(n_nu_mass)
        self.m_nu = float(m_nu)
        self.omega_c = omega_m - omega_b

        if sigma8 is None:
            self.a_s = float(a_s)
        else:
            self.sigma8 = float(sigma8)

        self._cosmo = ccl.Cosmology(Omega_c=self.omega_c, Omega_b=self.omega_b,
                                    h=1.0, n_s=self.n_s, sigma8=self.sigma8,
                                    A_s=self.a_s, w0=self.w, m_nu=self.m_nu,
                                    N_nu_rel=self.n_eff,
                                    N_nu_mass=self.n_nu_mass)

    def zofR(self, r):
        """Calculate redshift from comoving radial distance.

        Parameters
        ----------
        r : np.array
            Array of comoving radial distances

        Returns
        -------
        z : np.array
            Array of redshifts corresponding to input comoving radial distance

        """
        z = 1 / ccl.scale_factor_of_chi(self._cosmo, r) - 1

        return z


    def rofZ(self, z):
        """Calculate comoving radial distance from redshift.

        Parameters
        ----------
        z : np.array
            Array of cosmological redshifts

        Returns
        -------
        r : np.array
            Array of comoving radial distance corresponding to input redshifts

        """

        r = ccl.comoving_radial_distance(self._cosmo, 1 / (z + 1.))
        return r

    def distanceModulus(self, z):
        """Compute the distance modulus as a function of redshift

        Parameters
        ----------
        z : array
            redshifts to compute distance modulus at

        Returns
        -------
        distance_modulus : array
            distance modulus at input redshifts

        """

        distance_modulus = ccl.distance_modulus(self._cosmo, 1 / (z + 1.))
        return distance_modulus

    def comovingVolume(self,z):
        """Comoving Volume out to redshift z

        Parameters
        ----------
        z : np.array
            The redshifts at which to compute the comoving volume

        Returns
        -------
        comoving_volume : np.array
            The comoving volume at the input redshifts

        """

        r = self.rofZ(z)
        comoving_volume = 4 * np.pi * r ** 3 / 3

        return comoving_volume

    def dVdz(self, z):
        """Derivative of comoving volume with respect to redshift

        Parameters
        ----------
        z : np.array
            Redshifts to compute the derivative at

        Returns
        -------
        dVdz : np.array
            derivative of volume with respect to redshift

        """


        f = lambda z : self.comovingVolume(z)
        dVdz = derivative(f, z)

        return dVdz

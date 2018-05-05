from __future__ import print_function, division
import pyccl as ccl


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
        self.omega_b = float(omega_b
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

        self._cosmo = ccl.Cosmology(Omega_c=self.omega_c, Omega_b=self.omega_b
                                    h=1.0, n_s=self.n_s, sigma8=self.sigma8
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

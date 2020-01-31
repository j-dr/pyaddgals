from __future__ import print_function, division
from copy import copy
import numpy as np

from .util import load_abundance_function
from .abundancematch import AbundanceFunction

class LuminosityFunction(object):

    def __init__(self, params, name=None):

        self.name = name
        self.params = params

    def genLuminosityFunction(self, lums, zs):

        self.lf = np.zeros((len(lums), len(zs)))

        for i, z in enumerate(zs):
            zp = self.evolveParams(z)
            self.lf[:,i] = self.calcNumberDensity(zp, lums)

    def genLuminosityFunctionZ(self, lums, z):
        zp = self.evolveParams(z)
        lf = self.calcNumberDensity(zp, lums)
        out = np.zeros(len(lf[0]), dtype=np.dtype([('mag',np.float), ('phi',np.float)]))
        out['mag'] = lf[0]
        out['phi'] = lf[1]

        return out

    def calcNumberDensity(self, p, lums):
        """
        Return number density in units of Mpc^{-3} h^{3}
        """
        pass

    def evolveParams(self, z):
        pass


class DSGLuminosityFunction(LuminosityFunction):

    def __init__(self, params=None, name=None):

        if params is None:
            params = np.array([  1.56000000e-02,  -1.66000000e-01,   6.71000000e-03,
                                 -1.52300000e+00,  -2.00100000e+01,   3.08000000e-05,
                                 -2.18500000e+01,   4.84000000e-01, -1, 0])

        LuminosityFunction.__init__(self,params,name='DSG')
        self.unitmap = {'mag':'magh', 'phi':'hmpc3dex'}

    def evolveParams(self, z):
        zp = copy(self.params)

        zp[0] += self.params[-1] * (1/(z+1) - 1/1.1)
        zp[2] += self.params[-1] * (1/(z+1) - 1/1.1)
        zp[5] += self.params[-1] * (1/(z+1) - 1/1.1)

        zp[4] += self.params[-2] * (1/(z+1) - 1/1.1)
        zp[6] += self.params[-2] * (1/(z+1) - 1/1.1)

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
            (p[0] * 10 ** (-0.4 * (lums - p[4])*(p[1]+1)) + \
            p[2] * 10 ** (-0.4 * (lums - p[4])*(p[3]+1))) + \
            p[5] / np.sqrt(2 * np.pi * p[7] ** 2) * \
            np.exp(-(lums - p[6]) ** 2 / (2 * p[7] ** 2))

        return lums, phi


def read_tabulated_loglf(filename):

    data = np.loadtxt(filename)
    lf = data[:,:2]
    lf[:,1] = 10**lf[:,1]

    return lf

def read_tabulated_lf(filename):

    data = np.loadtxt(filename)
    lf = data[:,:2]

    return lf

def read_tabulated_bbgs_lf(filename):

    data = np.loadtxt(filename)
    lf = data

    return lf

class BBGSLuminosityFunction(LuminosityFunction):

    def __init__(self, Q, P):

        self.lf = read_tabulated_bbgs_lf('/u/ki/jderose/ki23/l-addgals/src/training/rdel/LF_r_z0.1_bright_end_evol.txt')
        self.Q = Q
        self.P = P
        self.unitmap = {'mag':'mag', 'phi':'hmpc3dex'}

        LuminosityFunction.__init__(self,np.array([Q,P]),name='BBGS')

    def evolveParams(self, z):
        return self.params, z

    def calcNumberDensity(self, p, lums):
        Q = p[0][0]
        P = p[0][1]
        z = p[1]

        if z > 0.45:
            pz = 0.45
        elif z<=0.05:
            pz = 0.05
        else:
            pz = z

        phi  = self.lf[:,1] + (pz - 0.05) / 0.4 * self.lf[:,2]

        mag  = self.lf[:,0] + Q * (1/(z + 1.0) - 1/1.1)
        phi  = phi * 10 ** (0.4 * P * (pz - 0.1))
        af = AbundanceFunction(mag, phi, ext_range=(-26,10), nbin=2000, faint_end_fit_points=6)

        return self.lf[:,0], af(self.lf[:,0])

class CapozziLuminosityFunction(LuminosityFunction):

    def __init__(self, params=None):

        if params is None:

            self.phi0   = 39.4e-4 / 0.7**3
            self.mstar0 = -21.63 - 5 * np.log10(0.7)
            self.Q      = 2.9393
            self.P      = [-0.00480474, -0.06140413]

    def evolveParams(self, z):
        zp = copy([self.phi0, -1.2, self.mstar0])

        zp[0]  += self.P[0] * (1/(z+1) - 1/1.1) + self.P[1] * (1/(z+1) - 1/1.1) ** 2
        zp[-1] += self.Q * (1/(z+1) - 1/1.1)

        return zp

    def calcNumberDensity(self, p, lums):
        phi = (0.4 * np.log(10) * np.exp(-10**(-0.4 * (lums - p[2]))) *
               (p[0] * 10 ** (-0.4 * (lums - p[2])*(p[1]+1))))

        return phi


class BernardiLuminosityFunction(LuminosityFunction):

    def __init__(self, Q):

        self.lf = read_tabulated_loglf('/nfs/slac/g/ki/ki23/des/jderose/amatch/bernardi-test/anc/LF_SerExp.dat')
        print(self.lf[:,1])
        self.Q = Q
        self.unitmap = {'mag':'mag', 'phi':'mpc3dex'}

        LuminosityFunction.__init__(self,Q,name='Bernardi')

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

        self.lf[:,0] += Q*(1/(1+z)-1/1.1)

        return self.lf[:,0], self.lf[:,1]


class ReddickLuminosityFunction(LuminosityFunction):

    def __init__(self, Q):

        self.lf = load_abundance_function(log_phi=False)
        self.Q = Q
        self.unitmap = {'mag':'magh', 'phi':'hmpc3dex'}

        LuminosityFunction.__init__(self,Q,name='Reddick')

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

        self.lf[:,0] += Q*(1/(1+z)-1/1.1)

        return self.lf[:,0], self.lf[:,1]


class ReddickLuminosityFunction(LuminosityFunction):

    def __init__(self, Q):

        self.lf = load_abundance_function(log_phi=False)
        self.Q = Q
        self.unitmap = {'mag':'magh', 'phi':'hmpc3dex'}

        LuminosityFunction.__init__(self,Q,name='Reddick')

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

        self.lf[:,0] += Q*(1/(1+z)-1/1.1)

        return self.lf[:,0], self.lf[:,1]

class ReddickStellarMassFunction(LuminosityFunction):

    def __init__(self, Q):

        self.lf = load_abundance_function(log_phi=False, proxy='s', sample_cut=9.8)
        self.Q = Q
        self.unitmap = {'mag':'magh', 'phi':'hmpc3dex'}

        LuminosityFunction.__init__(self,Q,name='Reddick')

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

        self.lf[:,0] += Q*(1/(1+z)-1/1.1)

        return self.lf[:,0], self.lf[:,1]

import fitsio
import numpy as np
import sys
import os
from mpi4py import MPI
from glob import glob
from numpy.lib.recfunctions import merge_arrays


class DataObject(object):
    """Abstract base class to encapsulate info from FITS files."""

    def __init__(self, *arrays):
        """Constructs DataObject from arbitrary number of ndarrays.

        Each ndarray can have an arbitrary number of fields. Field
        names should all be capitalized and words in multi-word field
        names should be separated by underscores if necessary. ndarrays
        have a 'size' property---their sizes should all be equivalent.

        Args:
            arrays (numpy.ndarray): ndarrays with equivalent sizes.
        """
        self._ndarray = merge_arrays(arrays, flatten=True)

    @classmethod
    def from_fits_file(cls, filename):
        """
        Constructs DataObject directly from FITS file.

        Makes use of Erin Sheldon's fitsio reading routine. The fitsio
        package wraps cfitsio for maximum efficiency.

        Args:
            filename (string): the file path and name.

        Returns:
            DataObject, properly constructed.
        """
        array = fitsio.read(filename, ext=1)
        return cls(array)

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass
        if attr.upper() in self._ndarray.dtype.names:
            return self._ndarray[attr.upper()]
        elif attr.lower() in self._ndarray.dtype.names:
            return self._ndarray[attr.lower()]
        return object.__getattribute__(self, attr)

    def __setattr__(self, attr, val):
        if attr == '_ndarray':
            object.__setattr__(self, attr, val)
        elif attr.upper() in self._ndarray.dtype.names:
            self._ndarray[attr.upper()] = val
        else:
            object.__setattr__(self, attr, val)

    @property
    def dtype(self):
        """numpy.dtype: dtype associated with the DataObject."""
        return self._ndarray.dtype

    def add_fields(self, newdtype):
        array = np.zeros(self._ndarray.size, newdtype)
        self._ndarray = merge_arrays([self._ndarray, array], flatten=True)

    def __repr__(self):
        # return the representation of the underlying array
        return repr(self._ndarray)

    def __str__(self):
        # return the string of the underlying array
        return str(self._ndarray)

    def __dir__(self):
        # lower case list of all the available variables
        # also need to know original __dir__!
        # return [x.lower() for x in self._ndarray.dtype.names]
        return sorted(set(
            dir(type(self)) +
            self.__dict__.keys() +
            [x.lower() for x in self._ndarray.dtype.names]))


class Entry(DataObject):
    """Entries are extensions of DataObjects.

    The __init__ method simply calls the
    constructor for DataObject after it has verified that
    there is only a single entry being passed in.
    """

    def __init__(self, *arrays):
        if any([arr.size != 1 for arr in arrays]):
            raise ValueError("Input arrays must have length one.")
        super(Entry, self).__init__(*arrays)

    @classmethod
    def from_dict(cls, dict): pass

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass
        if attr.upper() in self._ndarray.dtype.names:
            return self._ndarray[attr.upper()][0]
        elif attr.lower() in self._ndarray.dtype.names:
            return self._ndarray[attr.lower()][0]
        return object.__getattribute__(self, attr)


class Catalog(DataObject):
    """Catalogs are extensions of DataObjects.

    Catalogs are composed of may Entry objects.
    Tom - I am not sure that this object is complete. TODO
    Eli - It might be.  The tricks here are so you can access
           these with "catalog.key" rather than "catalog['KEY']"
    """

    entry_class = Entry

    @property
    def size(self): return self._ndarray.size

    def __len__(self): return self.size

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.entry_class(self._ndarray.__getitem__(key))
        return type(self)(self._ndarray.__getitem__(key))

    def __setitem__(self, key, val):
        self._ndarray.__setitem__(key, val)

# %load /nfs/slac/g/ki/ki19/des/erykoff/src/redmapper/redmapper/redmapper/utilities.py
# miscellaneous utilities and functions


import numpy as np
from scipy import special
from scipy.linalg import solve_banded
from pkg_resources import resource_filename
import scipy.interpolate as interpolate
import fitsio
from scipy.special import erf
from numpy import random

###################################
## Useful constants/conversions ##
###################################
TOTAL_SQDEG = 4 * 180**2 / np.pi
SEC_PER_DEG = 3600


def astro_to_sphere(ra, dec):
    return np.radians(90.0 - dec), np.radians(ra)

# Equation 7 in Rykoff et al. 2014


def chisq_pdf(data, k):
    normalization = 1. / (2**(k / 2.) * special.gamma(k / 2.))
    return normalization * data**((k / 2.) - 1) * np.exp(-data / 2.)


def gaussFunction(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu)**2. / (2. * sigma**2))


######################################
# mstar LUT code
######################################
class MStar(object):
    def __init__(self, survey, band, mstarpath):
        self.survey = survey.strip()
        self.band = band.strip()

        try:
            self.mstar_file = mstarpath + '/mstar_%s_%s.fit' % (
                self.survey, self.band)
        except:
            raise IOError("Could not find mstar resource mstar_%s_%s.fit" % (
                self.survey, self.band))
        try:
            self._mstar_arr = fitsio.read(self.mstar_file, ext=1)
        except:
            raise IOError("Could not find mstar file mstar_%s_%s.fit" %
                          (self.survey, self.band))

        # Tom - why not use CubicSpline here? That's why it exists...
        self._f = CubicSpline(self._mstar_arr['Z'], self._mstar_arr['MSTAR'])
        #self._f = interpolate.interp1d(self._mstar_arr['Z'],self._mstar_arr['MSTAR'],kind='cubic')

    def __call__(self, z):
        # may want to check the type ... if it's a scalar, return scalar?  TBD
        return self._f(z)


#############################################################
# cubic spline interpolation, based on Eddie Schlafly's code, from NumRec
# http://faun.rc.fas.harvard.edu/eschlafly/apored/cubicspline.py
#############################################################
class CubicSpline(object):
    def __init__(self, x, y, yp=None):
        npts = len(x)
        mat = np.zeros((3, npts))
        # enforce continuity of 1st derivatives
        mat[1, 1:-1] = (x[2:] - x[0:-2]) / 3.
        mat[2, 0:-2] = (x[1:-1] - x[0:-2]) / 6.
        mat[0, 2:] = (x[2:] - x[1:-1]) / 6.
        bb = np.zeros(npts)
        bb[1:-1] = ((y[2:] - y[1:-1]) / (x[2:] - x[1:-1]) -
                    (y[1:-1] - y[0:-2]) / (x[1:-1] - x[0:-2]))
        if yp is None:  # natural cubic spline
            mat[1, 0] = 1.
            mat[1, -1] = 1.
            bb[0] = 0.
            bb[-1] = 0.
        elif yp == '3d=0':
            mat[1, 0] = -1. / (x[1] - x[0])
            mat[0, 1] = 1. / (x[1] - x[0])
            mat[1, -1] = 1. / (x[-2] - x[-1])
            mat[2, -2] = -1. / (x[-2] - x[-1])
            bb[0] = 0.
            bb[-1] = 0.
        else:
            mat[1, 0] = -1. / 3. * (x[1] - x[0])
            mat[0, 1] = -1. / 6. * (x[1] - x[0])
            mat[2, -2] = 1. / 6. * (x[-1] - x[-2])
            mat[1, -1] = 1. / 3. * (x[-1] - x[-2])
            bb[0] = yp[0] - 1. * (y[1] - y[0]) / (x[1] - x[0])
            bb[-1] = yp[1] - 1. * (y[-1] - y[-2]) / (x[-1] - x[-2])
        y2 = solve_banded((1, 1), mat, bb)
        self.x, self.y, self.y2 = (x, y, y2)

    def splint(self, x):
        npts = len(self.x)
        lo = np.searchsorted(self.x, x) - 1
        lo = np.clip(lo, 0, npts - 2)
        hi = lo + 1
        dx = self.x[hi] - self.x[lo]
        a = (self.x[hi] - x) / dx
        b = (x - self.x[lo]) / dx
        y = (a * self.y[lo] + b * self.y[hi] +
             ((a**3 - a) * self.y2[lo] + (b**3 - b) * self.y2[hi]) * dx**2. / 6.)
        return y

    def __call__(self, x):
        return self.splint(x)


def calc_theta_i(mag, mag_err, maxmag, limmag):
    """
    Calculate theta_i. This is reproduced from calclambda_chisq_theta_i.pr

    parameters
    ----------
    mag:
    mag_err:
    maxmag:
    limmag:

    returns
    -------
    theta_i:
    """

    theta_i = np.ones((len(mag)))
    eff_lim = np.clip(maxmag, 0, limmag)
    dmag = eff_lim - mag
    calc = dmag < 5.0
    N_calc = np.count_nonzero(calc == True)
    if N_calc > 0:
        theta_i[calc] = 0.5 + 0.5 * \
            erf(dmag[calc] / (np.sqrt(2) * mag_err[calc]))
    hi = mag > limmag
    N_hi = np.count_nonzero(hi == True)
    if N_hi > 0:
        theta_i[hi] = 0.0
    return theta_i


def apply_errormodels(maskgals, mag_in, b=None, err_ratio=1.0, fluxmode=False,
                      nonoise=False, inlup=False):
    """
    Find magnitude and uncertainty.

    parameters
    ----------
    mag_in    :
    nonoise   : account for noise / no noise
    zp:       : Zero point magnitudes
    nsig:     :
    fluxmode  :
    lnscat    :
    b         : parameters for luptitude calculation
    inlup     :
    errtflux  :
    err_ratio : scaling factor

    returns
    -------
    mag
    mag_err

    """
    f1lim = 10.**((maskgals.limmag - maskgals.zp[0]) / (-2.5))
    fsky1 = (((f1lim**2.) * maskgals.exptime) / (maskgals.nsig[0]**2.) - f1lim)
    fsky1 = np.clip(fsky1, 0.001, None)

    if inlup:
        bnmgy = b * 1e9
        tflux = maskgals.exptime * 2.0 * bnmgy * \
            np.sinh(-np.log(b) - 0.4 * np.log(10.0) * mag_in)
    else:
        tflux = maskgals.exptime * 10.**((mag_in - maskgals.zp[0]) / (-2.5))

    noise = err_ratio * np.sqrt(fsky1 * maskgals.exptime + tflux)

    if nonoise:
        flux = tflux
    else:
        flux = tflux + noise * random.standard_normal(mag_in.size)

    if fluxmode:
        mag = flux / maskgals.exptime
        mag_err = noise / maskgals.exptime
    else:
        if b is not None:
            bnmgy = b * 1e9

            flux_new = flux / maskgals.exptime
            noise_new = noise / maskgals.exptime

            mag = 2.5 * np.log10(1.0 / b) - np.arcsinh(0.5 *
                                                       flux_new / bnmgy) / (0.4 * np.log(10.0))
            mag_err = 2.5 * noise_new / \
                (2.0 * bnmgy * np.log(10.0) *
                 np.sqrt(1.0 + (0.5 * flux_new / bnmgy)**2.0))
        else:
            mag = maskgals.zp[0] - 2.5 * np.log10(flux / maskgals.exptime)
            mag_err = (2.5 / np.log(10.0)) * (noise / flux)

            bad, = np.where(np.isfinite(mag) == False)
            mag[bad] = 99.0
            mag_err[bad] = 99.0

    return mag, mag_err


# %load /nfs/slac/g/ki/ki19/des/erykoff/src/redmapper/redmapper/redmapper/redsequence.py
import fitsio
import numpy as np
from scipy import interpolate


class RedSequenceColorPar(object):
    """
    Class which contains a red-sequence parametrization

    parameters
    ----------
    filename: string
        red sequence parameter file
    zbinsize: float, optional
        interpolation bin size in redshift
    minsig: float, optional
        minimum intrinsic scatter.  Default 0.01
    fine: bool, optional
        finest binning, set in filename header.  Default False
    zrange: float*2, optional
        redshift range [zlo,zhi].  Default from filename header
        (maximum range)

    """

    def __init__(self, filename, zbinsize=None, minsig=0.01, fine=False, zrange=None,
                 mstarpath=None):

        pars, hdr = fitsio.read(filename, ext=1, header=True)
        try:
            limmag = hdr['LIMMAG']
            if (zrange is None):
                zrange = np.array([hdr['ZRANGE0'], hdr['ZRANGE1']])
            alpha = hdr['ALPHA']
            mstar_survey = hdr['MSTARSUR']
            mstar_band = hdr['MSTARBAN']
            ncol = hdr['NCOL']
        except:
            raise ValueError("Missing field from parameter header.")

        if len(zrange) != 2:
            raise ValueError("zrange must have 2 elements")

        if zbinsize is None:
            try:
                if fine:
                    zbinsize = hdr['ZBINFINE']
                else:
                    zbinsize = hdr['ZBINCOAR']
            except:
                raise ValueError("Missing field from parameter header.")

        try:
            lowzmode = hdr['LOWZMODE']
        except:
            lowzmode = 0

        nmag = ncol + 1
        self.nmag = nmag

        bvalues = np.zeros(nmag)
        do_lupcorr = False
        try:
            for i in range(nmag):
                bvalues[i] = hdr['BVALUE%1d' % (i + 1)]
            do_lupcorr = True
        except:
            bvalues[:] = 0.0

        try:
            ref_ind = hdr['REF_IND']
        except:
            try:
                ref_ind = hdr['I_IND']
            except:
                raise ValueError("Need REF_IND or I_IND")

        nz = np.round((zrange[1] - zrange[0]) /
                      zbinsize).astype('i4')  # nr of bins
        self.z = zbinsize * np.arange(nz) + zrange[0]  # z bins

        # append a high-end overflow bin
        # for computation this will be the same as the top bin, and at the end
        # we set it to some absurdly large number.
        self.z = np.append(self.z, self.z[self.z.size - 1])
        nz = nz + 1

        self.zbinsize = zbinsize
        self.zbinscale = int(1. / zbinsize)

        ms = MStar(mstar_survey, mstar_band, mstarpath)

        refmagbinsize = 0.01
        if (lowzmode):
            refmagrange = np.array([10.0, limmag], dtype='f4')
            lumrefmagrange = np.array(
                [10.0, ms(zrange[1]) - 2.5 * np.log10(0.1)])
        else:
            refmagrange = np.array([12.0, limmag], dtype='f4')
            lumrefmagrange = np.array(
                [12.0, ms(zrange[1]) - 2.5 * np.log10(0.1)])
        self.refmagbins = np.arange(
            refmagrange[0], refmagrange[1], refmagbinsize, dtype='f8')
        self.lumrefmagbins = np.arange(
            lumrefmagrange[0], lumrefmagrange[1], refmagbinsize, dtype='f8')

        # and for fast look-ups...
        self.refmagbins = np.append(
            self.refmagbins, self.refmagbins[self.refmagbins.size - 1])
        self.lumrefmagbins = np.append(
            self.lumrefmagbins, self.lumrefmagbins[self.lumrefmagbins.size - 1])

        self.refmagbinsize = refmagbinsize
        self.refmagbinscale = int(1. / refmagbinsize)
        self.refmaginteger = (
            self.refmagbins * self.refmagbinscale).astype(np.int64)
        self.lumrefmaginteger = (
            self.lumrefmagbins * self.refmagbinscale).astype(np.int64)

        # is this an old or new structure?
        if 'PIVOTMAG_Z' in pars.dtype.names:
            refmag_name = 'REFMAG'
            pivotmag_name = 'PIVOTMAG'
        else:
            refmag_name = 'IMAG'
            pivotmag_name = 'REFMAG'

        # mark the extrapolated values
        self.extrapolated = np.zeros(nz, dtype=np.bool_)
        loz, = np.where(self.z < np.min(pars[pivotmag_name + '_Z']))
        hiz, = np.where(self.z > np.max(pars[pivotmag_name + '_Z']))
        if (loz.size > 0):
            self.extrapolated[loz] = True
        if (hiz.size > 0):
            self.extrapolated[hiz] = True

        # set the pivotmag
        spl = CubicSpline(pars[0][pivotmag_name + '_Z'],
                          pars[0][pivotmag_name])
        self.pivotmag = spl(self.z)

        # c/slope
        self.c = np.zeros((nz, ncol), dtype=np.float64)
        self.slope = np.zeros((nz, ncol), dtype=np.float64)
        for j in range(ncol):
            jstring = '%02d' % (j)
            spl = CubicSpline(pars[0]['Z' + jstring], pars[0]['C' + jstring])
            self.c[:, j] = spl(self.z)
            spl = CubicSpline(pars[0]['ZS' + jstring],
                              pars[0]['SLOPE' + jstring])
            self.slope[:, j] = spl(self.z)

        # sigma/covmat
        self.sigma = np.zeros((ncol, ncol, nz), dtype=np.float64)
        self.covmat = np.zeros((ncol, ncol, nz), dtype=np.float64)

        # diagonals
        for j in range(ncol):
            spl = CubicSpline(pars[0]['COVMAT_Z'], pars[0]['SIGMA'][j, j, :])
            self.sigma[j, j, :] = spl(self.z)

            self.covmat[j, j, :] = self.sigma[j, j, :] * self.sigma[j, j, :]

        # off-diagonals
        for j in range(ncol):
            for k in range(j + 1, ncol):
                spl = CubicSpline(pars[0]['COVMAT_Z'],
                                  pars[0]['SIGMA'][j, k, :])
                self.sigma[j, k, :] = spl(self.z)

                too_high, = np.where(self.sigma[j, k, :] > 0.9)
                if (too_high.size > 0):
                    self.sigma[j, k, too_high] = 0.9
                too_low, = np.where(self.sigma[j, k, :] < -0.9)
                if (too_low.size > 0):
                    self.sigma[j, k, too_low] = -0.9

                self.sigma[k, j, :] = self.sigma[j, k, :]

                self.covmat[j, k, :] = self.sigma[k, j, :] * \
                    self.sigma[j, j, :] * self.sigma[k, k, :]
                self.covmat[k, j, :] = self.covmat[j, k, :]

        # volume factor
        spl = CubicSpline(pars[0]['VOLUME_FACTOR_Z'], pars[0]['VOLUME_FACTOR'])
        self.volume_factor = spl(self.z)

        # corrections
        spl = CubicSpline(pars[0]['CORR_Z'], pars[0]['CORR'])
        self.corr = spl(self.z)
        spl = CubicSpline(pars[0]['CORR_SLOPE_Z'], pars[0]['CORR_SLOPE'])
        self.corr_slope = spl(self.z)

        spl = CubicSpline(pars[0]['CORR_Z'], pars[0]['CORR2'])
        self.corr2 = spl(self.z)
        spl = CubicSpline(pars[0]['CORR_SLOPE_Z'], pars[0]['CORR2_SLOPE'])
        self.corr2_slope = spl(self.z)

        if 'CORR_R' in pars.dtype.names:
            # protect against stupidity
            if (pars[0]['CORR_R'][0] <= 0.0):
                self.corr_r = np.ones(nz)
            else:
                spl = CubicSpline(pars[0]['CORR_SLOPE_Z'], pars[0]['CORR_R'])
                self.corr_r = spl(self.z)

            test, = np.where(self.corr_r < 0.5)
            if (test.size > 0):
                self.corr_r[test] = 0.5

            if (pars[0]['CORR2_R'][0] <= 0.0):
                self.corr2_r = np.ones(nz)
            else:
                spl = CubicSpline(pars[0]['CORR_SLOPE_Z'], pars[0]['CORR2_R'])
                self.corr2_r = spl(self.z)

            test, = np.where(self.corr2_r < 0.5)
            if (test.size > 0):
                self.corr2_r[test] = 0.5

        else:
            self.corr_r = np.ones(nz)
            self.corr2_r = np.ones(nz)

        # mstar
        # create LUT
        self._mstar = ms(self.z)

        # luminosity function integrations
        self.lumnorm = np.zeros((self.lumrefmagbins.size, nz))
        self.alpha = alpha
        for i in range(nz):
            f = 10.**(0.4 * (self.alpha + 1.0) * (self._mstar[i] - self.lumrefmagbins)) * np.exp(-10.**(
                0.4 * (self._mstar[i] - self.lumrefmagbins)))
            self.lumnorm[:, i] = refmagbinsize * np.cumsum(f)

        # lupcorr (annoying!)
        self.lupcorr = np.zeros((self.refmagbins.size, nz, ncol), dtype='f8')
        if (do_lupcorr):
            bnmgy = bvalues * 1e9

            for i in range(nz):
                mags = np.zeros((self.refmagbins.size, nmag))
                lups = np.zeros((self.refmagbins.size, nmag))

                mags[:, ref_ind] = self.refmagbins

                # go redward
                for j in range(ref_ind + 1, nmag):
                    mags[:, j] = mags[:, j - 1] - (self.c[i, j - 1] + self.slope[i, j - 1] * (
                        mags[:, ref_ind] - self.pivotmag[i]))
                # blueward
                for j in range(ref_ind - 1, -1, -1):
                    mags[:, j] = mags[:, j + 1] + \
                        (self.c[i, j] + self.slope[i, j] *
                         (mags[:, ref_ind] - self.pivotmag[i]))

                # and the luptitude conversion
                for j in range(nmag):
                    flux = 10.**((mags[:, j] - 22.5) / (-2.5))
                    lups[:, j] = 2.5 * np.log10(1.0 / bvalues[j]) - np.arcsinh(
                        0.5 * flux / bnmgy[j]) / (0.4 * np.log(10.0))

                magcol = mags[:, 0:ncol] - mags[:, 1:ncol + 1]
                lupcol = lups[:, 0:ncol] - lups[:, 1:ncol + 1]

                self.lupcorr[:, i, :] = lupcol - magcol

        # set top overflow bins to very large number
        self.z[self.z.size - 1] = 1000.0
        self.zinteger = np.round(self.z * self.zbinscale).astype(np.int64)
        self.refmagbins[self.refmagbins.size - 1] = 1000.0
        self.refmaginteger = np.round(
            self.refmagbins * self.refmagbinscale).astype(np.int64)
        self.lumrefmagbins[self.lumrefmagbins.size - 1] = 1000.0
        self.lumrefmaginteger = np.round(
            self.lumrefmagbins * self.refmagbinscale).astype(np.int64)
        self.ncol = ncol
        self.alpha = alpha
        self.mstar_survey = mstar_survey
        self.mstar_band = mstar_band
        self.limmag = limmag

        # don't make this into a catalog
        #super(RedSequenceColorPar, self).__init__(zredstr)

    def mstar(self, z):
        """
        M_star lookup

        parameters
        ----------
        z: array of floats
           redshift

        returns
        -------
        mstar: array of floats

        """
        # lookup and return mstar...awesome.
        zind = self.zindex(z)
        return self._mstar[zind]

    def zindex(self, z):
        """
        redshift index lookup

        parameters
        ----------
        z: array of floats

        returns
        -------
        indices: array of integers
            redshift indices

        """
        # return the z index/indices with rounding.

        zind = np.searchsorted(self.zinteger, np.round(
            np.atleast_1d(z) * self.zbinscale).astype(np.int64))
        if (zind.size == 1):
            return np.asscalar(zind)
        else:
            return zind

        # and check for top overflows.  Minimum is always 0
        #test,=np.where(zind == self.z.size)
        #if (test.size > 0): zind[test] = self.z.size-1

    def refmagindex(self, refmag):
        """
        reference magnitude index lookup

        parameters
        ----------
        refmag: array of floats

        returns
        -------
        indices: array of integers
            refmag indices
        """
        # return the refmag index/indices with rounding

        refmagind = np.searchsorted(self.refmaginteger, np.round(
            np.atleast_1d(refmag) * self.refmagbinscale).astype(np.int64))
        if (refmagind.size == 1):
            return np.asscalar(refmagind)
        else:
            return refmagind

    def lumrefmagindex(self, lumrefmag):
        """
        luminosity reference magnitude index lookup

        parameters
        ----------
        lumrefmag: array of floats

        returns
        -------
        indices: array of integers
            lumrefmag indices

        """

        lumrefmagind = np.searchsorted(self.lumrefmaginteger, np.round(
            np.atleast_1d(lumrefmag) * self.refmagbinscale).astype(np.int64))
        if (lumrefmagind.size == 1):
            return np.asscalar(lumrefmagind)
        else:
            return lumrefmagind


def add_red_sequence_noise(buzzard_rs_model, data_rs_model, g, nbands, rs_mult=1.0):

    ds = np.zeros((len(g), nbands - 1))
    isred = (g['AMAG'][:, 0] - g['AMAG'][:, 1]) > (0.095 - 0.035 * g['AMAG'][:, 1])

    for i in range(nbands - 1):
        zidx = buzzard_rs_model.z.searchsorted(g['Z'])
        ds[:, i] = data_rs_model.sigma[i, i, zidx]**2 - buzzard_rs_model.sigma[i, i, zidx]**2
        ds[ds[:, i] < 0, i] = 0
        ds[:, i] = np.sqrt(ds[:, i])

    for i in range(nbands - 2):
        ds[:, i] = ds[:, i]**2 - ds[:, i + 1]**2
        ds[ds[:, i] < 0, i] = 0
        ds[:, i] = np.sqrt(ds[:, i])

    mag = np.copy(g['TMAG'])

    for i in range(nbands - 1):
        mag[isred, i] += rs_mult * ds[isred, i] * np.random.randn(len(g[isred]))

    g['TMAG'] = mag
    for im in range(nbands):
        g['LMAG'][:, im] = g['TMAG'][:, im] - 2.5 * np.log10(g['MU'])

    return g


def shift_mean_red_sequence(buzzard_rs_model, data_rs_model, g, nbands, rs_mult=1.0):

    dm = np.zeros((len(g['Z']), nbands - 1))

    isred = (g['AMAG'][:, 0] - g['AMAG'][:, 1]) > (0.095 - 0.035 * g['AMAG'][:, 1])

    for i in range(nbands - 1):
        zidx = buzzard_rs_model.z.searchsorted(g['Z'])
        dm[:, i] = data_rs_model.c[zidx, i] - buzzard_rs_model.c[zidx, i]

    mag = np.copy(g['TMAG'])

    for i in range(nbands - 1):
        mag[isred, i] += dm[isred, i]

    g['TMAG'] = mag
    for im in range(nbands):
        g['LMAG'][:, im] = g['TMAG'][:, im] - 2.5 * np.log10(g['MU'])

    return g


def shift_mean_and_add_red_sequence_noise(buzzard_rs_model, data_rs_model, g, nbands, rs_mult=1.0):

    ds = np.zeros((len(g['Z']), nbands - 1))
    dm = np.zeros((len(g['Z']), nbands - 1))

    isred = (g['AMAG'][:, 0] - g['AMAG'][:, 1]) > (0.095 - 0.035 * g['AMAG'][:, 1])

    for i in range(nbands - 1):
        zidx = buzzard_rs_model.z.searchsorted(g['Z'])
        dm[:, i] = data_rs_model.c[zidx, i] - buzzard_rs_model.c[zidx, i]

    for i in range(nbands - 1):
        zidx = buzzard_rs_model.z.searchsorted(g['Z'])
        ds[:, i] = data_rs_model.sigma[i, i, zidx]**2 - buzzard_rs_model.sigma[i, i, zidx]**2
        ds[ds[:, i] < 0, i] = 0
        ds[:, i] = np.sqrt(ds[:, i])

    for i in range(nbands - 2):
        ds[:, i] = ds[:, i]**2 - ds[:, i + 1]**2
        ds[ds[:, i] < 0, i] = 0
        ds[:, i] = np.sqrt(ds[:, i])

    mag = np.copy(g['TMAG'])

    for i in range(nbands - 1):
        mag[isred, i] += dm[isred, i] + rs_mult * ds[isred, i] * np.random.randn(len(g['Z'][isred]))

    g['TMAG'] = mag
    for im in range(nbands):
        g['LMAG'][:, im] = g['TMAG'][:, im] - 2.5 * np.log10(g['MU'])

    return g


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    filepath = sys.argv[1]
    buzzard_rs_model = sys.argv[2]
    data_rs_model = sys.argv[3]
    nbands = int(sys.argv[4])
    mstarpath = sys.argv[5]

    if len(sys.argv) > 6:
        scatflag = int(sys.argv[6])
    else:
        scatflag = 0

    if scatflag == 0:
        add_scatter = True
        shift_mean = False
    elif scatflag == 1:
        add_scatter = False
        shift_mean = True
    else:
        add_scatter = True
        shift_mean = True

    if len(sys.argv) > 6:
        rs_mult = float(sys.argv[6])
    else:
        rs_mult = 1.0

    files = glob(filepath)

    buzzard_rs_model = RedSequenceColorPar(buzzard_rs_model, mstarpath=mstarpath)
    data_rs_model = RedSequenceColorPar(data_rs_model, mstarpath=mstarpath)

    files = files[rank::size]

    for i in range(len(files)):
        if add_scatter & (not shift_mean):
            ofile = files[i].replace('lensed', 'lensed_rs_scat')
        elif (not add_scatter) & shift_mean:
            ofile = files[i].replace('lensed', 'lensed_rs_shift')
        else:
            ofile = files[i].replace('lensed', 'lensed_rs_scat_shift')

        if os.path.exists(ofile):
            continue
        g = fitsio.read(files[i])

        if add_scatter & (not shift_mean):
            g = add_red_sequence_noise(buzzard_rs_model, data_rs_model, g, nbands, rs_mult=rs_mult)
        elif (not add_scatter) & shift_mean:
            g = shift_mean_red_sequence(buzzard_rs_model, data_rs_model, g, nbands, rs_mult=rs_mult)
        else:
            g = shift_mean_and_add_red_sequence_noise(buzzard_rs_model, data_rs_model, g, nbands, rs_mult=rs_mult)

        fitsio.write(ofile, g)

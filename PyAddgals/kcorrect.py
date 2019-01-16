from scipy.interpolate import InterpolatedUnivariateSpline as ius
from numba import jit
from . import config
import numpy as np
import os


@jit(nopython=True)
def k_reconstruct_maggies(rmatrix, coeffs, z, zvals):
    """Use the template projection matrix to calculate fluxes in maggies
    from a set of galaxy kcorrect coefficients and redshifts.

    Parameters
    ----------
    rmatrix : np.array
        Projection matrix containing projections of the templates onto
        the desired filters for a range of redshifts.
    coeffs : np.array
        Array of kcorrect coefficients. Shape of (n_gal, n_template)
    z : np.array
        Array of redshifts. Shape of (n_gal)
    zvals : np.array
        Redshifts at which the templates have been projected onto
        the filters
    Returns
    -------
    maggies : np.array
        Array of maggies of shape (n_gal, nk) where nk is the number of
        filters.

    """

    n_gal = z.size
    nk = rmatrix.shape[2]
    maggies = np.zeros((n_gal, nk))

    for i in range(n_gal):
        for k in range(nk):
            idx = np.searchsorted(zvals, z[i])
            maggies[i, k] = np.sum(rmatrix[idx, :, k] * coeffs[i, :])

    return maggies


class KCorrect(object):

    def __init__(self, minz=0.0, maxz=3.0, nz=1500, template_name='default'):
        """Initialize KCorrect object

        Parameters
        ----------
        minz : float
            Minimum redshift to compute template projections for.
        maxz : float
            Maximum redshift to compute template projections for.
        nz : int
            Number of redshifts to compute template projections for.
        template_name : str
            Name of template set.

        Returns
        -------
        None

        """

        self.zvals = np.linspace(minz, maxz, nz)
        self.c = 2.99792e+18  # speed of light in angstroms per sec
        self.abfnu = 3.631e-20  # AB system normalization flux density
        self.template_name = template_name
        self.load_templates()

    def load_templates(self):
        """Load a template set as specified by self.template_name.

        Returns
        -------
        None

        """
        # Get base directory containing all templates
        template_dir = '{}/data/templates/'.format(os.path.dirname(config.__file__))

        self.templates = np.genfromtxt('{}/vmatrix.{}.dat'.format(template_dir, self.template_name),
                                       skip_header=1).reshape(5, 10000)
        self.template_lambda = np.genfromtxt('{}/lambda.{}.dat'.format(template_dir, self.template_name),
                                             skip_header=1)

    def load_filters(self, filter_names):
        """Load a list of filters.

        Parameters
        ----------
        filter_names : list
            List of nk filter names

        Returns
        -------
        filter_lambda : list
            List of nk sets of filter wavelengths in angstroms
        filter_pass : list
            List of nk sets of filter transmissions
        """

        # get base directory containing all filters
        filter_dir = '{}/data/filters/'.format(os.path.dirname(config.__file__))
        filter_names = ['{}/{}'.format(filter_dir, f) for f in filter_names]

        filter_lambda = []
        filter_pass = []

        for i, fn in enumerate(filter_names):
            fl, fp = self.read_filter(fn)
            filter_lambda.append(fl)
            filter_pass.append(fp)

        return filter_lambda, filter_pass

    def read_filter(self, filename):
        """Read a filter in yanny file format.

        Parameters
        ----------
        filename : string
            The file name of the filter to be read.

        Returns
        -------
        filter_lambda : np.array
            Array of wavelengths in angstroms the filter is detemined at.
        filter_pass : np.array
            Array of filter transmissions at the wavelengths in filter_lambda
        """

        with open(filename, 'r') as fp:
            line = fp.readline()
            header = False
            headercount = 0
            passcol = 0
            filter_lambda = []
            filter_pass = []
            key = None

            while True:

                if not line:
                    break

                ls = line.split()
                if (line[0] == '#'):
                    line = fp.readline()

                elif 'typedef' in line:
                    header = True
                    headercount += 1

                elif header & ('}' not in line):
                    if 'pass;' in line:
                        passcol = headercount
                    if 'lambda;' in line:
                        lambdacol = headercount
                    headercount += 1

                elif header:
                    key = ls[-1].split(';')[0]
                    header = False

                elif key and (len(ls) > 0) and (key in ls[0]):

                    filter_lambda.append(np.float(ls[lambdacol]))
                    filter_pass.append(np.float(ls[passcol]))

                line = fp.readline()

        return np.array(filter_lambda), np.array(filter_pass)

    def zero_pad(self, template_lambda, filter_lambda, filter_pass):
        """Pad filter pass with zeros so that it spans the
        same range in wavelength as the template set.

        Parameters
        ----------
        template_lambda : np.array
            Wavelengths that the template is defined at.
        filter_lambda : np.array
            Wavelengths that the filter is defined at.
        filter_pass : np.array
            Filter transmission.

        Returns
        -------
        filter_lambda_zp : np.array
            Wavelengths of zero padded filter. These now extend
            from the minimum to the maximum wavelength of template_lambda
            but the wavelengths where the filter is defined are left
            unchanged.

        filter_pass_zp : np.array
            Zero padded transmissions of the filter. This is zero
            for wavelengths not defined by the filter.

        """

        nlf = filter_lambda.shape[0]

        filter_lambda_zp = np.zeros(3 * nlf)
        filter_pass_zp = np.zeros(3 * nlf)

        filter_lambda_zp[:nlf] = np.linspace(
            template_lambda[0], filter_lambda[0] - 1, nlf)
        filter_lambda_zp[nlf:2 * nlf] = filter_lambda
        filter_lambda_zp[2 * nlf:] = np.linspace(
            filter_lambda[-1] + 1, template_lambda[-1], nlf)

        filter_pass_zp[nlf:2 * nlf] = filter_pass

        return filter_lambda_zp, filter_pass_zp

    def k_projection_table(self, filter_pass, filter_lambda, band_shift):
        """Calculate the projection of the templates onto the set of filters
        shifted by a range of redshifts, so that we can easily use kcorrect
        coefficients to calculate magnitudes for each galaxy.

        Parameters
        ----------
        filter_pass : list
            List of arrays containing filter transmissions
        filter_lambda : list
            List of arrays containing filter wavelengths
        band_shift : list
            List of shifts to apply to filters.

        Returns
        -------
        rmatrix : np.array
            Array of template projectsion of shape (nz, nv, nk) where nz is
            the number of redshifts in self.zvals, nv is the number of templates
            and nk is the number of templates.

        """

        nk = len(filter_pass)  # number of filters
        nv = self.templates.shape[0]  # number of templates
        nz = self.zvals.shape[0]  # number of redshifts

        templates = self.templates
        template_lambda = self.template_lambda
        zvals = self.zvals

        rmatrix = np.zeros((nz, nv, nk))

        template_lambda_mean = (template_lambda[1:] + template_lambda[:-1]) / 2
        dlambda = template_lambda[1:] - template_lambda[:-1]

        for k in range(nk):

            filter_lambda_k, filter_pass_k = self.zero_pad(
                template_lambda, filter_lambda[k], filter_pass[k])

            filter_pass_spline = ius(
                filter_lambda_k / (1 + band_shift[k]), filter_pass_k, k=1)
            filter_pass_interp = filter_pass_spline(template_lambda_mean)
            scale = 1 / np.sum(filter_pass_interp *
                               dlambda * self.abfnu * self.c / template_lambda_mean)

            for i in range(nz):
                z = zvals[i]
                filter_pass_interp = filter_pass_spline(
                    template_lambda_mean * (1 + z))

                for v in range(nv):
                    rmatrix[i, v, k] = scale * np.sum(template_lambda_mean * (1 + z) *
                                                      filter_pass_interp * dlambda *
                                                      templates[v, :], axis=-1)

        return rmatrix

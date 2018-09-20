import numpy as np
from numba import jit


#@jit(nopython=True)
def sampleConditionalGMM(fvec, idx, idx_c, lil, predmat, featcov, ifeatcov, mu,
                         weights, n_components, n_feat, npred):
    """Condition the model based on known values for some
    features.

    Parameters
    ----------
    fvec : array_like (optional), shape = (n_condition, )
        An array of input values. Inputs set to NaN are not set, and
        become features to the resulting distribution. Order is
        preserved. Either an X array or an X dictionary is required
        (default=None).

    Returns
    -------
        samp: np.array
            A sample from the conditioned GMM.
    """
    mud = np.atleast_2d(np.zeros((n_components, n_feat)))
    cmo = np.atleast_2d(np.zeros((n_components, npred)))
    condMeans = np.zeros((n_components, npred))

    for i in range(n_components):
        mud[i, :] = fvec - mu[i, idx]

    for i in range(n_components):
        cmo[i, :] = np.dot(lil[i, :, :], mud[i, :])

    for i in range(n_components):
        condMeans[i, :] = mu[i, idx_c] + cmo[i, :]

    fsamples = np.zeros(n_components)

    for i in range(n_components):
        fsamples[i] = (np.exp(-0.5 * np.dot((fvec - mu[i, idx]).T,
                                            np.dot(ifeatcov[i], (fvec - mu[i, idx])))) /
                       np.sqrt((2 * np.pi)**n_feat * np.linalg.det(featcov[i])))

    condWeights = np.array([weights[i] * fsamples[i]
                            for i in range(n_components)])
    condWeights = condWeights / np.sum(condWeights)

    # Sample from conditional distribution
    # first select the component to associate input with
    weightCDF = np.cumsum(condWeights)
    rand = np.random.random()
    comp = np.searchsorted(weightCDF, rand)

    # Sample component associated with an input feature

    rand = np.random.randn(npred)

    samp = np.dot(predmat[comp], rand.T) + condMeans[comp]
    return samp


class GMM(object):

    def __init__(self, n_components=1,
                 cov=None, mu=None, weights=None):

        self.n_components = n_components
        self.cov = cov
        self.icov = np.linalg.inv(self.cov)
        self.mu = mu
        self.weights = weights

    def calculateConditionalCovs(self, idx):
        """Computes the ingredients generic across different conditionings
        that are needed for sampling from conditional GMMs.

        Parameters
        ----------
        idx : np.array
            Boolean array defining indices that are to be conditioned on.

        Returns
        -------
        None
        """

        idx_c = np.where(~np.array(idx))
        idx = np.where(np.array(idx))

        x, y, z = np.meshgrid(np.arange(self.n_components),
                              idx, idx, indexing='ij')
        x_c, y_c, z_c = np.meshgrid(
            np.arange(self.n_components), idx_c, idx_c, indexing='ij')
        x_cr, y_cr, z_cr = np.meshgrid(
            np.arange(self.n_components), idx_c, idx, indexing='ij')
        self.featcov = self.cov[x, y, z]
        self.ifeatcov = np.linalg.inv(self.featcov)
        temp = self.cov[x_cr, y_cr, z_cr]
        predcov = self.cov[x_c, y_c, z_c]

        self.predcov = predcov - np.array([np.dot(temp[i], np.dot(np.linalg.inv(
            self.featcov[i]), temp[i].T)) for i in range(self.n_components)])
        self.lil = np.array([np.dot(temp[i, :, :], self.ifeatcov[i, :, :])
                             for i in range(self.n_components)])

        predeig, predeigvec = np.linalg.eig(self.predcov)

        self.predeig = np.array([np.diag(predeig[i, :])
                                 for i in range(self.n_components)])
        self.predeigvec = predeigvec
        self.predmat = np.linalg.cholesky(self.predcov)


class GMMShapes(object):

    def __init__(self, cosmo, n_components=None, cov_file=None, means_file=None,
                 weights_file=None, conditional_fields=None,
                 conditional_field_mean=None, conditional_field_std=None,
                 size_mean=None, size_std=None, epsilon_mean=None,
                 epsilon_std=None, modeltype=None):

        if n_components is None:
            raise(ValueError("GMMShapes needs to specify number of components"))

        if cov_file is None:
            raise(ValueError("GMMShapes needs to specify a covariance matrix file"))

        if means_file is None:
            raise(ValueError("GMMShapes needs to specify a means of GMM"))

        if weights_file is None:
            raise(ValueError("GMMShapes needs to specify set of compenent weights"))

        if conditional_fields is None:
            raise(ValueError("GMMShapes needs to specify the fields that shapes are conditioned on"))

        self.cosmo = cosmo
        self.n_components = n_components
        self.cov_file = cov_file
        self.means_file = means_file
        self.weights_file = weights_file
        self.conditional_fields = conditional_fields
        self.n_feat = len(self.conditional_fields)
        self.n_pred = 2

        self.conditional_field_mean = np.array(conditional_field_mean)
        self.conditional_field_std = np.array(conditional_field_std)

        self.size_mean = size_mean
        self.size_std = size_std

        self.epsilon_mean = epsilon_mean
        self.epsilon_std = epsilon_std

        self.cov = np.load(self.cov_file)
        self.means = np.load(self.means_file)
        self.weights = np.load(self.weights_file)

        self.gmm = GMM(n_components=self.n_components,
                       cov=self.cov, mu=self.means,
                       weights=self.weights)
        self.idx = np.zeros(len(self.conditional_fields) + 2, dtype=np.bool)
        self.idx[-2:] = False
        self.idx[:-2] = True

        self.gmm.calculateConditionalCovs(self.idx)

    def randomlyOrientedEllipticity(self, epsilon_norm):
        """Generate two angular components of ellipticity from
        an absolute ellipticity.

        Parameters
        ----------
        epsilon_norm : np.array
            absolute value of ellipticity for each galaxy, w/ shape of  (n_gal)

        Returns
        -------
        epsilon : np.array
            ra and dec components of ellipticity, shape of (n_gal, 2)

        """

        position_angle = np.random.uniform(size=len(epsilon_norm)) * np.pi * 2
        epsilon = np.zeros((len(position_angle), 2))

        epsilon[:, 0] = np.cos(position_angle) * epsilon_norm
        epsilon[:, 1] = np.sin(position_angle) * epsilon_norm

        return epsilon

    def angularSize(self, log_comoving_size, z):
        """Convert log comoving size to an angular size.

        Parameters
        ----------
        log_comoving_size : np.array
            Array of log10 size in comoving Mpc/h
        z : np.array
            Redshift (cosmological)

        Returns
        -------
        angular_size : np.array
            Array of angular sizes in arcsec
        """

        da = self.cosmo.angularDiameterDistance(z)
        angular_size = 10**log_comoving_size * 180 * 60**2 / np.pi / da

        return angular_size

    def sampleShapes(self, galaxies):

        X = np.zeros((len(galaxies['PX']), len(self.conditional_fields)))

        for i, f in enumerate(self.conditional_fields):
            if len(self.conditional_fields[i]) == 2:
                X[:, i] = galaxies[self.conditional_fields[i][0]][:, self.conditional_fields[i][1]]
            else:
                X[:, i] = galaxies[self.conditional_fields[i][0]]

        # first component should be size, second ellipticity
        shapes = self.sampleAll(X, self.conditional_field_mean,
                                self.conditional_field_std,
                                slice(0, self.n_feat),
                                slice(self.n_feat, self.n_feat + 2),
                                self.gmm.lil, self.gmm.predmat,
                                self.gmm.featcov, self.gmm.ifeatcov,
                                self.gmm.mu, self.gmm.weights,
                                self.n_components, self.n_feat,
                                self.n_pred)

        shapes[:, 0] = shapes[:, 0] * self.size_std + self.size_mean
        shapes[:, 1] = shapes[:, 1] * self.epsilon_std + self.epsilon_mean

        epsilon = self.randomlyOrientedEllipticity(shapes[:, 1])
        angular_size = self.angularSize(shapes[:, 0], galaxies['Z_COS'])

        return shapes[:, 0], angular_size, epsilon

    def sampleAll(self, X, Xmean, Xstd, idx, idx_c, lil, predmat, featcov,
                  ifeatcov, mu, weights, n_components, n_feat, npred):

        Xnorm = (X - Xmean[idx]) / Xstd[idx]
        Xnorm[Xnorm > 1.5] = 1.5

        samps = np.zeros((len(X), npred))

        for i in range(len(X)):
            samps[i, :] = sampleConditionalGMM(Xnorm[i, :], idx, idx_c, lil,
                                               predmat, featcov, ifeatcov, mu,
                                               weights, n_components,
                                               n_feat, npred)

        return samps

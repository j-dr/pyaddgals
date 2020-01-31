__all__ = ['load_projected_correlation', 'load_abundance_function', 'guess_rbins_from_rp', 'cov2err']

import os
import numpy as np
from scipy.optimize import minimize
from helpers.SimulationAnalysis import readHlist
from helpers.readGadgetSnapshot import readGadgetSnapshot
try:
    from halotools.sim_manager import TabularAsciiReader
    noht = False
except:
    noht = True
import fitsio

def _get_fname(s, cut):
    cut_str = '{0:g}'.format(cut)
    fn = s.format(cut_str)
    if os.path.isfile(fn):
        return fn
    cut_str = '{0:.1f}'.format(cut)
    fn = s.format(cut_str)
    if os.path.isfile(fn):
        return fn
    raise ValueError('`cut` value {0:g} is not found.'.format(cut))


def cov2err(wp_cov):
    """
    Get the diagnal error from a covarince matrix

    Parameters
    ----------
    wp_cov : 2-d ndarray
        The covarince matrix.

    Returns
    -------
    wp_err : 1-d ndarray
        diagnal error, i.e. sqrt(diag(wp_cov)).
    """
    return np.sqrt(np.diag(wp_cov))


def guess_rbins_from_rp(rp):
    """
    Guess the edges of the bins used in calculating coorelation functions
    from the rp values used in plots.

    Parameters
    ----------
    rp : 1-d ndarray
        rp values used in plots.

    Returns
    -------
    rbins : 1-d ndarray
        the edges of the bins.
    """
    r_log = np.log10(rp)
    n = len(r_log)
    obj_fn = lambda p: np.fabs(r_log - np.linspace(p[0], p[0]+p[1]*(n-1), n)).sum()
    p = minimize(obj_fn, [r_log[0], (r_log[-1]-r_log[0])/float(n-1)]).x
    return np.logspace(p[0]-p[1]*0.5, p[0]+p[1]*(n-0.5), n+1)


def load_projected_correlation(cut, cut_in='l', source='zehavi'):
    """
    Load the projected correlation function in Reddick et al. (2013).

    Parameters
    ----------
    cut : int or float
        The threshold used in the projected correlation.
        For a luminosity cut, the cut value should be in Mag*(-1).
        For a stellar-mass cut, the cut value should be log10(M_s).
    cut_in : {'l', 's'}, optional
        Whether the cut is in luminosity ('l') or stellar mass ('s').
    source : {'tinker', 'zehavi'}, optional
        If `cut_in` is 'l', one can further choose to use the data
        from Tinker or from Zehavi.

    Returns
    -------
    rp : 1-d ndarray
        rp values.
    wp : 1-d ndarray
        wp values.
    wp_cov : 1-d ndarray
        The len(wp) by len(wp) covariance matrix of wp.
    """
    base_dir = '/u/ki/rmredd/data/corr_wp'

    cut_in = str(cut_in).lower()
    if cut_in.startswith('l'):
        cut_in = 'l'
        if cut < 0:
            cut = -cut
    elif cut_in.startswith('s'):
        cut_in = 's'
        if cut > 100:
            cut = np.log10(cut)
    else:
        raise ValueError('`cut_in` should be either "l" (luminosity) or "s" (stellar mass).')

    source = str(source).lower()
    if source in ['zehavi', 'idit', 'iz']:
        if cut_in == 's':
            raise ValueError('Only luminosity cuts exist for Zehavi wp.')
        fn = _get_fname(base_dir+'/idit_final/wp_dr7_final_idit_{0}.dat', cut)
        rp, wp = np.loadtxt(fn, usecols=(0,1), unpack=True)
        fn = _get_fname(base_dir+'/idit_final/wp_covar_{0}.dat', cut)
        wp_cov = np.fromfile(fn, sep=' ')
    elif source in ['tinker', 'jeremy', 'jt']:
        if cut_in == 'l':
            fn = _get_fname(base_dir+'/tinker_sdss_wp/wp_{0}.dat', cut)
            rp, wp = np.loadtxt(fn, usecols=(0,1), unpack=True)
            fn = _get_fname(base_dir+'/tinker_sdss_wp/wp_covar_{0}.dat', cut)
            wp_cov = np.loadtxt(fn, usecols=(2,))
        else:
            fn = _get_fname(base_dir+'/tinker_sdss_wp_sm/wp_{0}_vlim.dat.txt', cut)
            rp, wp = np.loadtxt(fn, usecols=(0,2), unpack=True)
            fn = _get_fname(base_dir+'/tinker_sdss_wp_sm/wp_{0}_vlim.covar.txt', cut)
            wp_cov = np.loadtxt(fn, usecols=(2,))
    else:
        raise ValueError('`source` should be either "tinker" or "zehavi".')

    return rp, wp, wp_cov.reshape(len(wp), len(wp))


def load_abundance_function(proxy='l', sample_cut=18, \
        log_phi=True, flip_mag_sign=False, unpack=False):
    """
    Load the abundance (luminosity or stellar mass) function in Reddick et al. (2013).

    Parameters
    ----------
    proxy : {'l', 's'}, optional
        Whether to load the luminosity ('l') or stellar mass ('s') function.
    sample_cut : int or float, optional
        If `proxy` is 'l', one can further choose to the sample cut (in Mag*(-1))
        used in the luminosity function.
    log_phi : bool, optional
        Whether the returned values of phi are in log10 or not.
        Default is True.
    flip_mag_sign : bool, optional
        Whether the returned values of Magnitudes have an additional sign or not.
        Default is True.
    unpack : bool, optional
        Whether or not to unpack the returned abundnace function.
        Default is False.

    Returns
    -------
    af : 2-d ndarray
        The first column, af[:,0], is the proxy values.
        For luminosity, the proxy values are in Mag,
        or Mag*(-1) if `flip_mag_sign` is True.
        For stellar mass, the proxy values are in log10(M_s).
        The second column, af[:,1], is phi,
        in the format of Mag^{-1} (Mpc/h)^{-3}.
        If `log_phi` is True, the values will be in log10.
    """
    proxy = str(proxy).lower()
    if proxy.startswith('l'):
        if sample_cut < 0:
            sample_cut = -sample_cut
        fn = _get_fname('/u/ki/rmredd/data/lf/tinker/lf_jt_{0}.dat', sample_cut)
    elif proxy.startswith('s'):
        if sample_cut == 19:
            sample_cut = 9.8
        fn = _get_fname('/nfs/slac/g/ki/ki10/rmredd/sham_models/tinker_sm/smf_jt{0}_val.dat', sample_cut)
    else:
        raise ValueError('`proxy` should be either "l" (luminosity) or "s" (stellar mass).')

    af = np.loadtxt(fn, usecols=(0,1))
    af = af[af[:,1]>0]
    if log_phi:
        af[:,1] = np.log10(af[:,1])
    if proxy.startswith('l') and flip_mag_sign:
        af[:,0] *= -1
    elif proxy.startswith('s'):
        af = af[::-1]
    if unpack:
        return af[:,0], af[:,1]
    else:
        return af

def hlist2bin(hlistname, field_dict, outdir):

    reader = TabularAsciiReader(hlistname, field_dict)
    hs   = hlistname.split('/')
    hout = '{}/{}.fits.gz'.format(outdir, hs[-1])
    
    #return if file already exists
    if os.path.isfile(hout): return

    try:
        halos = reader.read_ascii()
        #readHlist(hlistname, fields)
    except Exception as e:
        print(e)
        print('****Cannot compress file: {0}****'.format(hlistname))
        return

    fitsio.write(hout, halos, compress='gzip', clobber=True)

#def downsample_snapshot(snapdir, downsample_frac=0.01):
#
#    blockfiles = glob('{}/snapshot*'.format(snapdir))
#
#    for bf in blockfiles:
#        downsample_block(bf, downsample_frac=downsample_frac)


#def downsample_block(blockfiles):

    
        
        

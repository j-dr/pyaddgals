from __future__ import print_function, division
from AbundanceMatching import AbundanceFunction, calc_number_densities, add_scatter, rematch, LF_SCATTER_MULT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def abundanceMatchSnapshot(proxy, scatter, lf, box_size,
                           minmag=-25., maxmag=10., debug=False,
                           figname=None):

    af = AbundanceFunction(lf['mag'], lf['phi'], ext_range=(
        minmag, maxmag), nbin=2000, faint_end_fit_points=6)

    # check the abundance function
    if debug:
        plt.clf()
        plt.semilogy(lf['mag'], lf['phi'])
        x = np.linspace(minmag, maxmag, 101)
        plt.semilogy(x, af(x))
        plt.savefig('abundance_fcn.png')

    # deconvolution and check results (it's a good idea to always check this)
    remainder = af.deconvolute(scatter * LF_SCATTER_MULT, 40)
    x, nd = af.get_number_density_table()

    if debug:
        plt.clf()
        plt.semilogy(x, np.abs(remainder / nd))
        plt.savefig('nd_remainder.png')

    # get number densities of the halo catalog
    nd_halos = calc_number_densities(proxy, box_size)

    # do abundance matching with some scatter
    catalog_sc = af.match(nd_halos, scatter * LF_SCATTER_MULT)

    if debug:
        plt.clf()
        c, e = np.histogram(catalog_sc[~np.isnan(
            catalog_sc)], bins=np.linspace(minmag, maxmag, 101))
        c = c / box_size ** 3 / (e[1:] - e[:-1])
        me = (e[:-1] + e[1:]) / 2
        plt.semilogy(me, c)
        plt.semilogy(me, af(me))
        plt.savefig('lf_in_v_out.png')

    return catalog_sc


def abundanceMatchSnapshotSMF(proxy, scatter, smf, box_size,
                              minmag=7, maxmag=13, debug=False,
                              figname=None):

    af = AbundanceFunction(smf['mag'], smf['phi'], ext_range=(
        maxmag, minmag), nbin=2000, faint_end_fit_points=6)

    # check the abundance function
    if debug:
        plt.clf()
        plt.semilogy(smf['mag'], smf['phi'])
        x = np.linspace(minmag, maxmag, 101)
        plt.semilogy(x, af(x))
        plt.savefig('abundance_fcn.png')

    # deconvolution and check results (it's a good idea to always check this)
    remainder = af.deconvolute(scatter, 40)
    x, nd = af.get_number_density_table()

    if debug:
        plt.clf()
        plt.semilogy(x, np.abs(remainder / nd))
        plt.savefig('nd_remainder.png')

    # get number densities of the halo catalog
    nd_halos = calc_number_densities(proxy, box_size)

    # do abundance matching with some scatter
    catalog_sc = af.match(nd_halos, scatter)

    if debug:
        plt.clf()
        c, e = np.histogram(catalog_sc[~np.isnan(
            catalog_sc)], bins=np.linspace(minmag, maxmag, 101))
        c = c / box_size ** 3 / (e[1:] - e[:-1])
        me = (e[:-1] + e[1:]) / 2
        plt.semilogy(me, c)
        plt.semilogy(me, af(me))
        plt.savefig('lf_in_v_out.png')

    return catalog_sc

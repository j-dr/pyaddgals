from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
import fitsio

import healpy as hp
from PyAddgals.config import parseConfig
from PyAddgals.cosmology import Cosmology
from PyAddgals.domain import Domain
from PyAddgals.nBody import NBody
from PyAddgals.addgalsModel import ADDGALSModel
from PyAddgals.kcorrect import KCorrect
import sys
import matplotlib

from Corrfunc.mocks import DDrppi_mocks
from Corrfunc.utils import convert_3d_counts_to_cf
import healpix_util as hu

fs = 20
afs = 20
linewidth = 4
markersize = 20

matplotlib.rc('font', size=fs)  # , family='serif', serif='Times')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rc('xtick', labelsize=afs)
matplotlib.rc('xtick.major', width=linewidth)
matplotlib.rc('xtick.minor', width=linewidth / 2)


matplotlib.rc('ytick', labelsize=afs)
matplotlib.rc('ytick.major', width=linewidth)
matplotlib.rc('ytick.minor', width=linewidth / 2)


matplotlib.rc('axes', labelsize=30, linewidth=linewidth / 2)
matplotlib.rc('legend', fontsize=afs)
matplotlib.rc('lines', linewidth=2, markersize=markersize)
matplotlib.rc('figure', figsize=[10, 10])


def calc_uniform_errors(tmag, maglims, exptimes, lnscat, zp=22.5):

    nmag = len(maglims)
    ngal = len(tmag)

    tmag = tmag.reshape(len(tmag), nmag)

    # calculate fsky1 -- sky in 1 second
    flux1_lim = 10**((maglims - zp) / (-2.5))
    flux1_lim[flux1_lim < 120 / exptimes] = 120 / \
        exptimes[flux1_lim < 120 / exptimes]
    fsky1 = (flux1_lim**2 * exptimes) / 100. - flux1_lim

    oflux = np.zeros((ngal, nmag))
    ofluxerr = np.zeros((ngal, nmag))
    omag = np.zeros((ngal, nmag))
    omagerr = np.zeros((ngal, nmag))
    oamag = np.zeros((ngal, nmag))
    offset = 0.0

    for i in range(nmag):
        tflux = exptimes[i] * 10**((tmag[:, i] - offset - zp) / (-2.5))
        noise = np.exp(np.log(np.sqrt(fsky1[i] * exptimes[i] + tflux))
                       + lnscat[i] * np.random.randn(ngal))

        flux = tflux + noise * np.random.randn(ngal)

        oflux[:, i] = flux / exptimes[i]
        ofluxerr[:, i] = noise / exptimes[i]

        oflux[:, i] *= 10 ** ((zp - 22.5) / -2.5)
        ofluxerr[:, i] *= 10 ** ((zp - 22.5) / -2.5)

        omag[:, i] = 22.5 - 2.5 * np.log10(oflux[:, i])
        omagerr[:, i] = (2.5 / np.log(10.)) * (ofluxerr[:, i] / oflux[:, i])

    bad, = np.where(~np.isfinite(omag[:, i]))
    nbad = len(bad)
    if (nbad > 0):
        omag[bad, i] = 99.0
        omagerr[bad, i] = 99.0

    return omag, omagerr, oflux, ofluxerr


def generateCartesianRandoms(x, y, z, rand_nside, rand_nest, rand_ra_lims, rand_dec_lims, rand_fact=10):
    rsize = len(x) * rand_fact

    if rand_nside:
        rdtype = np.dtype([('px', np.float32), ('py', np.float32),
                           ('pz', np.float32), ('redshift', np.float32)])

    else:
        rdtype = np.dtype([('px', np.float32), ('py', np.float32),
                           ('pz', np.float32)])

    gr = np.zeros(rsize, dtype=rdtype)

    if rand_nside is None:

        gr['px'] = np.random.uniform(low=np.min(x),
                                     high=np.max(x),
                                     size=rsize)
        gr['py'] = np.random.uniform(low=np.min(y),
                                     high=np.max(y),
                                     size=rsize)
        gr['pz'] = np.random.uniform(low=np.min(z),
                                     high=np.max(z),
                                     size=rsize)
    else:
        print('generating randoms using nside {}'.format(rand_nside))
        print('Nest ordering : {}'.format(rand_nest))
        r = np.sqrt(x**2 + y**2 + z**2)
        rr = np.random.choice(r, size=rsize)

        pix = hp.vec2pix(rand_nside, x, y, z, nest=rand_nest)
        pix = np.unique(pix)
        hmap = np.zeros(12 * rand_nside**2)

        hmap[pix] = 1

        if rand_nest:
            hmap = hu.DensityMap('nest', hmap)
        else:
            hmap = hu.DensityMap('ring', hmap)

        theta, phi = hmap.genrand(rsize, system='ang')

        if (rand_ra_lims is not None):
            ra = phi * 180. / np.pi
            idx = (rand_ra_lims[0] < ra) & (ra < rand_ra_lims[1])

            theta = theta[idx]
            phi = phi[idx]
            rr = rr[idx]
            gr = gr[idx]

            del ra, idx

        if (rand_dec_lims is not None):
            dec = 90. - theta * 180. / np.pi
            idx = (rand_dec_lims[0] < dec) & (dec < rand_dec_lims[1])
            print(gr.shape)
            theta = theta[idx]
            phi = phi[idx]
            rr = rr[idx]
            gr = gr[idx]

            del dec, idx
            print(gr.shape)

        pos = hp.ang2vec(theta, phi)
        pos *= rr.reshape(-1, 1)

        gr['px'] = pos[:, 0]
        gr['py'] = pos[:, 1]
        gr['pz'] = pos[:, 2]
        gr['redshift'] = rr

    return gr


def generateAngularRandoms(aza, pla, z=None, urand_factor=20,
                           rand_factor=10, nside=8, nest=True):
    """
    Generate a set of randoms from a catalog by pixelating the input
    catalog and uniformly distributing random points within the pixels
    occupied by galaxies.

    Also option to assign redshifts with the same distribution as the input
    catalog to allow for generation of randoms once for all z bins.
    """

    dt = aza.dtype

    if z is not None:
        rdtype = np.dtype([('azim_ang', dt), ('polar_ang', dt),
                           ('redshift', dt)])
    else:
        rdtype = np.dtype([('azim_ang', dt), ('polar_ang', dt)])

    rsize = len(aza) * urand_factor

    cpix = hp.ang2pix(nside, (90 - pla) * np.pi / 180.,
                      aza * np.pi / 180., nest=nest)
    cpix = np.unique(cpix)
    pmap = np.zeros(12 * nside ** 2)

    pmap[cpix] = 1
    if nest:
        pmap = hu.DensityMap('nest', pmap)
    else:
        pmap = hu.DensityMap('ring', pmap)

    grand = np.zeros(len(aza) * rand_factor, dtype=rdtype)
    grand['azim_ang'], grand['polar_ang'] = pmap.genrand(
        len(aza) * rand_factor, system='eq')

    if z is not None:
        grand['redshift'] = np.random.choice(z, size=len(aza) * rand_factor)
        zidx = grand['redshift'].argsort()
        grand = grand[zidx]

    return grand


def jackknife(njacktot, arg, reduce_jk=True):

    jdata = np.zeros(arg.shape)

    for i in range(njacktot):
        # generalized so can be used if only one region
        if njacktot == 1:
            idx = [0]
        else:
            idx = [j for j in range(njacktot) if i != j]

        # jackknife indices should always be first
        jl = len(arg.shape)
        jidx = [slice(0, arg.shape[j]) if j != 0 else idx for j in range(jl)]
        jdidx = [slice(0, arg.shape[j]) if j != 0 else i for j in range(jl)]
        jdata[jdidx] = np.sum(arg[jidx], axis=0)

    if reduce_jk:
        jest = np.sum(jdata, axis=0) / njacktot
        jvar = np.sum((jdata - jest)**2, axis=0) * (njacktot - 1) / njacktot

        return jdata, jest, jvar

    else:
        return jdata


def compute_wprp(ra, dec, z, rbins, nside=8, rand_nside=8, nest=True, pimax=80, nthreads=16):

    if nside == 0:
        pix = np.zeros(len(ra))
        upix = np.array([0])
        npix = 1
    else:
        pix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
        upix = np.unique(pix)
        npix = len(upix)

    cz = z * 299792.458

    dd = np.zeros((len(upix), len(rbins) - 1, pimax))
    dr = np.zeros((len(upix), len(rbins) - 1, pimax))
    rr = np.zeros((len(upix), len(rbins) - 1, pimax))
    nd = np.zeros((len(upix)))
    nr = np.zeros((len(upix)))

    print('npix: {}'.format(npix))

    for i, p in enumerate(upix):
        print('pixel {}'.format(i))
        idx = pix == p
        rands = generateAngularRandoms(
            ra[idx], dec[idx], cz[idx], rand_nside, nest=nest)

        nd[i] = len(ra[idx])
        nr[i] = len(rands)

        ddresults = DDrppi_mocks(1,
                                 1, nthreads,
                                 pimax, rbins,
                                 ra[idx], dec[idx],
                                 cz[idx])

        dd[i, :, :] = ddresults['npairs'].reshape(-1, int(pimax))

        drresults = DDrppi_mocks(0,
                                 1, nthreads,
                                 pimax, rbins,
                                 ra[idx], dec[idx],
                                 cz[idx], RA2=rands['azim_ang'],
                                 DEC2=rands['polar_ang'],
                                 CZ2=rands['redshift'])

        dr[i, :, :] = drresults['npairs'].reshape(-1, int(pimax))

        rrresults = DDrppi_mocks(1,
                                 1, nthreads,
                                 pimax, rbins,
                                 rands['azim_ang'],
                                 rands['polar_ang'],
                                 rands['redshift'])

        rr[i, :, :] = rrresults['npairs'].reshape(-1, int(pimax))

    jdd = jackknife(npix, dd, reduce_jk=False)
    jdr = jackknife(npix, dr, reduce_jk=False)
    jrr = jackknife(npix, rr, reduce_jk=False)
    jnd = jackknife(npix, nd, reduce_jk=False)
    jnr = jackknife(npix, nr, reduce_jk=False)

    fnorm = (jnr / jnd).reshape(-1, 1, 1)

    jwprppi = (fnorm ** 2 * jdd - 2 * fnorm * jdr + jrr) / jrr
    jwprp = 2 * np.sum(jwprppi, axis=2)

    wprppi = np.sum(jwprppi, axis=0) / npix
    wprp = np.sum(jwprp, axis=0) / npix

    varwprppi = np.sum((jwprppi - wprppi)**2, axis=0) * (npix - 1) / npix
    varwprp = np.sum((jwprp - wprp)**2, axis=0) * (npix - 1) / npix

    return wprp, wprppi, varwprp, varwprppi, dd, dr, rr, nd, nr


if __name__ == '__main__':

    config_file = sys.argv[1]
    filepath = sys.argv[2]
    valpath = sys.argv[3]

    files = glob(filepath)
    config = parseConfig(config_file)

    comm = None

    cc = config['Cosmology']
    nb_config = config['NBody']

    cosmo = Cosmology(**cc)

    domain = Domain(cosmo, **nb_config.pop('Domain'))
    domain.decomp(comm, 0, 1)

    for d in domain.yieldDomains():
        nbody = NBody(cosmo, d, **nb_config)
        break

    model = ADDGALSModel(nbody, **config['GalaxyModel']['ADDGALSModel'])

    for i, f in enumerate(files):
        print(i)
        sys.stdout.flush()
        gi = fitsio.read(f, columns=['PX', 'PY', 'PZ', 'TRA', 'TDEC',
                                     'Z', 'Z_COS', 'MAG_R', 'SEDID', 'TMAG', 'CENTRAL', 'M200'])
        zidx = (gi['Z'] < 0.25) & (gi['TMAG'][:, 1] < 20)

        if i == 0:
            pxo = gi['PX'][zidx]
            pyo = gi['PY'][zidx]
            pzo = gi['PZ'][zidx]
            zco = gi['Z_COS'][zidx]
            rao = gi['TRA'][zidx]
            deco = gi['TDEC'][zidx]
            zo = gi['Z'][zidx]
            mo = gi['MAG_R'][zidx]
            co = gi['SEDID'][zidx]
            cen = gi['CENTRAL'][zidx]
            mass = gi['M200'][zidx]

        else:
            pxo = np.hstack([pxo, gi['PX'][zidx]])
            pyo = np.hstack([pyo, gi['PY'][zidx]])
            pzo = np.hstack([pzo, gi['PZ'][zidx]])
            zco = np.hstack([zo, gi['Z_COS'][zidx]])
            rao = np.hstack([rao, gi['TRA'][zidx]])
            deco = np.hstack([deco, gi['TDEC'][zidx]])
            zo = np.hstack([zo, gi['Z'][zidx]])
            mo = np.hstack([mo, gi['MAG_R'][zidx]])
            co = np.hstack([co, gi['SEDID'][zidx]])
            cen = np.hstack([cen, gi['CENTRAL'][zidx]])
            mass = np.hstack([mass, gi['M200'][zidx]])

    kcorr = KCorrect()
    train = model.colorModel.trainingSet
    filters = ['sdss/sdss_u0.par', 'sdss/sdss_g0.par',
               'sdss/sdss_r0.par', 'sdss/sdss_i0.par', 'sdss/sdss_z0.par']
    omag, amag = model.colorModel.computeMagnitudes(
        mo, zo, train['COEFFS'][co], filters)

    train_omag = 22.5 - 2.5 * np.log10(train['ABMAGGIES'] / 1e-9)
    train_ug = train_omag[:, 0] - train_omag[:, 1]
    train_gr = train_omag[:, 1] - train_omag[:, 2]
    train_ri = train_omag[:, 2] - train_omag[:, 3]
    train_iz = train_omag[:, 3] - train_omag[:, 4]

    ug = omag[:, 0] - omag[:, 2]
    gr = omag[:, 1] - omag[:, 2]
    ri = omag[:, 2] - omag[:, 3]
    iz = omag[:, 3] - omag[:, 4]

    amagbins = np.linspace(-22, -19, 4)

    maglims = np.array([20.425, 21.749, 21.239, 20.769, 19.344])
    exptimes = np.array([21.00, 159.00, 126.00, 99.00, 15.00])
    lnscat = np.array([0.284, 0.241, 0.229, 0.251, 0.264])

    emag, emagerr, flux, fluxerr = calc_uniform_errors(
        omag, maglims, exptimes, lnscat)
    ug = emag[:, 0] - emag[:, 1]
    gr = emag[:, 1] - emag[:, 2]
    ri = emag[:, 2] - emag[:, 3]
    iz = emag[:, 3] - emag[:, 4]

    eamag = amag + np.random.randn(amag.shape[0], amag.shape[1]) * emagerr
    gr_rf = eamag[:, 1] - eamag[:, 2]
    isred = gr_rf > (0.21 - 0.03 * amag[:, 2])
    czo = zo * 299792.458

    wpz2221_rb = np.genfromtxt('{}/wp_dr7_final_idit_22_21_rb.dat'.format(valpath))
    wpz2120_rb = np.genfromtxt('{}/wp_dr7_final_idit_21_20_rb.dat'.format(valpath))
    wpz2019_rb = np.genfromtxt('{}/wp_dr7_final_idit_20_19_rb.dat'.format(valpath))

    wpz2221 = np.genfromtxt('{}/wp_dr7_final_idit_22_21.dat'.format(valpath))
    wpz2120 = np.genfromtxt('{}/wp_dr7_final_idit_21_20.dat'.format(valpath))
    wpz2019 = np.genfromtxt('{}/wp_dr7_final_idit_20_19.dat'.format(valpath))

    colors = ['r', 'b']

    rbins = np.logspace(-1, np.log10(50), 14)

    idx = (-22 < mo) & (mo < -21) & (19900 < czo) & (czo <
                                                     47650) & (14.5 < emag[:, 2]) & (emag[:, 2] < 17.6)
    idx_red = (-22 < mo) & (mo < -21) & (19900 < czo) & (czo < 47650) & isred & (14.5 < emag[:, 2]) & (emag[:, 2] < 17.6)
    idx_blue = (-22 < mo) & (mo < -21) & (19900 < czo) & (czo < 47650) & (~isred) & (14.5 < emag[:, 2]) & (emag[:, 2] < 17.6)

    rbins = np.logspace(-1, np.log10(50), 14)

    wprp_22, wprppi_22, varwprp, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx], deco[idx], zo[idx], rbins, nside=0, nest=True, pimax=60, nthreads=16)
    wprp_22jk, wprppi_22jk, varwprp_22, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx], deco[idx], zo[idx], rbins, nside=2, nest=True, pimax=60, nthreads=16)

    wprp_red22, wprppi_red22, varwprp, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_red], deco[idx_red], zo[idx_red], rbins, nside=0, nest=True, pimax=60, nthreads=16)
    wprp_red22jk, wprppi_red22jk, varwprp_red22, varwprppi_red22, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_red], deco[idx_red], zo[idx_red], rbins, nside=2, nest=True, pimax=60, nthreads=16)

    wprp_blue22, wprppi_blue22, varwprp, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_blue], deco[idx_blue], zo[idx_blue], rbins, nside=0, nest=True, pimax=60, nthreads=16)
    wprp_blue22jk, wprppi_blue22jk, varwprp_blue22, varwprppi_blue22, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_blue], deco[idx_blue], zo[idx_blue], rbins, nside=2, nest=True, pimax=60, nthreads=16)

    idx = (-21 < mo) & (mo < -20) & (19900 < czo) & (czo <
                                                     47650) & (14.5 < emag[:, 2]) & (emag[:, 2] < 20)

    idx_red = (-21 < mo) & (mo < -20) & (19900 < czo) & (czo <
                                                         47650) & isred & (14.5 < emag[:, 2]) & (emag[:, 2] < 20)
    idx_blue = (-21 < mo) & (mo < -20) & (19900 < czo) & (czo <
                                                          47650) & (~isred) & (14.5 < emag[:, 2]) & (emag[:, 2] < 20)

    wprp_21, wprppi_21, varwprp, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx], deco[idx], zo[idx], rbins, nside=0, nest=True, pimax=60, nthreads=16)
    wprp_21jk, wprppi_21jk, varwprp_21, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx], deco[idx], zo[idx], rbins, nside=2, nest=True, pimax=60, nthreads=16)

    wprp_red21, wprppi_red21, varwprp, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_red], deco[idx_red], zo[idx_red], rbins, nside=0, nest=True, pimax=60, nthreads=16)
    wprp_red21jk, wprppi_red21jk, varwprp_red21, varwprppi_red21, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_red], deco[idx_red], zo[idx_red], rbins, nside=2, nest=True, pimax=60, nthreads=16)

    wprp_blue21, wprppi_blue21, varwprp, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_blue], deco[idx_blue], zo[idx_blue], rbins, nside=0, nest=True, pimax=60, nthreads=16)
    wprp_blue21jk, wprppi_blue21jk, varwprp_blue21, varwprppi_blue21, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_blue], deco[idx_blue], zo[idx_blue], rbins, nside=2, nest=True, pimax=60, nthreads=16)

    idx = (-20 < mo) & (mo < -19) & (8050 < czo) & (czo <
                                                    50000) & (14.5 < emag[:, 2]) & (emag[:, 2] < 17.6)
    idx_red = (-20 < mo) & (mo < -19) & (8050 < czo) & (czo <
                                                        50000) & isred & (14.5 < emag[:, 2]) & (emag[:, 2] < 17.6)
    idx_blue = (-20 < mo) & (mo < -19) & (8050 < czo) & (czo <
                                                         50000) & (~isred) & (14.5 < emag[:, 2]) & (emag[:, 2] < 17.6)

    wprp_20, wprppi_20, varwprp, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx], deco[idx], zo[idx], rbins, nside=0, nest=True, pimax=60, nthreads=16)
    wprp_20jk, wprppi_20jk, varwprp_20, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx], deco[idx], zo[idx], rbins, nside=2, nest=True, pimax=60, nthreads=16)

    wprp_red20, wprppi_red20, varwprp, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_red], deco[idx_red], zo[idx_red], rbins, nside=0, nest=True, pimax=40, nthreads=16)
    wprp_red20jk, wprppi_red20jk, varwprp_red20, varwprppi_red20, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_red], deco[idx_red], zo[idx_red], rbins, nside=2, nest=True, pimax=40, nthreads=16)

    wprp_blue20, wprppi_blue20, varwprp, varwprppi, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_blue], deco[idx_blue], zo[idx_blue], rbins, nside=0, nest=True, pimax=40, nthreads=16)
    wprp_blue20jk, wprppi_blue20jk, varwprp_blue20, varwprppi_blue20, dd, dr, rr, nd, nr = compute_wprp(
        rao[idx_blue], deco[idx_blue], zo[idx_blue], rbins, nside=2, nest=True, pimax=40, nthreads=16)

    np.savetxt('{}/wprp_22_21.txt'.format(valpath), wprp_22)
    np.savetxt('{}/varwprp_22_21.txt'.format(valpath), varwprp_22)
    np.savetxt('{}/wprp_22_21_red.txt'.format(valpath), wprp_red22)
    np.savetxt('{}/varwprp_22_21_red.txt'.format(valpath), varwprp_red22)
    np.savetxt('{}/wprp_22_21_blue.txt'.format(valpath).format(valpath), wprp_blue22)
    np.savetxt('{}/varwprp_22_21_blue.txt'.format(valpath), varwprp_blue22)

    np.savetxt('{}/wprp_21_20.txt'.format(valpath), wprp_21)
    np.savetxt('{}/varwprp_21_20.txt'.format(valpath), varwprp_21)
    np.savetxt('{}/wprp_21_20_red.txt'.format(valpath), wprp_red21)
    np.savetxt('{}/varwprp_21_20_red.txt'.format(valpath), varwprp_red21)
    np.savetxt('{}/wprp_21_20_blue.txt'.format(valpath), wprp_blue21)
    np.savetxt('{}/varwprp_21_20_blue.txt'.format(valpath), varwprp_blue21)

    np.savetxt('{}/wprp_20_19.txt'.format(valpath), wprp_20)
    np.savetxt('{}/varwprp_20_19.txt'.format(valpath), varwprp_20)
    np.savetxt('{}/wprp_20_19_red.txt'.format(valpath), wprp_red20)
    np.savetxt('{}/varwprp_20_19_red.txt'.format(valpath), varwprp_red20)
    np.savetxt('{}/wprp_20_19_blue.txt'.format(valpath), wprp_blue20)
    np.savetxt('{}/varwprp_20_19_blue.txt'.format(valpath), varwprp_blue20)

    rmean = (rbins[1:] + rbins[:-1]) / 2

    f, ax = plt.subplots(2, 3, sharex=True, sharey='row')
    rmean = (rbins[1:] + rbins[:-1]) / 2
    amagbins = [-22, -21, -20, -19]

    ms = 3

    lb = ax[0][0].plot(rmean, wprp_red22)

    ax[0][0].fill_between(rmean, wprp_red22 - np.sqrt(varwprp_red22),
                          wprp_red22 + np.sqrt(varwprp_red22), color=colors[0], alpha=0.5)
    ls = ax[0][0].errorbar(wpz2221_rb[:, 0], wpz2221_rb[:, 1], wpz2221_rb[:, 3],
                           color='k', marker='o', linestyle='', markersize=ms)

    ax[0][0].plot(rmean, wprp_blue22, color=colors[-1])
    ax[0][0].fill_between(rmean, wprp_blue22 - np.sqrt(varwprp_blue22),
                          wprp_blue22 + np.sqrt(varwprp_blue22), color=colors[-1], alpha=0.5)
    ax[0][0].errorbar(wpz2221_rb[:, 0], wpz2221_rb[:, 2], wpz2221_rb[:, 4],
                      color='k', marker='o', linestyle='', markersize=ms, alpha=0.5)

    ax[0][0].set_yscale('log')
    ax[0][0].set_xscale('log')
    ax[0][0].set_xlim([0.1, 30])

    ax[0][1].plot(rmean, wprp_red21)

    ax[0][1].fill_between(rmean, wprp_red21 - np.sqrt(varwprp_red21),
                          wprp_red21 + np.sqrt(varwprp_red21), color=colors[0], alpha=0.5)
    ax[0][1].errorbar(wpz2120_rb[:, 0], wpz2120_rb[:, 1], wpz2120_rb[:, 3],
                      color='k', marker='o', linestyle='', markersize=ms)

    ax[0][1].plot(rmean, wprp_blue21, color=colors[-1])
    ax[0][1].fill_between(rmean, wprp_blue21 - np.sqrt(varwprp_blue21),
                          wprp_blue21 + np.sqrt(varwprp_blue21), color=colors[-1], alpha=0.5)
    ax[0][1].errorbar(wpz2120_rb[:, 0], wpz2120_rb[:, 2], wpz2120_rb[:, 4],
                      color='k', marker='o', linestyle='', markersize=ms, alpha=0.5)

    ax[0][1].set_yscale('log')
    ax[0][1].set_xscale('log')
    ax[0][1].set_xlim([0.1, 30])

    ax[0][2].plot(rmean, wprp_red20)

    ax[0][2].fill_between(rmean, wprp_red20 - np.sqrt(varwprp_red20),
                          wprp_red20 + np.sqrt(varwprp_red20), color=colors[0], alpha=0.5)
    ax[0][2].errorbar(wpz2019_rb[:, 0], wpz2019_rb[:, 1], wpz2019_rb[:, 3],
                      color='k', marker='o', linestyle='', markersize=ms)

    ax[0][2].plot(rmean, wprp_blue20, color=colors[-1])
    ax[0][2].fill_between(rmean, wprp_blue20 - np.sqrt(varwprp_blue20),
                          wprp_blue20 + np.sqrt(varwprp_blue20), color=colors[-1], alpha=0.5)
    ax[0][2].errorbar(wpz2019_rb[:, 0], wpz2019_rb[:, 2], wpz2019_rb[:, 4],
                      color='k', marker='o', linestyle='', markersize=ms, alpha=0.5)

    lb = ax[1][0].plot(rmean, wprp_red22 / wprp_22)
    ax[1][0].fill_between(rmean, wprp_red22 / wprp_22 - np.sqrt(varwprp_red22 / wprp_22**2),
                          wprp_red22 / wprp_22 + np.sqrt(varwprp_red22 / wprp_22**2), color=colors[0], alpha=0.5)
    ls = ax[1][0].errorbar(wpz2221_rb[:, 0], wpz2221_rb[:, 1] / wpz2221[:, 1],
                           wpz2221_rb[:, 3] / wpz2221[:, 1], color='k', marker='o', linestyle='', markersize=ms)

    ax[1][0].plot(rmean, wprp_blue22 / wprp_22, color=colors[-1])
    ax[1][0].fill_between(rmean, wprp_blue22 / wprp_22 - np.sqrt(varwprp_blue22 / wprp_22**2),
                          wprp_blue22 / wprp_22 + np.sqrt(varwprp_blue22 / wprp_22**2), color=colors[-1], alpha=0.5)
    ax[1][0].errorbar(wpz2221_rb[:, 0], wpz2221_rb[:, 2] / wpz2221[:, 1], wpz2221_rb[:, 4] /
                      wpz2221[:, 1], color='k', marker='o', linestyle='', markersize=ms, alpha=0.5)

    ax[1][0].set_xscale('log')
    ax[1][0].set_xlim([0.1, 30])

    ax[1][1].plot(rmean, wprp_red21 / wprp_21)
    ax[1][1].fill_between(rmean, wprp_red21 / wprp_21 - np.sqrt(varwprp_red21 / wprp_21**2),
                          wprp_red21 / wprp_21 + np.sqrt(varwprp_red21 / wprp_21**2), color=colors[0], alpha=0.5)
    ax[1][1].errorbar(wpz2120_rb[:, 0], wpz2120_rb[:, 1] / wpz2120[:, 1], wpz2120_rb[:,
                                                                                     3] / wpz2120[:, 1], color='k', marker='o', linestyle='', markersize=ms)

    ax[1][1].plot(rmean, wprp_blue21 / wprp_21, color=colors[-1])
    ax[1][1].fill_between(rmean, wprp_blue21 / wprp_21 - np.sqrt(varwprp_blue21 / wprp_21**2),
                          wprp_blue21 / wprp_21 + np.sqrt(varwprp_blue21 / wprp_21**2), color=colors[-1], alpha=0.5)
    ax[1][1].errorbar(wpz2120_rb[:, 0], wpz2120_rb[:, 2] / wpz2120[:, 1], wpz2120_rb[:, 4] /
                      wpz2120[:, 1], color='k', marker='o', linestyle='', markersize=ms, alpha=0.5)

    ax[1][1].set_xscale('log')
    ax[1][1].set_xlim([0.1, 30])

    ax[1][2].plot(rmean, wprp_red20 / wprp_20)

    ax[1][2].fill_between(rmean, wprp_red20 / wprp_20 - np.sqrt(varwprp_red20 / wprp_20**2),
                          wprp_red20 / wprp_20 + np.sqrt(varwprp_red20 / wprp_20**2), color=colors[0], alpha=0.5)
    ax[1][2].errorbar(wpz2019_rb[:, 0], wpz2019_rb[:, 1] / wpz2019[:, 1], wpz2019_rb[:,
                                                                                     3] / wpz2019[:, 1], color='k', marker='o', linestyle='', markersize=ms)

    ax[1][2].plot(rmean, wprp_blue20 / wprp_20, color=colors[-1])
    ax[1][2].fill_between(rmean, wprp_blue20 / wprp_20 - np.sqrt(varwprp_blue20 / wprp_20**2),
                          wprp_blue20 / wprp_20 + np.sqrt(varwprp_blue20 / wprp_20**2), color=colors[-1], alpha=0.5)
    ax[1][2].errorbar(wpz2019_rb[:, 0], wpz2019_rb[:, 2] / wpz2019[:, 1], wpz2019_rb[:, 4] /
                      wpz2019[:, 1], color='k', marker='o', linestyle='', markersize=ms, alpha=0.5)

    ax[1][0].set_ylim([0.1, 3])

    ax[0][2].set_xscale('log')
    ax[0][2].set_xlim([0.1, 30])
    ax[0][2].set_ylim([1, 10**3])
    f.set_figheight(8)
    ax[1][1].set_xlabel(r'$r_{p}\,[h^{-1}{\rm Mpc}]$', fontsize=20)
    ax[0][0].set_ylabel(r'$w_{p}(r_{p})$', fontsize=20)

    ax[1][0].set_ylabel(
        r'$w_{p}(r_{p})\bigg/ w_{p}(r_{p})^{fid}$', fontsize=20)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    for j in range(3):
        locmin = matplotlib.ticker.AutoMinorLocator()
        ax[1][j].yaxis.set_minor_locator(locmin)
        ax[0][j].tick_params(which='both', direction="in")
        ax[0][j].text(0.2, 0.9, r'${:.0f}< M_{{r}} < {:.0f}$'.format(
                amagbins[j], amagbins[j + 1]), transform=ax[0][j].transAxes, fontsize=16)

        ax[1][j].tick_params(which='both', direction="in")

    f.set_figheight(7)
    ax[0][0].legend([lb[0], ls[0]], [
                    r'\textsc{Addgals}', r'\textsc{SDSS}'], fontsize=16)
    plt.savefig('{}/addgals_sdss_wprp_colordep_twopanel_smallpoints.pdf'.format(valpath),
                dpi=100, bbox_inches='tight')

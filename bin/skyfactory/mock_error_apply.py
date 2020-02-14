#!/usr/bin/env python
from __future__ import print_function, division
from glob import glob
from mpi4py import MPI
from rot_mock_tools import rot_mock_file
import redmapper
import healsparse
import numpy as np
import healpy as hp
import fitsio
import pickle
import yaml
import sys
import os
import h5py as h5
from sklearn.neighbors import KDTree

models = {
    'DR8':
        {'maglims': [20.425, 21.749, 21.239, 20.769, 19.344],
         'exptimes': [21.00, 159.00, 126.00, 99.00, 15.00],
         'lnscat': [0.284, 0.241, 0.229, 0.251, 0.264]
         },

    'STRIPE82':
        {
            'maglims': [22.070, 23.432, 23.095, 22.649, 21.160],
            'exptimes': [99.00, 1172.00, 1028.00, 665.00, 138.00],
            'lnscat': [0.280, 0.229, 0.202, 0.204, 0.246]
        },

    'CFHTLS':
        {
            'maglims': [24.298, 24.667, 24.010, 23.702, 22.568],
            'exptimes': [2866.00, 7003.00, 4108.00, 3777.00, 885.00],
            'lnscat': [0.259, 0.244, 0.282, 0.258, 0.273]
        },
    'DEEP2':
        {
            'maglims': [24.730, 24.623, 24.092],
            'exptimes': [7677.00, 8979.00, 4402.00],
            'lnscat': [0.300, 0.293, 0.300]
        },
    'FLAMEX':
        {
            'maglims': [21.234, 20.929],
            'exptimes': [259.00, 135.00],
            'lnscat': [0.300, 0.289]
        },
    'IRAC':
        {
            'maglims': [19.352, 18.574],
            'exptimes': [8.54, 3.46],
            'lnscat': [0.214, 0.283]
        },
    'NDWFS':
        {
            'maglims': [25.142, 23.761, 23.650],
            'exptimes': [6391.00, 1985.00, 1617.00],
            'lnscat': [0.140, 0.294, 0.272]
        },

    'RCS':
        {
            'maglims': [23.939, 23.826, 23.067, 21.889],
            'exptimes': [2850.00, 2568.00, 1277.00, 431.00],
            'lnscat': [0.164, 0.222, 0.250, 0.271]
        },
    'VHS':
        {
            'maglims': [20.141, 19.732, 19.478],
            'exptimes': [36.00, 31.00, 23.00],
            'lnscat': [0.097, 0.059, 0.069]
        },
    'VIKING':
        {
            'maglims': [21.643, 20.915, 20.768, 20.243, 20.227],
            'exptimes': [622.00, 246.00, 383.00, 238.00, 213.00],
            'lnscat': [0.034, 0.048, 0.052, 0.040, 0.066]
        },
    'DC6B':
        {
            'maglims': [24.486, 23.473, 22.761, 22.402],
            'exptimes': [2379.00, 1169.00, 806.00, 639.00],
            'lnscat': [0.300, 0.300, 0.300, 0.300]
        },

    'DES':
        {
            'maglims': [24.956, 24.453, 23.751, 23.249, 21.459],
            'exptimes': [14467.00, 12471.00, 6296.00, 5362.00, 728.00],
            'lnscat': [0.2, 0.2, 0.2, 0.2, 0.2]
        },

    'BCS_LO':
        {
            'maglims': [23.082, 22.618, 22.500, 21.065],
            'exptimes': [809.00, 844.00, 641.00, 108.00],
            'lnscat': [0.277, 0.284, 0.284, 0.300]
        },

    'BCS':
        {
            'maglims': [23.360, 23.117, 22.539, 21.335],
            'exptimes': [838.00, 1252.00, 772.00, 98.00],
            'lnscat': [0.276, 0.272, 0.278, 0.279]
        },

    'DES_SV':
    {
        'maglims': [23.621, 23.232, 23.008, 22.374, 20.663],
        'exptimes': [4389.00, 1329.00, 1405.00, 517.00, 460.00],
        'lnscat': [0.276, 0.257, 0.247, 0.241, 0.300]
            },

    'DES_SV_OPTIMISTIC':
        {
        'maglims': [23.621 + 0.5, 23.232 + 0.5, 23.008, 22.374, 20.663],
        'exptimes': [4389.00, 1329.00, 1405.00, 517.00, 460.00],
        'lnscat': [0.276, 0.257, 0.247, 0.241, 0.300]
            },
    'WISE':
        {
        'maglims': [19.352, 18.574],
        'exptimes': [8.54, 3.46],
        'lnscat': [0.214, 0.283]
            },

    'DECALS':
        {
            'maglims': [23.3, 23.3, 22.2, 20.6, 19.9],
            'exptimes': [1000, 3000, 2000, 1500, 1500],
            'lnscat': [0.2, 0.2, 0.2, 0.2, 0.2]
        },

}


def calc_nonuniform_errors(exptimes, limmags, mag_in, nonoise=False, zp=22.5,
                           nsig=10.0, fluxmode=False, lnscat=None, b=None,
                           inlup=False, detonly=False):

    f1lim = 10**((limmags - zp) / (-2.5))
    fsky1 = ((f1lim**2) * exptimes) / (nsig**2) - f1lim
    fsky1[fsky1 < 0.001] = 0.001

    if inlup:
        bnmgy = b * 1e9
        tflux = exptimes * 2.0 * bnmgy * \
            np.sinh(-np.log(b) - 0.4 * np.log(10.0) * mag_in)
    else:
        tflux = exptimes * 10**((mag_in - zp) / (-2.5))

    noise = np.sqrt(fsky1 * exptimes + tflux)

    if lnscat is not None:
        noise = np.exp(np.log(noise) + lnscat * np.random.randn(len(mag_in)))

    if nonoise:
        flux = tflux
    else:
        flux = tflux + noise * np.random.randn(len(mag_in))

    # convert to nanomaggies

#        noise = np.sqrt(flux)/exptimes
        flux = flux / exptimes
        noise = noise / exptimes

        flux = flux * 10 ** ((zp - 22.5) / -2.5)
        noise = noise * 10 ** ((zp - 22.5) / -2.5)

    if fluxmode:
        mag = flux
        mag_err = noise
    else:
        if b is not None:
            bnmgy = b * 1e9
            flux_new = flux
            noise_new = noise
            mag = 2.5 * np.log10(1.0 / b) - asinh2(0.5 *
                                                   flux_new / (bnmgy)) / (0.4 * np.log(10.0))

            mag_err = 2.5 * noise_new / \
                (2. * bnmgy * np.log(10.0) *
                 np.sqrt(1.0 + (0.5 * flux_new / (bnmgy))**2.))

        else:
            mag = 22.5 - 2.5 * np.log10(flux)
            mag_err = (2.5 / np.log(10.)) * (noise / flux)

            # temporarily changing to cut to 10-sigma detections in i,z
            bad = np.where((np.isfinite(mag) == False))
            nbad = len(bad)

            if detonly:
                mag[bad] = 99.0
                mag_err[bad] = 99.0

    return mag, mag_err


def calc_uniform_errors(model, tmag, maglims, exptimes, lnscat, zp=22.5):

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


def setup_redmapper_infodict(depthmapfile, maskfile, mode, bands, refband):
    mask = healsparse.HealSparseMap.read(maskfile)
    depth = healsparse.HealSparseMap.read(depthmapfile)

    area = np.sum(mask.getValuePixel(mask.validPixels)) * \
        hp.nside2pixarea(mask.nsideSparse, degrees=True)

    print('Area = ', area)

    lim_ref = np.max(depth.getValuePixel(depth.validPixels)['m50'])

    print('Lim_ref = ', lim_ref)

    if mode == 'DES':
        b_array = np.array([3.27e-12, 4.83e-12, 6.00e-12, 9.00e-12])

    zp = 22.5

    nmag = len(bands)

    ref_ind = bands.index(refband)

    redmapper_dtype = [('id', 'i8'),             # galaxy id number (unique)
                       ('ra', 'f8'),             # right ascension (degrees)
                       ('dec', 'f8'),            # declination (degrees)
                       # total magnitude in reference band
                       ('refmag', 'f4'),
                       ('refmag_err', 'f4'),     # error in total reference mag
                       ('mag', 'f4', nmag),      # mag array
                       ('mag_err', 'f4', nmag),  # magnitude error array
                       ('ebv', 'f4'),            # E(B-V) (systematics checking)
                       # ztrue if from a simulated catalog
                       ('ztrue', 'f4'),
                       # m200 of halo if from a simulated catalog
                       ('m200', 'f4'),
                       # central? 1 if yes (if from sims)
                       ('central', 'i2'),
                       ('halo_id', 'i8')]        # halo_id if from a simulated catalog

    info_dict = {}
    info_dict['LIM_REF'] = lim_ref
    info_dict['REF_IND'] = ref_ind
    info_dict['AREA'] = area
    info_dict['NMAG'] = nmag
    info_dict['MODE'] = mode  # currently SDSS, DES, or LSST
    info_dict['ZP'] = zp
    info_dict['B'] = b_array  # if magnitudes are actually luptitudes

    info_dict['mask'] = mask
    info_dict['depth'] = depth

    if mode == 'DES':
        info_dict['G_IND'] = 0  # g-band index
        info_dict['R_IND'] = 1  # r-band index
        info_dict['I_IND'] = 2  # i-band index
        info_dict['Z_IND'] = 3  # z-band index
    elif mode == 'SDSS':
        info_dict['U_IND'] = 0  # u-band index
        info_dict['G_IND'] = 1  # g-band index
        info_dict['R_IND'] = 2  # r-band index
        info_dict['I_IND'] = 3  # i-band index
        info_dict['Z_IND'] = 4  # z-band index

    return info_dict, redmapper_dtype


def write_redmapper_files(galaxies, filename_base, info_dict,
                          redmapper_dtype, maker):

    mask = info_dict['mask']
    depth = info_dict['depth']

    b_array = info_dict['B']
    bscale = info_dict['B'] * (10.**(info_dict['ZP'] / 2.5))
    ref_ind = info_dict['REF_IND']

    gals = np.zeros(galaxies.size, dtype=redmapper_dtype)

    gals['id'] = galaxies['ID']
    gals['ra'] = galaxies['RA']
    gals['dec'] = galaxies['DEC']
    gals['ebv'] = 0.0
    gals['ztrue'] = galaxies['Z']
    gals['m200'] = galaxies['M200']
    gals['central'] = galaxies['CENTRAL']
    gals['halo_id'] = galaxies['HALOID']

    for i, band in enumerate(bands):
        influx = galaxies['FLUX_%s' % (band.upper())]
        influx_err = 1. / np.sqrt(galaxies['IVAR_%s' % (band.upper())])

        mag = 2.5 * np.log10(1.0 / b_array[i]) - np.arcsinh(
            0.5 * influx / bscale[i]) / (0.4 * np.log(10.0))
        mag_err = 2.5 * influx_err / \
            (2.0 * bscale[i] * np.log(10.0) *
             np.sqrt(1.0 + (0.5 * influx / bscale[i])**2.))

        gals['mag'][:, i] = mag
        gals['mag_err'][:, i] = mag_err

    gals['refmag'][:] = gals['mag'][:, ref_ind]
    gals['refmag_err'][:] = gals['mag_err'][:, ref_ind]

    use, = np.where((mask.getValueRaDec(gals['ra'], gals['dec']) > 0) &
                    (depth.getValueRaDec(gals['ra'], gals['dec'])['m50'] > gals['refmag']))

    if use.size == 0:
        print('No good galaxies in pixel!')
    else:
        print('Ingesting %d galaxies...' % (use.size))

        maker.append_galaxies(gals[use])


def make_output_structure(ngals, dbase_style=False, bands=None, nbands=None,
                          all_obs_fields=True, blind_obs=False,
                          balrog_bands=None,
                          bal):

    if all_obs_fields & dbase_style:
        if bands is None:
            raise(ValueError("Need names of bands in order to use database formatting!"))

        fields = [('ID', np.int), ('RA', np.float), ('DEC', np.float),
                  ('EPSILON1', np.float), ('EPSILON2', np.float),
                  ('SIZE', np.float), ('PHOTOZ_GAUSSIAN', np.float)]

        for b in bands:
            fields.append(('MAG_{0}'.format(b.upper()), np.float))
            fields.append(('MAGERR_{0}'.format(b.upper()), np.float))
            fields.append(('FLUX_{0}'.format(b.upper()), np.float))
            fields.append(('IVAR_{0}'.format(b.upper()), np.float))

        if balrog_bands is not None:
            for b in balrog_bands:
                fields.append(('MCAL_MAG_{0}'.format(b.upper()), np.float))
                fields.append(('MCAL_MAGERR_{0}'.format(b.upper()), np.float))
                fields.append(('MCAL_FLUX_{0}'.format(b.upper()), np.float))
                fields.append(('MCAL_IVAR_{0}'.format(b.upper()), np.float))

    if all_obs_fields & (not dbase_style):

        fields = [('ID', np.int), ('RA', np.float), ('DEC', np.float),
                  ('EPSILON1', np.float), ('EPSILON2', np.float),
                  ('SIZE', np.float), ('PHOTOZ_GAUSSIAN', np.float),
                  ('MAG', (np.float, nbands)), ('FLUX', (np.float, nbands)),
                  ('MAGERR', (np.float, nbands)), ('IVAR', (np.float, nbands))]

    if (not all_obs_fields) & dbase_style:
        fields = [('ID', np.int)]
        for b in bands:
            fields.append(('MAG_{0}'.format(b.upper()), np.float))
            fields.append(('MAGERR_{0}'.format(b.upper()), np.float))
            fields.append(('FLUX_{0}'.format(b.upper()), np.float))
            fields.append(('IVAR_{0}'.format(b.upper()), np.float))

    if (not all_obs_fields) & (not dbase_style):
        fields = [('ID', np.int), ('MAG', (np.float, nbands)),
                  ('FLUX', (np.float, nbands)), ('MAGERR', (np.float, nbands)),
                  ('IVAR', (np.float, nbands))]

    if blind_obs:
        fields.extend([('M200', np.float), ('Z', np.float),
                       ('CENTRAL', np.int), ('HALOID', np.int64),
                       ('R200', np.float), ('Z_COS', np.float)])

    odtype = np.dtype(fields)

    out = np.zeros(ngals, dtype=odtype)

    return out


def setup_deep_bal_cats(detection_catalog):

    # only keep things with good matches
    match_idx = detection_catalog['match_flag_1.5_asec'] < 2
    detection_catalog = detection_catalog[match_idx]

    # get unique deep field galaxies
    _, uidx = np.unique(detection_catalog['true_id'], return_index=True)
    true_deep_cat = detection_catalog[uidx]

    # rename ids so that they are contiguous
    sidx = true_deep_cat['true_id'].argsort()
    true_deep_cat = true_deep_cat[sidx]
    old_id = np.copy(true_deep_cat['true_id'])
    true_deep_cat['true_id'] = np.arange(len(true_deep_cat))
    map_dict = dict(zip(old_id, true_deep_cat['true_id']))
    detection_catalog['true_id'] = np.array(
        [map_dict[detection_catalog['true_id'][i]] for i in range(len(detection_catalog['true_id']))])

    # sort detection catalog by true_id
    deep_sidx = detection_catalog['true_id'].argsort()
    detection_catalog = detection_catalog[deep_sidx]

    return detection_catalog, true_deep_cat


def generate_bal_id(detection_catalog, true_deep_cat, sim_mag_true):

    n_injections, _ = np.histogram(
        detection_catalog['true_id'], np.arange(len(true_deep_cat) + 1))
    cum_injections = np.cumsum(n_injections)
    deep_tree = KDTree(true_deep_cat['true_bdf_mag_deredden'][:, 1:])
    _, deep_idx = deep_tree.query(sim_mag_true)

    rand = np.random.uniform(size=len(sim_mag_true))
    bal_id = cum_injections[deep_idx].flatten(
    ) - cum_injections[0] + np.floor(rand * n_injections[deep_idx].flatten())

    return detection_catalog['bal_id'][bal_id.astype(np.int)], bal_id.astype(np.int)


def balrog_error_apply(detection_catalog, true_deep_cat, matched_balrog_cat, mag_in,
                       matched_cat_sorter=None, zp=30., zp_data=30.,
                       matched_cat_flux_cols=['flux_r', 'flux_i', 'flux_z'],
                       matched_cat_flux_err_cols=[
                           'flux_err_r', 'flux_err_i', 'flux_err_z'],
                       true_cat_mag_cols=[1, 2, 3]):

    flux_out = np.zeros_like(mag_in)
    flux_err = np.zeros_like(mag_in)
    flux_err_report = np.zeros_like(mag_in)

    # get balrog injection ids for all simulated galaxies
    bal_id, bal_cat_idx = generate_bal_id(
        detection_catalog, true_deep_cat, mag_in)

    # determine which are detected
    detected = detection_catalog['detected'][bal_cat_idx].astype(np.bool)

    # find matches in matched cat to get wide field measured fluxes
    matched_idx = matched_balrog_cat['catalog/unsheared/bal_id'][:].searchsorted(bal_id[detected],
                                                                                 sorter=matched_cat_sorter)

    # calculate error
    if matched_cat_sorter is not None:
        for i in range(len(matched_cat_flux_cols)):
            flux_err[detected, i] = matched_balrog_cat['catalog/unsheared/{}'.format(
                matched_cat_flux_cols[i])][:][matched_cat_sorter][matched_idx]
            flux_err_report[detected, i] = matched_balrog_cat['catalog/unsheared/{}'.format(
                matched_cat_flux_err_cols[i])][:][matched_cat_sorter][matched_idx]

    else:
        for i in range(len(matched_cat_flux_cols)):
            flux_err[detected, i] = matched_balrog_cat['catalog/unsheared/{}'.format(
                matched_cat_flux_cols[i])][:][matched_idx]
            flux_err_report[detected, i] = matched_balrog_cat['catalog/unsheared/{}'.format(
                matched_cat_flux_err_cols[i])][:][matched_idx]

    for i in range(len(matched_cat_flux_cols)):
        flux_err[detected, i] = zp_data - 2.5 * np.log10(flux_err[detected, i])
        flux_err[detected, i] -= detection_catalog['true_bdf_mag_deredden'][bal_cat_idx[detected.astype(
            np.bool)], true_cat_mag_cols[i]]
        flux_out[detected, :] = mag_in[detected, :] + flux_err[detected, :]

    flux_out = 10**((flux_out - zp) / -2.5)
    flux_out[~detected, :] = -99

    return flux_out, flux_err_report

def apply_nonuniform_errormodel(g, obase, odir, d, dhdr,
                                survey, magfile=None, usemags=None,
                                nest=False, bands=None, balrog_bands=None,
                                usebalmags=None, all_obs_fields=True,
                                dbase_style=True, use_lmag=True,
                                sigpz=0.03, blind_obs=False, filter_obs=True,
                                refbands=None, zp=22.5, maker=None,
                                redmapper_info_dict=None,
                                redmapper_dtype=None,
                                detection_catalog=None,
                                true_deep_cat=None,
                                matched_catalog=None,
                                matched_cat_sorter=None):

    if magfile is not None:
        mags = fitsio.read(magfile)
        if use_lmag:
            if ('LMAG' in mags.dtype.names) and (mags['LMAG'] != 0).any():
                imtag = 'LMAG'
                omag = mags['LMAG']
            else:
                raise(KeyError("No LMAG field!"))
        else:
            try:
                imtag = 'TMAG'
                omag = mags['TMAG']
            except:
                imtag = 'OMAG'
                omag = mags['OMAG']
    else:
        if use_lmag:
            if ('LMAG' in g.dtype.names) and (g['LMAG'] != 0).any():
                imtag = 'LMAG'
                omag = g['LMAG']
            else:
                raise(ValueError("No LMAG field"))
        else:
            try:
                imtag = 'TMAG'
                omag = g['TMAG']
            except:
                imtag = 'OMAG'
                omag = g['OMAG']

    if use_lmag:
        ra = g['RA']
        dec = g['DEC']
    else:
        print('Using unlensed positions!')

        vec = np.zeros((len(g), 3))
        vec[:, 0] = g['PX']
        vec[:, 1] = g['PY']
        vec[:, 2] = g['PZ']

        ra, dec = hp.vec2ang(vec, lonlat=True)

    if balrog_bands is not None:
        apply_balrog_errors = True

    if dbase_style:
        mnames = ['MAG_{0}'.format(b.upper()) for b in bands]
        menames = ['MAGERR_{0}'.format(b.upper()) for b in bands]
        fnames = ['FLUX_{0}'.format(b.upper()) for b in bands]
        fenames = ['IVAR_{0}'.format(b.upper()) for b in bands]

        if apply_balrog_errors:
            bfnames = ['MCAL_FLUX_{0}'.format(b.upper()) for b in balrog_bands]
            bfenames = ['MCAL_IVAR_{0}'.format(b.upper()) for b in balrog_bands]

        if filter_obs & (refbands is not None):
            refnames = ['MAG_{}'.format(b.upper()) for b in refbands]
        elif filter_obs:
            refnames = mnames
    else:
        if filter_obs & (refbands is not None):
            refnames = refbands
        elif filter_obs:
            refnames = range(len(usemags))

    fs = fname.split('.')
    oname = "{0}/{1}_obs.{2}.fits".format(odir, obase, fs[-2])

    # get mags to use
    if usemags is None:
        nmag = omag.shape[1]
        usemags = range(nmag)
    else:
        nmag = len(usemags)

    # make output structure
    obs = make_output_structure(len(g), dbase_style=dbase_style, bands=bands,
                                nbands=len(usemags),
                                all_obs_fields=all_obs_fields,
                                blind_obs=blind_obs,
                                apply_balrog_errors=apply_balrog_errors)

    if ("Y1" in survey) | ("Y3" in survey) | (survey == "DES") | (survey == "SVA") | (survey == 'Y3'):
        mindec = -90.
        maxdec = 90
        minra = 0.0
        maxra = 360.

    elif survey == "DR8":
        mindec = -20
        maxdec = 90
        minra = 0.0
        maxra = 360.

#    theta = (90 - dec) * np.pi / 180.
#    phi = (ra * np.pi / 180.)

    pix = hp.ang2pix(dhdr['NSIDE'], ra, dec, nest=nest, lonlat=True)

    guse = np.in1d(pix, d['HPIX'])
    guse, = np.where(guse)

    if not any(guse):
        print("No galaxies in this pixel are in the footprint")
        return

    pixind = d['HPIX'].searchsorted(pix[guse], side='right')
    pixind -= 1

    oidx = np.zeros(len(omag), dtype=bool)
    oidx[guse] = True

    if apply_balrog_errors:
        flux_bal, fluxerr_bal = balrog_error_apply(detection_catalog,
                                                   true_deep_cat,
                                                   matched_catalog,
                                                   omag[guse, usebalmags],
                                                   matched_cat_sorter=matched_cat_sorter,
                                                   zp=zp,
                                                   true_cat_mag_cols=balusemags)

    bal_idx = dict(zip(usebalmags, np.arange(len(usebalmags))))

    for ind, i in enumerate(usemags):

        flux, fluxerr = calc_nonuniform_errors(d['EXPTIMES'][pixind, ind],
                                               d['LIMMAGS'][pixind, ind],
                                               omag[guse, i], fluxmode=True,
                                               zp=zp)

        if not dbase_style:

            obs['OMAG'][:, ind] = 99
            obs['OMAGERR'][:, ind] = 99

            obs['FLUX'][guse, ind] = flux
            obs['IVAR'][guse, ind] = 1 / fluxerr**2
            obs['OMAG'][guse, ind] = 22.5 - 2.5 * np.log10(flux)
            obs['OMAGERR'][guse, ind] = 1.086 * fluxerr / flux

            bad = (flux <= 0)

            obs['OMAG'][guse[bad], ind] = 99.0
            obs['OMAGERR'][guse[bad], ind] = 99.0

            r = np.random.rand(len(pixind))

            if len(d['FRACGOOD'].shape) > 1:
                bad = r > d['FRACGOOD'][pixind, ind]
            else:
                bad = r > d['FRACGOOD'][pixind]

            if len(bad) > 0:
                obs['OMAG'][guse[bad], ind] = 99.0
                obs['OMAGERR'][guse[bad], ind] = 99.0

            if filter_obs and (ind in refnames):
                oidx &= obs['OMAG'][:, ind] < (d['LIMMAGS'][pixind, ind] + 0.5)

        else:
            obs[mnames[ind]] = 99.0
            obs[menames[ind]] = 99.0

            obs[fnames[ind]][guse] = flux_bal
            obs[fenames[ind]][guse] = 1 / fluxerr_bal**2
            obs[mnames[ind]][guse] = 22.5 - 2.5 * np.log10(flux)
            obs[menames[ind]][guse] = 1.086 * fluxerr / flux

            bad = (flux <= 0)

            obs[mnames[ind]][guse[bad]] = 99.0
            obs[menames[ind]][guse[bad]] = 99.0

            # Set fluxes, magnitudes of non detections to zero, 99
            ntobs = ~np.isfinite(flux)
            obs[fnames[ind]][guse[ntobs]] = 0.0
            obs[fenames[ind]][guse[ntobs]] = 0.0
            obs[mnames[ind]][guse[ntobs]] = 99.0
            obs[mnames[ind]][guse[ntobs]] = 99.0

            if apply_balrog_errors:
                if i in balusemags:
                    obs[bfnames[bal_idx[ind]]][guse] = flux_bal[:, i]
                    obs[bfenames[bal_idx[ind]]][guse] = 1 / fluxerr_bal[:, i]**2
                    bad = (flux_bal[:, i] <= 0)
                    obs[bfnames[bal_idx[ind]]][guse[bad]] = 0.0
                    obs[bfenames[bal_idx[ind]]][guse[bad]] = 0.0

            r = np.random.rand(len(pixind))

            if len(d['FRACGOOD'].shape) > 1:
                bad = r > d['FRACGOOD'][pixind, ind]
            else:
                bad = r > d['FRACGOOD'][pixind]
            if any(bad):
                obs[mnames[ind]][guse[bad]] = 99.0
                obs[menames[ind]][guse[bad]] = 99.0

                if apply_balrog_errors:
                    obs[bfnames[bal_idx[ind]]][guse[bad]] = 0.0
                    obs[bfenames[bal_idx[ind]]][guse[bad]] = 0.0

            if (filter_obs) and (mnames[ind] in refnames):
                oidx[guse] &= obs[mnames[ind]][guse] < (
                    d['LIMMAGS'][pixind, ind] + 0.5)



    obs['RA'] = ra
    obs['DEC'] = dec

    obs['ID'] = g['ID']
    obs['EPSILON1'] = g['EPSILON'][:, 0]
    obs['EPSILON2'] = g['EPSILON'][:, 1]
    obs['SIZE'] = g['SIZE']
    obs['PHOTOZ_GAUSSIAN'] = g['Z'] + sigpz * \
        (1 + g['Z']) * (np.random.randn(len(g)))

    if blind_obs:
        obs['M200'] = g['M200']
        obs['CENTRAL'] = g['CENTRAL']
        obs['Z'] = g['Z']
        obs['R200'] = g['R200']
        obs['HALOID'] = g['HALOID']
        obs['Z_COS'] = g['Z_COS']

    fitsio.write(oname, obs, clobber=True)

    if maker is not None:
        write_redmapper_files(obs[oidx], odir, redmapper_info_dict,
                              redmapper_dtype, maker)

    return oidx


def apply_uniform_errormodel(g, obase, odir, survey, filename_base,
                             magfile=None, usemags=None,
                             bands=None, all_obs_fields=True,
                             dbase_style=True, use_lmag=True,
                             sigpz=0.03, blind_obs=False, filter_obs=True,
                             refbands=None, zp=22.5, maker=None,
                             redmapper_info_dict=None,
                             redmapper_dtype=None):

    if magfile is not None:
        mags = fitsio.read(magfile)
        if use_lmag:
            if ('LMAG' in mags.dtype.names) and (mags['LMAG'] != 0).any():
                imtag = 'LMAG'
                omag = mags['LMAG']
            else:
                raise(KeyError("No LMAG field!"))
        else:
            try:
                imtag = 'TMAG'
                omag = mags['TMAG']
            except:
                imtag = 'OMAG'
                omag = mags['OMAG']
    else:
        if use_lmag:
            if ('LMAG' in g.dtype.names) and (g['LMAG'] != 0).any():
                imtag = 'LMAG'
                omag = g['LMAG']
            else:
                raise(ValueError("No LMAG field"))
        else:
            try:
                imtag = 'TMAG'
                omag = g['TMAG']
            except:
                imtag = 'OMAG'
                omag = g['OMAG']

    if dbase_style:
        mnames = ['MAG_{0}'.format(b.upper()) for b in bands]
        menames = ['MAGERR_{0}'.format(b.upper()) for b in bands]
        fnames = ['FLUX_{0}'.format(b.upper()) for b in bands]
        fenames = ['IVAR_{0}'.format(b.upper()) for b in bands]

        if filter_obs & (refbands is not None):
            refnames = ['MAG_{}'.format(b.upper()) for b in refbands]
        elif filter_obs:
            refnames = mnames
    else:
        if filter_obs & (refbands is not None):
            refnames = refbands
        elif filter_obs:
            refnames = range(len(usemags))

    fs = fname.split('.')
    oname = "{0}/{1}_obs.{2}.fits".format(odir, obase, fs[-2])

    # get mags to use
    if usemags is None:
        nmag = omag.shape[1]
        usemags = range(nmag)
    else:
        nmag = len(usemags)

    # make output structure
    obs = make_output_structure(len(g), dbase_style=dbase_style, bands=bands,
                                nbands=len(usemags),
                                all_obs_fields=all_obs_fields,
                                blind_obs=blind_obs)

    if ("Y1" in survey) | (survey == "DES") | (survey == "SVA"):
        mindec = -90.
        maxdec = 90
        minra = 0.0
        maxra = 360.

    elif survey == "DR8":
        mindec = -20
        maxdec = 90
        minra = 0.0
        maxra = 360.

    maglims = np.array(models[model]['maglims'])
    exptimes = np.array(models[model]['exptimes'])
    lnscat = np.array(models[model]['lnscat'])

    oidx = np.ones(len(omag), dtype=bool)

    for ind, i in enumerate(usemags):

        _, _, flux, fluxerr = calc_uniform_errors(model, omag[:, i],
                                                  np.array([maglims[ind]]),
                                                  np.array([exptimes[ind]]),
                                                  np.array([lnscat[ind]]),
                                                  zp=zp)

        flux = flux.reshape(len(flux))
        fluxerr = fluxerr.reshape(len(fluxerr))

        if not dbase_style:

            obs['FLUX'][:, ind] = flux
            obs['IVAR'][:, ind] = 1 / fluxerr**2
            obs['OMAG'][:, ind] = 22.5 - 2.5 * np.log10(flux)
            obs['OMAGERR'][:, ind] = 1.086 * fluxerr / flux

            bad = (flux <= 0)

            obs['OMAG'][bad, ind] = 99.0
            obs['OMAGERR'][bad, ind] = 99.0

            if filter_obs and (ind in refnames):
                oidx &= obs['OMAG'][:, ind] < (maglims[ind] + 0.5)

        else:
            obs[mnames[ind]] = 99.0
            obs[menames[ind]] = 99.0

            obs[fnames[ind]] = flux
            obs[fenames[ind]] = 1 / fluxerr**2
            obs[mnames[ind]] = 22.5 - 2.5 * np.log10(flux)
            obs[menames[ind]] = 1.086 * fluxerr / flux

            bad = (flux <= 0)

            obs[mnames[ind]][bad] = 99.0
            obs[menames[ind]][bad] = 99.0

            if (filter_obs) and (mnames[ind] in refnames):
                print('filtering {}'.format(mnames[ind]))
                oidx &= obs[mnames[ind]] < (maglims[ind] + 0.5)
            else:
                print('mnames[ind]: {}'.format(mnames[ind]))

#    print('filter_obs: {}'.format(filter_obs))
#    print('refnames: {}'.format(refnames))
#    print('maglims: {}'.format(maglims))
#    print('oidx.any(): {}'.format(oidx.any()))

    if use_lmag:
        obs['RA'] = g['RA']
        obs['DEC'] = g['DEC']
    else:
        obs['RA'] = g['TRA']
        obs['DEC'] = g['TDEC']

    obs['ID'] = g['ID']
    obs['EPSILON1'] = g['EPSILON'][:, 0]
    obs['EPSILON2'] = g['EPSILON'][:, 1]
    obs['SIZE'] = g['SIZE']
    obs['PHOTOZ_GAUSSIAN'] = g['Z'] + sigpz * \
        (1 + g['Z']) * (np.random.randn(len(g)))

    if blind_obs:
        obs['M200'] = g['M200']
        obs['CENTRAL'] = g['CENTRAL']
        obs['Z'] = g['Z']
        obs['R200'] = g['R200']
        obs['HALOID'] = g['HALOID']
        obs['Z_COS'] = g['Z_COS']

    fitsio.write(oname, obs, clobber=True)

    if maker is not None:
        write_redmapper_files(obs[oidx], odir, redmapper_info_dict,
                              redmapper_dtype, maker)


def setup_balrog_error_model(detection_file, matched_cat_file):

    detection_catalog = fitsio.read(detection_file,
                                    columns=['match_flag_1.5_asec',
                                             'true_id',
                                             'true_bdf_mag_deredden',
                                             'bal_id',
                                             'detected'])

    detection_catalog, true_deep_cat = setup_deep_bal_cats(detection_catalog)
    matched_catalog = h5.File(matched_cat_file, 'r')
    bal_id_sidx = matched_catalog['catalog/unsheared/bal_id'][:].argsort()

    return detection_catalog, true_deep_cat, matched_catalog, bal_id_sidx


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cfgfile = sys.argv[1]
    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp)

    gpath = cfg['GalPath']
    model = cfg['Model']
    odir = cfg['OutputDir']
    obase = cfg['OutputBase']

    fnames = np.array(glob(gpath))

    if 'DepthFile' in cfg.keys():
        dfile = cfg['DepthFile']
        uniform = False
        if 'Nest' in cfg.keys():
            nest = bool(cfg['Nest'])
        else:
            nest = False

        d, dhdr = fitsio.read(dfile, header=True)
        pidx = d['HPIX'].argsort()
        d = d[pidx]
    else:
        uniform = True

    if ('MagPath' in cfg.keys()) and (cfg['MagPath'] is not None):
        mnames = np.array(glob(cfg['MagPath']))

        fpix = np.array([int(f.split('.')[-2]) for f in fnames])
        mpix = np.array([int(f.split('.')[-2]) for f in mnames])

        fidx = fpix.argsort()
        midx = mpix.argsort()

        assert((fpix[fidx] == mpix[midx]).all())

        fnames = fnames[fidx]
        mnames = mnames[midx]

    else:
        mnames = [None] * len(fnames)

    if 'UseMags' in cfg.keys():
        usemags = cfg['UseMags']
    else:
        usemags = None

    if ('DataBaseStyle' in cfg.keys()) & (cfg['DataBaseStyle'] == True):
        if ('Bands' in cfg.keys()):
            dbstyle = True
            bands = cfg['Bands']
        else:
            raise(KeyError("Need to specify bands for database style formatting"))
    else:
        dbstyle = False

    if ('AllObsFields' in cfg.keys()):
        all_obs_fields = bool(cfg['AllObsFields'])
    else:
        all_obs_fields = True

    if ('BlindObs' in cfg.keys()):
        blind_obs = bool(cfg['BlindObs'])
    else:
        blind_obs = True

    if ('UseLMAG' in cfg.keys()):
        use_lmag = bool(cfg['UseLMAG'])
    else:
        use_lmag = False

    if ('FilterObs' in cfg.keys()):
        filter_obs = bool(cfg['FilterObs'])
    else:
        filter_obs = True

    if ('RefBands' in cfg.keys()):
        refbands = cfg['RefBands']
    else:
        refbands = None

    zp = cfg.pop('zp', 22.5)
    print('zp: {}'.format(zp))

    truth_only = cfg.pop('TruthOnly', False)

    if rank == 0:
        try:
            os.makedirs(odir)
        except Exception as e:
            pass

    if ('RotOutDir' in cfg.keys()):
        if ('MatPath' in cfg.keys()):
            rodir = cfg['RotOutDir']
            robase = cfg['RotBase']
            rpath = cfg['MatPath']
            with open(rpath, 'rb') as fp:
                rot = pickle.load(fp, encoding='latin1')
            try:
                os.makedirs(rodir)
            except Exception as e:
                pass
        else:
            raise(KeyError("No Matrix path!"))

    else:
        rodir = None
        rpath = None
        rot = None
        robase = None

    print("Rank {0} assigned {1} files".format(rank, len(fnames[rank::size])))

    if 'redmapper' in cfg.keys():
        mode = cfg['redmapper']['mode']
        depthmap_healsparse = cfg['redmapper']['depthmap_hs']
        mask_healsparse = cfg['redmapper']['mask_hs']

        fname = fnames[0]
        fs = fname.split('.')
        fp = fs[-2]
        oname = "{0}/{1}_obs_rmp".format(odir, obase)

        redmapper_info_dict, redmapper_dtype = setup_redmapper_infodict(depthmap_healsparse,
                                                                        mask_healsparse, mode,
                                                                        bands, refbands[0])
        maker = redmapper.GalaxyCatalogMaker(
            oname, redmapper_info_dict, parallel=True)

    else:
        maker, redmapper_info_dict, redmapper_dtype = None

    if 'BalrogBands' in cfg.keys():
        balrog_bands = cfg['BalrogBands']
        usebalmags = cfg['UseBalMags']
        detection_file = cfg['DetectionFile']
        matched_cat_file = cfg['MatchedCatFile']

    else:
        balrog_bands = None
        usebalmags = None
        detection_file = None
        matched_cat_file = None
        
    for fname, mname in zip(fnames[rank::size], mnames[rank::size]):
        if rodir is not None:
            p = fname.split('.')[-2]
            nfname = "{0}/{1}.{2}.fits".format(rodir, robase, p)
            g = rot_mock_file(fname, rot, nfname,
                              footprint=d, nside=dhdr['NSIDE'], nest=nest)

            # if returns none, no galaxies in footprint
            if g is None:
                continue
        else:
            g = fitsio.read(fname)

        fs = fname.split('.')
        fp = fs[-2]

        if truth_only:
            continue

        if uniform:
            apply_uniform_errormodel(g, obase, odir, model, magfile=mname,
                                     usemags=usemags,
                                     bands=bands,
                                     all_obs_fields=all_obs_fields,
                                     dbase_style=dbstyle,
                                     use_lmag=use_lmag,
                                     blind_obs=blind_obs,
                                     filter_obs=filter_obs,
                                     refbands=refbands,
                                     zp=zp, maker=maker,
                                     redmapper_info_dict=redmapper_info_dict,
                                     redmapper_dtype=redmapper_dtype)

        else:
            if balrog_bands is not None:
                detection_catalog, true_deep_cat, \
                 matched_catalog, matched_cat_sorter = \
                   setup_balrog_error_model(detection_file, matched_cat_file)
            else:
                detection_catalog = None
                true_deep_cat = None
                matched_catalog = None
                matched_cat_sorter = None

            oidx = apply_nonuniform_errormodel(g, obase, odir, d, dhdr,
                                               model, magfile=mname,
                                               usemags=usemags,
                                               usebalmags=usebalmags,
                                               nest=nest, bands=bands,
                                               balrog_bands=balrog_bands,
                                               all_obs_fields=all_obs_fields,
                                               dbase_style=dbstyle,
                                               use_lmag=use_lmag,
                                               blind_obs=blind_obs,
                                               filter_obs=filter_obs,
                                               refbands=refbands,
                                               zp=zp, maker=maker,
                                               redmapper_info_dict=redmapper_info_dict,
                                               redmapper_dtype=redmapper_dtype,
                                               detection_catalog=detection_catalog,
                                               true_deep_cat=true_deep_cat,
                                               matched_catalog=matched_catalog,
                                               matched_cat_sorter=matched_cat_sorter)

    if matched_catalog is not None:
        matched_catalog.close()

    comm.Barrier()

    if maker is not None:
        maker.finalize_catalog()

    if rank == 0:
        print("*******Rotation and error model complete!*******")

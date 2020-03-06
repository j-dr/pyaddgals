import numpy as np
import healpix_util as hu
import h5py
import sys
import os
import yaml
import healpy as hp
import healsparse
import fitsio
from scipy import spatial
from sklearn.cluster import KMeans

cats_redmagic = ['redmagic_highdens',
                 'redmagic_highlum', 'redmagic_higherlum']
cats_redmapper = ['lgt20_vl02_catalog', 'lgt5_vl02_catalog',
                  'lgt20_vl02_catalog_members', 'lgt5_vl02_catalog_members']
cats_redmapper_random = [
    'randcat_z0.10-0.95_lgt005_vl02', 'randcat_z0.10-0.95_lgt020_vl02']
cats_redmagic_table = ['highdens', 'highlum', 'higherlum']
cats_redmapper_table = ['lgt20', 'lgt5', 'lgt20/members', 'lgt5/members']
cats_redmapper_random_table = ['lgt5', 'lgt20']
lstar_map = {'redmagic_highdens': 'redmagic_05',
             'redmagic_highlum': 'redmagic_10',
             'redmagic_higherlum': 'redmagic_15'}

# Details for combining redmagic samples
combined_dict = {
    'samples': ['redmagic_highdens', 'redmagic_highlum'],
    'binedges': [[0.15, 0.35, 0.5, 0.65], [0.65, 0.85, 0.95]],
    'label': 'combined_sample_fid',
    'fracgood': 0.8,
    'zlum': 4.,
}


def convert_rm_to_h5(rmg_filebase=None, rmp_filebase=None,
                     file='buzzard-3_1.6_y3_run_redmapper_v6.4.20',
                     file_ext='fit',
                     make_combined=True):
    """
    Converts redmagic+redmapper fits files into a single h5 file with separate tables for each including randoms.
    """

    if rmp_filebase is None:
        no_redmapper = True
    else:
        no_redmapper = False

    # Create h5 file
    f = h5py.File(rmg_filebase + file + '.h5', 'w')
    # Loop over redmagic cats fits files and dump into h5
    for i in range(len(cats_redmagic)):
        print('working on {}'.format(cats_redmagic[i]))
        # Read fits file
        cat = fitsio.FITS(rmg_filebase + file + '_' +
                          cats_redmagic[i] + '.' + file_ext)[1].read()
        # Get all col names
        cols = [name for name in cat.dtype.names]
        # Get total length
        total_length = fitsio.FITS(
            rmg_filebase + file + '_' + cats_redmagic[i] + '.' + file_ext)[1].read_header()['NAXIS2']
        # Sort by healpix id
        s = np.argsort(hp.ang2pix(16384, np.pi / 2. -
                                  np.radians(cat['dec']), np.radians(cat['ra']), nest=True))
        # Loop over columns and stick in h5 file
        for name in cols:
            if name.lower() == 'id':
                #                print 'coadd'
                f.create_dataset('catalog/redmagic/' + cats_redmagic_table[i] + '/coadd_object_id', maxshape=(
                    total_length,), shape=(total_length,), dtype=int, chunks=(total_length,))
                f['catalog/redmagic/' + cats_redmagic_table[i] +
                    '/coadd_object_id'][:] = cat[name][s]
            else:
                f.create_dataset('catalog/redmagic/' + cats_redmagic_table[i] + '/' + name.lower(), maxshape=(
                    total_length,), shape=(total_length,), dtype=cat.dtype[name], chunks=(total_length,))
                f['catalog/redmagic/' + cats_redmagic_table[i] +
                    '/' + name.lower()][:] = cat[name][s]

    # Loop over masks and put in h5 file
    for i in range(len(cats_redmagic)):
        maskfile = rmg_filebase + file + '_' + cats_redmagic[i] + '_vlim_zmask.' + file_ext
        if not os.path.exists(maskfile):
            maskfile = maskfile.replace(cats_redmagic[i], lstar_map[cats_redmagic[i]])
        mask = healsparse.HealSparseMap.read(maskfile, nside_coverage=4098)
        mask_values = mask.get_values_pix(mask.valid_pixels)
        cols = [name for name in mask_values.dtype.names]
        total_length = len(mask.valid_pixels)
#        mask['hpix'] = hp.ring2nest(4096, mask['hpix'])
#        s = np.argsort(mask.valid_pixels)
        for name in cols:
            f.create_dataset('masks/redmagic/' + cats_redmagic_table[i] + '/' + name.lower(), maxshape=(
                total_length,), shape=(total_length,), dtype=mask_values.dtype[name], chunks=(100000,))
            f['masks/redmagic/' + cats_redmagic_table[i] +
                '/' + name.lower()][:] = mask_values[name]

        f.create_dataset('masks/redmagic/' + cats_redmagic_table[i] + '/hpix', maxshape=(
            total_length,), shape=(total_length,), dtype=np.int, chunks=(100000,))
        f['masks/redmagic/' + cats_redmagic_table[i] + '/hpix'][:] = mask.valid_pixels

    # Loop over randoms and put in h5 file
    for i in range(len(cats_redmagic)):
        cat = fitsio.FITS(rmg_filebase + file + '_' +
                          cats_redmagic[i] + '_randoms.' + file_ext)[1].read()
        cols = [name for name in cat.dtype.names]
        total_length = fitsio.FITS(
            rmg_filebase + file + '_' + cats_redmagic[i] + '_randoms.' + file_ext)[1].read_header()['NAXIS2']
        s = np.argsort(hp.ang2pix(16384, np.pi / 2. -
                                  np.radians(cat['dec']), np.radians(cat['ra']), nest=True))
        for name in cols:
            f.create_dataset('randoms/redmagic/' + cats_redmagic_table[i] + '/' + name.lower(), maxshape=(
                total_length,), shape=(total_length,), dtype=cat.dtype[name], chunks=(1000000,))
            f['randoms/redmagic/' + cats_redmagic_table[i] +
                '/' + name.lower()][:] = cat[name][s]

    if not no_redmapper:
        # Loop over redmapper cats and put in h5 file
        for i in range(len(cats_redmapper)):
            cat = fitsio.FITS(rmp_filebase + file + '_' +
                              cats_redmapper[i] + '.' + file_ext)[1].read()
            cols = [name for name in cat.dtype.names]
            total_length = fitsio.FITS(
                rmp_filebase + file + '_' + cats_redmapper[i] + '.' + file_ext)[1].read_header()['NAXIS2']
            s = np.argsort(hp.ang2pix(16384, np.pi / 2. -
                                      np.radians(cat['dec']), np.radians(cat['ra']), nest=True))
            for name in cols:
                f.create_dataset('catalog/redmapper/' + cats_redmapper_table[i] + '/' + name.lower(), maxshape=(
                    total_length,), shape=(total_length,), dtype=cat.dtype[name], chunks=(total_length,))
                f['catalog/redmapper/' + cats_redmapper_table[i] +
                    '/' + name.lower()][:] = cat[name][s]

        # Loop over redmapper randoms and put in h5 file
        for i in range(len(cats_redmapper_random)):
            try:
                cat = fitsio.FITS(rmp_filebase + file + '_' +
                                  cats_redmapper_random[i] + '.' + file_ext)[1].read()

                cols = [name for name in cat.dtype.names]
                total_length = fitsio.FITS(
                    rmp_filebase + file + '_' + cats_redmapper_random[i] + '.' + file_ext)[1].read_header()['NAXIS2']
                s = np.argsort(hp.ang2pix(16384, np.pi / 2. -
                                          np.radians(cat['dec']), np.radians(cat['ra']), nest=True))
                for name in cols:
                    f.create_dataset('randoms/redmapper/' + cats_redmapper_random_table[i] + '/' + name.lower(
                    ), maxshape=(total_length,), shape=(total_length,), dtype=cat.dtype[name], chunks=(total_length,))
                    f['randoms/redmapper/' + cats_redmapper_random_table[i] +
                        '/' + name.lower()][:] = cat[name][s]
            except OSError as e:
                print(e)
                print('Redmapper randoms do not exist')

    # Make combined catalog version and add in new h5 table
    if make_combined:
        for i in range(len(combined_dict['samples'])):
            zmax_cut = combined_dict['binedges'][i][-1]
            if i == 0:
                maskfile = rmg_filebase + file + '_' + combined_dict['samples'][i] + '_vlim_zmask.' + file_ext
                mask_master = healsparse.HealSparseMap.read(maskfile, nside_coverage=4098)
                select_zmax = (mask_master.get_values_pix(mask_master.valid_pixels)['zmax'] >= zmax_cut)
                mask_master_pix = mask_master.valid_pixels[select_zmax]
                mask_master_values = mask_master.get_values_pix(mask_master_pix)

            else:
                maskfile = rmg_filebase + file + '_' + combined_dict['samples'][i] + '_vlim_zmask.' + file_ext
                mask = healsparse.HealSparseMap.read(maskfile, nside_coverage=4098)
                mask_values = mask.get_values_pix(mask.valid_pixels)
                print(mask_values['zmax'])
                print(zmax_cut)
                select_zmax = (mask_values['zmax'] >= zmax_cut)
                badpix = mask.valid_pixels[~select_zmax]
                print(badpix)
                select_badpix = np.in1d(mask_master_pix, badpix)
                mask_master_pix = mask_master_pix[~select_badpix]
                mask_master_values = mask_master_values[~select_badpix]
            print(len(mask_master_pix))

        select_fracdet = (mask_master_values['fracgood'] > combined_dict['fracgood'])
        mask_master_values = mask_master_values[select_fracdet]
        mask_master_pix = mask_master_pix[select_fracdet]
        cols = [name for name in mask_master_values.dtype.names]
        total_length = len(mask_master_values)
        for name in cols:
            f.create_dataset('masks/redmagic/' + combined_dict['label'] + '/' + name.lower(), maxshape=(
                total_length,), shape=(total_length,), dtype=mask_master_values.dtype[name], chunks=(100000,))
            f['masks/redmagic/' + combined_dict['label'] +
                '/' + name.lower()][:] = mask_master_values[name]

        f.create_dataset('masks/redmagic/' + combined_dict['label'] + '/hpix', maxshape=(
            total_length,), shape=(total_length,), dtype=mask_master_pix.dtype, chunks=(100000,))
        f['masks/redmagic/' + combined_dict['label'] + '/hpix'][:] = mask_master_pix

        # combined catalog
        for i in range(len(combined_dict['samples'])):
            cat_sample = fitsio.FITS(
                rmg_filebase + file + '_' + combined_dict['samples'][i] + '.' + file_ext)[1].read()
            ran_sample_ = fitsio.FITS(
                rmg_filebase + file + '_' + combined_dict['samples'][i] + '_randoms.' + file_ext)[1].read()
            binedges = combined_dict['binedges'][i]
            select_zrange = (
                cat_sample['zredmagic'] >= binedges[0]) * (cat_sample['zredmagic'] < binedges[-1])
            cat_sample = cat_sample[select_zrange]
            select_zrange = (
                ran_sample_['z'] >= binedges[0]) * (ran_sample_['z'] < binedges[-1])
            ran_sample_ = ran_sample_[select_zrange]
            if i == 0:
                cat = cat_sample
                ran_sample = ran_sample_
            else:
                cat = np.append(cat, cat_sample)
                ran_sample = np.append(ran_sample, ran_sample_)

        # apply combined mask
        # from healpix_util import HealPix
        # hpix = HealPix('ring',4096)
        # catpix = hpix.eq2pix(cat['RA'],cat['DEC'])
        catpix = hp.ang2pix(
            4096, np.pi / 2. - np.radians(ran_sample['dec']), np.radians(ran_sample['ra']), nest=True)
        select_inmask = np.in1d(catpix, mask_master_pix)
        ran_sample = ran_sample[select_inmask]

        catpix = hp.ang2pix(
            4096, np.pi / 2. - np.radians(cat['dec']), np.radians(cat['ra']), nest=True)
        select_inmask = np.in1d(catpix, mask_master_pix)

        # apply ZLUM cut
        select_zlum = (cat['lum'] < combined_dict['zlum'])

        # remove dupes
        seen = {}
        dupes = []
        for item in cat['id']:
            if item in seen:
                dupes.append(item)
            seen[item] = 1

        print('removing', len(dupes), 'duplicates')
        select_keep = np.ones(len(cat['id'])).astype('bool')
        for d in dupes:
            # location of all objects with this id
            loc = np.where(cat['id'] == d)[0]
            dupe_z = cat['zredmagic'][loc]
            loc_remove = loc[dupe_z != dupe_z.max()]
            select_keep[loc_remove] = False  # set all but last value to keep

        cat = cat[select_zlum * select_inmask * select_keep]

        cols = [name for name in cat.dtype.names]
        total_length = len(cat)
        s = np.argsort(hp.ang2pix(16384, np.pi / 2. -
                                  np.radians(cat['dec']), np.radians(cat['ra']), nest=True))
        for name in cols:
            if name.lower() == 'id':
                print('coadd')
                f.create_dataset('catalog/redmagic/' + combined_dict['label'] + '/coadd_object_id', maxshape=(
                    total_length,), shape=(total_length,), dtype=int, chunks=(total_length,))
                f['catalog/redmagic/' + combined_dict['label'] +
                    '/coadd_object_id'][:] = cat[name][s]
            else:
                f.create_dataset('catalog/redmagic/' + combined_dict['label'] + '/' + name.lower(), maxshape=(
                    total_length,), shape=(total_length,), dtype=cat.dtype[name], chunks=(total_length,))
                f['catalog/redmagic/' + combined_dict['label'] +
                    '/' + name.lower()][:] = cat[name][s]

        s = np.argsort(hp.ang2pix(
            16384, np.pi / 2. - np.radians(ran_sample['dec']), np.radians(ran_sample['ra']), nest=True))
        for name in ran_sample.dtype.names:
            f.create_dataset('randoms/redmagic/' + combined_dict['label'] + '/' + name.lower(), maxshape=(
                len(ran_sample),), shape=(len(ran_sample),), dtype=ran_sample.dtype[name], chunks=(1000000,))
            f['randoms/redmagic/' + combined_dict['label'] +
                '/' + name.lower()][:] = ran_sample[name][s]
    f.close()

    return rmg_filebase + file + '.h5'


def include_dnf(f, dnffile):

    dnf = fitsio.read(dnffile)

    ngal = len(f['catalog/gold/ra'])

    dnfarray = np.zeros(ngal, dtype=dnf.dtype)
    coadd_id = f['catalog/gold/coadd_object_id'][:]

    iidx = np.argsort(coadd_id)
    cat_idx = coadd_id.searchsorted(dnf['ID'], sorter=iidx)
    dnfarray[iidx[cat_idx]] = dnf

    cols = dnfarray.dtype.names
    total_length = len(dnfarray)

    for name in cols:
        print(name)
        if name.lower() == 'id':
            try:
                f.create_dataset('catalog/dnf/unsheared/coadd_object_id', maxshape=(
                    total_length,), shape=(total_length,), dtype=int, chunks=(1000000,))
            except Exception as e:
                print(e)
                pass
            f['catalog/dnf/unsheared/coadd_object_id'][:] = dnfarray[name]
        else:
            try:
                f.create_dataset('catalog/dnf/unsheared/' + name.lower(), maxshape=(
                    total_length,), shape=(total_length,), dtype=dnfarray.dtype[name],
                    chunks=(1000000,))
            except Exception as e:
                print(e)
                pass
            f['catalog/dnf/unsheared/' + name.lower()][:] = dnfarray[name]


def generate_jk_centers_from_mask(outfile, regionfile, nrand=1e5):

    with h5py.File(outfile, 'r') as f:
        mask = f['index/mask/hpix'][:]

    nside = 4096

    pmap = np.zeros(12 * nside**2)
    pmap[mask] = 1

    pmap = hu.DensityMap('nest', pmap)

    rand_ra, rand_dec = pmap.genrand(int(nrand), system='eq')

    centers = KMeans(n_clusters=1000, random_state=0).fit(
        np.vstack([rand_ra, rand_dec]).T)

    centers = centers.cluster_centers_
    kdt = spatial.cKDTree(centers)
    dist, idx = kdt.query(centers, 2)
    centers_dist = np.zeros((1000, 3))
    centers_dist[:, :2] = centers
    centers_dist[:, 2] = dist[:, 1]

    fitsio.write(regionfile, centers_dist)


def assign_jk_regions(mastercat, regionsfile, nside=512, just_rm=False):

    if not just_rm:
        catalogs = ['catalog/gold', 'catalog/metacal/unsheared',
                    'catalog/redmagic/combined_sample_fid',
                    'catalog/redmapper/lgt20'
                    'catalog/redmapper/lgt5'
                    'randoms/redmagic/combined_sample_fid',
                    'randoms/redmapper/lgt20',
                    'randoms/redmapper/lgt5',
                    'randoms/maglim']
    else:
        catalogs = ['catalog/redmagic/combined_sample_fid',
                    'randoms/redmagic/combined_sample_fid']

    f = h5py.File(mastercat, 'r+')

    centers = fitsio.read(regionsfile)

    # assign healpix cells to regions
    pixra, pixdec = hp.pix2ang(nside, np.arange(
        12 * nside**2), nest=True, lonlat=True)
    pixcenters = np.vstack([pixra, pixdec]).T
    _, jk_idx = spatial.cKDTree(centers).query(pixcenters)

    for i, cat in enumerate(catalogs):
        try:
            cat_size = len(f[cat + '/ra'])
            ra = f[cat + '/ra'][:]
            dec = f[cat + '/dec'][:]
            pix = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)
            jk_region = jk_idx[pix]
        except:
            pass

        try:
            f.create_dataset('regions/' + cat + '/region', maxshape=(
                cat_size,), shape=(cat_size,), dtype=int, chunks=(1000000,))
            f['regions/' + cat + '/region'][:] = jk_region
        except:
            pass

    f.create_dataset('regions/centers/ra', maxshape=(len(centers),),
                     shape=(len(centers),), dtype=centers.dtype, chunks=(len(centers),))
    f['regions/centers/ra'][:] = centers[:, 0]

    f.create_dataset('regions/centers/dec', maxshape=(len(centers),),
                     shape=(len(centers),), dtype=centers.dtype, chunks=(len(centers),))
    f['regions/centers/dec'][:] = centers[:, 1]

    f.create_dataset('regions/centers/dist', maxshape=(len(centers),),
                     shape=(len(centers),), dtype=centers.dtype, chunks=(len(centers),))
    f['regions/centers/dist'][:] = centers[:, 2]

    f.create_dataset('regions/centers/number', maxshape=(len(centers),),
                     shape=(1,), dtype='>i8', chunks=(len(centers),))
    f['regions/centers/number'][:] = len(centers)

    f.close()


def make_mcal_selection(f, x_opt):

    psfmap = {'PIXEL': f['maps/hpix'][:], 'SIGNAL': f['maps/i/fwhm'][:]}

    pidx = psfmap['PIXEL'].argsort()
    psfmap['SIGNAL'] = psfmap['SIGNAL'][pidx]
    psfmap['PIXEL'] = psfmap['PIXEL'][pidx]
    del pidx

    gpix = f['catalog/gold/hpix_16384'][:] // (
        hp.nside2npix(16384) // hp.nside2npix(4096))

    psfidx = psfmap['PIXEL'].searchsorted(gpix)
    del gpix

    gpsf = psfmap['SIGNAL'][psfidx]
    del psfmap['PIXEL'], psfmap['SIGNAL']
    del psfmap

    idx = np.sqrt(f['catalog/metacal/unsheared/size']
                  [:]**2 + gpsf**2) > (x_opt[2] * gpsf)
    del gpsf

    idx &= np.abs(f['catalog/metacal/unsheared/e1'][:]) < 1
    idx &= np.abs(f['catalog/metacal/unsheared/e2'][:]) < 1
    idx &= f['catalog/gold/mag_err_r'][:] < 0.25
    idx &= f['catalog/gold/mag_err_i'][:] < 0.25
    idx &= f['catalog/gold/mag_err_z'][:] < 0.25
    idx &= f['catalog/gold/mag_i'][:] < (x_opt[0] +
                                         x_opt[1] * f['catalog/bpz/unsheared/z'][:])

    return idx


def make_altlens_selection(f, x_opt, mask_hpix, zdata='catalog/bpz/unsheared/z'):
    mag_i = f['catalog/gold/mag_i'][:]
    z = f[zdata][:]
    idx = (mag_i < (x_opt[0] * z + x_opt[1])) & (z > 0)
    del z

    idx &= (mag_i > 17.5) & (mag_i < 23)
    idx &= (f['catalog/gold/mag_err_i'][:] < 0.1)
    del mag_i

    idx = np.where(idx)[0]
    pix = hp.ang2pix(4096, f['catalog/gold/ra'][:][idx],
                     f['catalog/gold/dec'][:][idx], lonlat=True)
    idx = idx[np.in1d(pix, mask_hpix)]

    return idx


def make_master_bcc(x_opt, x_opt_altlens, outfile='./Y3_mastercat_v2_6_20_18.h5',
                    shapefile='y3v02-mcal-002-blind-v1.h5', goldfile='Y3_GOLD_2_2.h5',
                    bpzfile='Y3_GOLD_2_2_BPZ.h5', rmfile='y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22.h5',
                    mapfile='Y3_GOLD_2_2_1_maps.h5', maskfile=None, dnffile=None,
                    good=1):
    """
    Create master h5 file that links the individual catalog h5 files and
    outfile='./Y3_mastercat_v1_6_20_18.h5'; shapefile='y3v02-mcal-002-blind-v1.h5'; goldfile='Y3_GOLD_2_2.h5'; rmfile='y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22.h5'; bpzfile='Y3_GOLD_2_2_BPZ.h5'; dnffile='Y3_GOLD_2_2_DNF.h5'; mapfile='Y3_GOLD_2_2_1_maps.h5'
    """

    # still need to add mask to gold h5 file
    mask = hp.read_map(maskfile, nest=True, partial=True)
    mask = mask == good
    hpix = np.where(mask)[0].astype(int)

    try:
        with h5py.File(goldfile, 'r+') as fp:
            fp.create_dataset('masks/gold/hpix', maxshape=(np.sum(mask),),
                              shape=(np.sum(mask),), dtype=hpix.dtype,
                              chunks=(1000000,))
            fp['masks/gold/hpix'][:] = hpix
    except:
        with h5py.File(goldfile, 'r+') as fp:
            del fp['masks/gold/hpix']
            fp.create_dataset('masks/gold/hpix', maxshape=(np.sum(mask),),
                              shape=(np.sum(mask),), dtype=hpix.dtype,
                              chunks=(1000000,))
            fp['masks/gold/hpix'][:] = hpix

    #    # Open catalog h5 files for sorting by healpix id
    f = h5py.File(goldfile, 'r+')
    m = h5py.File(shapefile, 'r+')
    if bpzfile is not None:
        b = h5py.File(bpzfile, 'r+')
    else:
        b = None

    # Sort by healpix id and loop over all columns in catalogs, reordering
    s = np.argsort(f['catalog']['gold']['coadd_object_id'][:])
    for col in f['catalog']['gold'].keys():
        print(col)
        sys.stdout.flush()
        c = f['catalog']['gold'][col][:]
        f['catalog']['gold'][col][:] = c[s]

    del s

    s = np.argsort(m['catalog']['unsheared']['metacal']['coadd_object_id'][:])
    for col in m['catalog']['unsheared']['metacal'].keys():
        print(col)
        sys.stdout.flush()

        c = m['catalog']['unsheared']['metacal'][col][:]
        m['catalog']['unsheared']['metacal'][col][:] = c[s]

    del s

    if b is not None:
        s = np.argsort(b['catalog']['bpz']['coadd_object_id'][:])
        for col in b['catalog']['bpz'].keys():
            print(col)
            sys.stdout.flush()

            c = b['catalog']['bpz'][col][:]
            b['catalog']['bpz'][col][:] = c[s]

        del s

    s = np.argsort(f['catalog']['gold']['hpix_16384'][:])
    for col in f['catalog']['gold'].keys():
        print(col)
        sys.stdout.flush()

        c = f['catalog']['gold'][col][:]
        f['catalog']['gold'][col][:] = c[s]

    for col in m['catalog']['unsheared']['metacal'].keys():
        print(col)
        sys.stdout.flush()

        c = m['catalog']['unsheared']['metacal'][col][:]
        m['catalog']['unsheared']['metacal'][col][:] = c[s]

    if b is not None:
        for col in b['catalog']['bpz'].keys():
            print(col)
            sys.stdout.flush()

            c = b['catalog']['bpz'][col][:]
            b['catalog']['bpz'][col][:] = c[s]
#
    f.close()
    m.close()
    if b is not None:
        b.close()

    # Create master h5 file and softlink all external data tables inside it
    try:
        f = h5py.File(outfile, mode='r+')
    except:
        f = h5py.File(outfile, mode='w')

    f['/catalog/metacal/unsheared'] = h5py.ExternalLink(
        shapefile, "/catalog/unsheared/metacal")
    f['/catalog/gold'] = h5py.ExternalLink(goldfile, "/catalog/gold")
    if b is not None:
        f['/catalog/bpz/unsheared'] = h5py.ExternalLink(bpzfile, "/catalog/bpz")

    f['/catalog/redmagic'] = h5py.ExternalLink(rmfile, "/catalog/redmagic")
    f['/catalog/redmapper'] = h5py.ExternalLink(
        rmfile, "/catalog/redmapper")
    f['/randoms/redmagic'] = h5py.ExternalLink(rmfile, "/randoms/redmagic")
    f['/masks/gold'] = h5py.ExternalLink(goldfile, "/masks/gold")
    f['/masks/redmagic'] = h5py.ExternalLink(rmfile, "/masks/redmagic")
    f['/maps'] = h5py.ExternalLink(mapfile, "/maps")

    f['catalog/metacal/unsheared/ra'] = h5py.ExternalLink(
        goldfile, 'catalog/gold/ra')
    f['catalog/metacal/unsheared/dec'] = h5py.ExternalLink(
        goldfile, 'catalog/gold/dec')
    f['catalog/metacal/unsheared/tra'] = h5py.ExternalLink(
        goldfile, 'catalog/gold/tra')
    f['catalog/metacal/unsheared/tdec'] = h5py.ExternalLink(
        goldfile, 'catalog/gold/tdec')

    # include index coadd id array in master file
    coadd = f['catalog/gold/coadd_object_id'][:]
    # Need sorted by coadd id for matching to redmagic later
    s_ = np.argsort(coadd)
    total_length = len(coadd)
    f.create_dataset('index/coadd_object_id', maxshape=(total_length,),
                     shape=(total_length,), dtype=int, chunks=(1000000,))
    f['index/coadd_object_id'][:] = coadd

    # construct indices to map gold onto the shape catalog

    f.create_dataset('index/metacal/match_gold', maxshape=(total_length,),
                     shape=(total_length,), dtype=int, chunks=(1000000,))
    f['index/metacal/match_gold'][:] = np.arange(total_length)
    gpix = f['catalog']['gold']['hpix_16384'][:]

    # construct indices to map gold onto the photoz catalogs
    if b is not None:
        for x in ['bpz']:
            s = np.arange(total_length)
            f.create_dataset('index/' + x + '/match_gold', maxshape=(len(s),),
                             shape=(len(s),), dtype=int, chunks=(1000000,))
            f['index/' + x + '/match_gold'][:] = s

    # construct gold level selection flags (in index form)
    ngal = len(f['catalog/gold/flags_gold'])
    f.create_dataset('index/gold/select', maxshape=(ngal,),
                     shape=(ngal,), dtype=int, chunks=(1000000,))
    f['index/gold/select'][:] = np.arange(ngal)

    mask = np.in1d(f['masks/redmagic/combined_sample_fid/hpix']
                   [:], f['masks/gold/hpix'][:])
    s = np.argsort(f['masks/redmagic/combined_sample_fid/hpix'][:][mask])
    for col in f['masks/redmagic/combined_sample_fid/'].keys():
        c = f['masks/redmagic/combined_sample_fid/' + col][:][mask][s]
        f.create_dataset('index/mask/' + col, maxshape=(len(c),),
                         shape=(len(c),), dtype=int, chunks=(1000000,))
        f['index/mask/' + col][:] = c

    del mask, s

    # construct indices to map gold onto the redmagic catalogs
    for table in f['catalog/redmagic'].keys():
        # del(f['index/redmagic/'+table+'/match_gold'])
        s = coadd.searchsorted(
            f['catalog/redmagic/' + table + '/coadd_object_id'][:], sorter=s_)
        f.create_dataset('index/redmagic/' + table + '/match_gold',
                         maxshape=(len(s),), shape=(len(s),), dtype=int, chunks=(10000,))
        f['index/redmagic/' + table + '/match_gold'][:] = s_[s]

    # Add masking from joint mask to redmagic
    for table in f['catalog/redmagic'].keys():
        gpix = hp.ang2pix(16384, f['catalog/redmagic/' + table + '/ra'][:],
                          f['catalog/redmagic/' + table + '/dec'][:], lonlat=True, nest=True)
        mask = np.in1d(gpix // (hp.nside2npix(16384) // hp.nside2npix(4096)),
                       f['index/mask/hpix'][:], assume_unique=False)
        f.create_dataset('index/redmagic/' + table + '/select', maxshape=(
            np.sum(mask),), shape=(np.sum(mask),), dtype=int, chunks=(100000,))
        f['index/redmagic/' + table + '/select'][:] = np.where(mask)[0]

    for table in f['randoms/redmagic'].keys():
        rpix = hp.ang2pix(16384, np.pi / 2. - np.radians(f['randoms/redmagic/' + table + '/dec'][:]), np.radians(
            f['randoms/redmagic/' + table + '/ra'][:]), nest=True)
        mask = np.in1d(rpix // (hp.nside2npix(16384) // hp.nside2npix(4096)),
                       f['index/mask/hpix'][:], assume_unique=False)
        f.create_dataset('index/redmagic/' + table + '/random_select', maxshape=(
            np.sum(mask),), shape=(np.sum(mask),), dtype=int, chunks=(100000,))
        f['index/redmagic/' + table + '/random_select'][:] = np.where(mask)[0]

    # construct gold-pz level selection flags (in index form)
    if b is not None:
        for x in ['bpz']:

            f.create_dataset('index/' + x + '/select', maxshape=(ngal,),
                             shape=(ngal,), dtype=int, chunks=(1000000,))
            f['index/' + x + '/select'][:] = np.arange(ngal)

    # construct shape catalog selection flags for default expected selection (in index form) including redmagic joint masking

    for table, suffix in tuple(zip(['unsheared'], [''])):
        idx = make_mcal_selection(f, x_opt)
        idx &= np.in1d(f['catalog/gold/hpix_16384'][:] //
                       (hp.nside2npix(16384) // hp.nside2npix(4096)), f['index/mask/hpix'][:])
        try:
            f.create_dataset('index/select' + suffix, maxshape=(
                np.sum(idx),), shape=(np.sum(idx),), dtype=int, chunks=(1000000,))
        except:
            del f['index/select' + suffix]
            f.create_dataset('index/select' + suffix, maxshape=(
                np.sum(idx),), shape=(np.sum(idx),), dtype=int, chunks=(1000000,))

        f['index/select' + suffix][:] = np.where(idx)[0]

        del idx

    if dnffile is not None:
        include_dnf(f, dnffile)
        idx = make_altlens_selection(f, x_opt_altlens, zdata='catalog/dnf/unsheared/z_mc')
    else:
        idx = make_altlens_selection(f, x_opt_altlens, zdata='catalog/bpz/unsheared/z')

    idx &= np.in1d(f['catalog/gold/hpix_16384'][:] //
                   (hp.nside2npix(16384) // hp.nside2npix(4096)), f['index/mask/hpix'][:])

    try:
        f.create_dataset('index/maglim/select', maxshape=(
                         np.sum(idx),), shape=(np.sum(idx),), dtype=int, chunks=(1000000,))
    except:
        del f['index/maglim/select']
        f.create_dataset('index/maglim/select', maxshape=(
                         np.sum(idx),), shape=(np.sum(idx),), dtype=int, chunks=(1000000,))

    f['index/maglim/select'][:] = np.where(idx)[0]

    make_maglim_randoms(f, rmfile)

    f['/randoms/maglim'] = h5py.ExternalLink(rmfile, "/randoms/maglim")

    f.close()


def match_shape_noise(filename, mcalfilename, zbins, sigma_e_data):

    with h5py.File(filename, 'r') as f:
        with h5py.File(mcalfilename, 'r+') as mf:

            size_tot = len(f['catalog/metacal/unsheared/e1'][:])
            idx = f['index/select'][:]
            zmean = f['catalog/bpz/unsheared/zmean_sof'][:][idx]
            e1 = f['catalog/metacal/unsheared/e1'][:][idx]
            e2 = f['catalog/metacal/unsheared/e2'][:][idx]

            try:
                mf.create_dataset('catalog/unsheared/metacal/e1_matched_se', maxshape=(
                    size_tot,), shape=(size_tot,), dtype=e1.dtype, chunks=(1000000,))
                mf.create_dataset('catalog/unsheared/metacal/e2_matched_se', maxshape=(
                    size_tot,), shape=(size_tot,), dtype=e1.dtype, chunks=(1000000,))
            except:
                del mf['catalog/unsheared/metacal/e1_matched_se'], mf['catalog/unsheared/metacal/e2_matched_se']

                mf.create_dataset('catalog/unsheared/metacal/e1_matched_se', maxshape=(
                    size_tot,), shape=(size_tot,), dtype=e1.dtype, chunks=(1000000,))
                mf.create_dataset('catalog/unsheared/metacal/e2_matched_se', maxshape=(
                    size_tot,), shape=(size_tot,), dtype=e1.dtype, chunks=(1000000,))

            e1_sn = np.zeros(len(f['catalog/metacal/unsheared/e1']))
            e2_sn = np.zeros(len(f['catalog/metacal/unsheared/e1']))

            for i in range(len(zbins) - 1):
                idxi = (zbins[i] < zmean) & (zmean < zbins[i + 1])
                sigma_e = np.std(e1[idxi])
                ds = np.sqrt((sigma_e_data[i]**2 - sigma_e**2))

                e1_sn[idx[idxi]] = e1[idxi] + ds * \
                    np.random.randn(np.sum(idxi))
                e2_sn[idx[idxi]] = e2[idxi] + ds * \
                    np.random.randn(np.sum(idxi))

                print(e1_sn[idx[idxi]])

            mf['catalog/unsheared/metacal/e1_matched_se'][:] = e1_sn
            mf['catalog/unsheared/metacal/e2_matched_se'][:] = e2_sn


def generateAngularRandoms(hpix, nrand, nside,
                           nest=True):

    fmask = np.zeros(12 * nside**2)
    fmask[hpix] = 1

    if nest:
        pmap = hu.DensityMap('nest', fmask)
    else:
        pmap = hu.DensityMap('ring', fmask)

    rand_ra, rand_dec = pmap.genrand(nrand, system='eq')

    return rand_ra, rand_dec


def make_maglim_randoms(f, rmgfile):

    with h5py.File(rmgfile, 'r+') as fr:

        maglim_select = f['index/maglim/select'][:]
        mag_i_maglim = f['catalog/gold/mag_i'][:][maglim_select]
        zmean_maglim = f['catalog/dnf/unsheared/z_mean'][:][maglim_select]

        mag_i_max = np.max(mag_i_maglim[zmean_maglim < 1.05])

        del mag_i_maglim, zmean_maglim

        mask_hpix = f['maps/buzzard/hpix'][:]
        depth_sof_i = f['maps/buzzard/i/sof_depth'][:]

        mask_hpix = mask_hpix[depth_sof_i < mag_i_max]

        del depth_sof_i

        f.create_dataset('masks/maglim/hpix', maxshape=(len(mask_hpix),), shape=(len(mask_hpix),), dtype=mask_hpix.dtype, chunks=(100000,))
        f['masks/maglim/hpix'][:] = mask_hpix

        print('generating randoms')
        rand_ra, rand_dec = generateAngularRandoms(
            mask_hpix, int(len(maglim_select) * 20), 4096)
        total_length = len(rand_ra)

        print('sorting')
        s = np.argsort(hp.ang2pix(16384, np.pi / 2. -
                                  np.radians(rand_dec), np.radians(rand_ra), nest=True))

        print('creating datasets')
        try:
            fr.create_dataset('randoms/maglim/ra', maxshape=(total_length,),
                              shape=(total_length,), dtype=rand_ra.dtype, chunks=(1000000,))
            fr.create_dataset('randoms/maglim/dec', maxshape=(total_length,),
                              shape=(total_length,), dtype=rand_ra.dtype, chunks=(1000000,))
            fr.create_dataset('randoms/maglim/z', maxshape=(total_length,),
                              shape=(total_length,), dtype=rand_ra.dtype, chunks=(1000000,))
            fr.create_dataset('randoms/maglim/weight', maxshape=(total_length,),
                              shape=(total_length,), dtype=rand_ra.dtype, chunks=(1000000,))
        except:
            del fr['randoms/maglim/ra'], fr['randoms/maglim/dec'], fr['randoms/maglim/z'], fr['randoms/maglim/weight']
            fr.create_dataset('randoms/maglim/ra', maxshape=(total_length,),
                              shape=(total_length,), dtype=rand_ra.dtype, chunks=(1000000,))
            fr.create_dataset('randoms/maglim/dec', maxshape=(total_length,),
                              shape=(total_length,), dtype=rand_ra.dtype, chunks=(1000000,))
            fr.create_dataset('randoms/maglim/z', maxshape=(total_length,),
                              shape=(total_length,), dtype=rand_ra.dtype, chunks=(1000000,))
            fr.create_dataset('randoms/maglim/weight', maxshape=(total_length,),
                              shape=(total_length,), dtype=rand_ra.dtype, chunks=(1000000,))

        print('writing')
        fr['randoms/maglim/ra'][:] = rand_ra[s]
        fr['randoms/maglim/dec'][:] = rand_dec[s]
        fr['randoms/maglim/z'][:] = np.random.choice(
            f['catalog/dnf/unsheared/z_mc'][:][maglim_select], size=len(rand_ra[s]))
        fr['randoms/maglim/weight'][:] = np.ones_like(rand_ra[s])


if __name__ == '__main__':

    cfgfile = sys.argv[1]

    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp)

    outfile = cfg['outfile']
    mcalfile = cfg['mcalfile']
    goldfile = cfg['goldfile']
    bpzfile = cfg['bpzfile']
    rmfile = cfg['rmfile']
    rmg_filebase = cfg['redmagic_filebase']
    rmp_filebase = cfg['redmapper_filebase']
    maskfile = cfg['footprint_maskfile']
    regionfile = cfg['regionfile']
    x_opt = cfg['x_opt']
    x_opt_altlens = cfg['x_opt_altlens']
    mapfile = cfg['mapfile']

    if 'dnffile' in cfg.keys():
        dnffile = cfg['dnffile']
    else:
        dnffile = None

    just_rm = cfg.pop('just_rm', False)

    goodmask_value = int(cfg.pop('goodmask_value', 1))

    h5rmfile = convert_rm_to_h5(rmg_filebase=rmg_filebase, rmp_filebase=rmp_filebase,
                                file=rmfile)

    if not just_rm:
        make_master_bcc(x_opt, x_opt_altlens, outfile=outfile, shapefile=mcalfile, goldfile=goldfile, bpzfile=bpzfile, rmfile=h5rmfile,
                        maskfile=maskfile, good=goodmask_value, mapfile=mapfile, dnffile=dnffile)

        match_shape_noise(outfile, mcalfile, cfg['zbins'], cfg['sigma_e_data'])

    if os.path.exists(regionfile):
        assign_jk_regions(outfile, regionfile)
    else:
        generate_jk_centers_from_mask(outfile)
        assign_jk_regions(outfile, regionfile)

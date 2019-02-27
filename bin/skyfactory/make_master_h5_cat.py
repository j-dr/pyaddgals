import numpy as np
import h5py
import os
import sys
import healpy as hp
import fitsio
from fast3tree import fast3tree

cats_redmagic = ['redmagic_highdens_0.5-10',
                 'redmagic_highlum_1.0-04', 'redmagic_higherlum_1.5-01']
cats_redmapper = ['lgt20_vl02_catalog', 'lgt5_vl02_catalog',
                  'lgt20_vl02_catalog_members', 'lgt5_vl02_catalog_members']
cats_redmapper_random = [
    'randcat_z0.10-0.95_lgt005_vl02', 'randcat_z0.10-0.95_lgt020_vl02']
cats_redmagic_table = ['highdens', 'highlum', 'higherlum']
cats_redmapper_table = ['lgt20', 'lgt5', 'lgt20/members', 'lgt5/members']
cats_redmapper_random_table = ['lgt5', 'lgt20']

# Details for combining redmagic samples
combined_dict = {
    'samples': ['redmagic_highdens_0.5-10', 'redmagic_highlum_1.0-04'],
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

    # Create h5 file
    f = h5py.File(rmg_filebase + file + '.h5', 'w')
    # Loop over redmagic cats fits files and dump into h5
    for i in range(len(cats_redmagic)):
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
                                  np.radians(cat['DEC']), np.radians(cat['RA']), nest=True))
        # Loop over columns and stick in h5 file
        for name in cols:
            if name.lower() == 'coadd_objects_id':
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
        mask = fitsio.FITS(rmg_filebase + file + '_' +
                           cats_redmagic[i][:-3] + '_vlim_zmask.' + file_ext)[1].read()
        cols = [name for name in mask.dtype.names]
        total_length = len(mask)
        mask['HPIX'] = hp.ring2nest(4096, mask['HPIX'])
        s = np.argsort(mask['HPIX'])
        for name in cols:
            f.create_dataset('masks/redmagic/' + cats_redmagic_table[i] + '/' + name.lower(), maxshape=(
                total_length,), shape=(total_length,), dtype=mask.dtype[name], chunks=(100000,))
            f['masks/redmagic/' + cats_redmagic_table[i] +
                '/' + name.lower()][:] = mask[name][s]

    # Loop over randoms and put in h5 file
    for i in range(len(cats_redmagic)):
        cat = fitsio.FITS(rmg_filebase + file + '_' +
                          cats_redmagic[i] + '_randoms.' + file_ext)[1].read()
        cols = [name for name in cat.dtype.names]
        total_length = fitsio.FITS(
            rmg_filebase + file + '_' + cats_redmagic[i] + '_randoms.' + file_ext)[1].read_header()['NAXIS2']
        s = np.argsort(hp.ang2pix(16384, np.pi / 2. -
                                  np.radians(cat['DEC']), np.radians(cat['RA']), nest=True))
        for name in cols:
            f.create_dataset('randoms/redmagic/' + cats_redmagic_table[i] + '/' + name.lower(), maxshape=(
                total_length,), shape=(total_length,), dtype=cat.dtype[name], chunks=(1000000,))
            f['randoms/redmagic/' + cats_redmagic_table[i] +
                '/' + name.lower()][:] = cat[name][s]

    # Loop over redmapper cats and put in h5 file
#    for i in range(len(cats_redmapper)):
#        cat  = fitsio.FITS(rmp_filebase+file+'_'+cats_redmapper[i]+'.'+file_ext)[1].read()
#        cols = [name for name in cat.dtype.names]
#        total_length = fitsio.FITS(file+'_'+cats_redmapper[i]+'.'+file_ext)[1].read_header()['NAXIS2']
#        s = np.argsort(hp.ang2pix(16384, np.pi/2.-np.radians(cat['DEC']),np.radians(cat['RA']), nest=True))
#        for name in cols:
#            f.create_dataset( 'catalog/redmapper/'+cats_redmapper_table[i]+'/'+name.lower(), maxshape=(total_length,), shape=(total_length,), dtype=cat.dtype[name], chunks=(total_length,) )
#            f['catalog/redmapper/'+cats_redmapper_table[i]+'/'+name.lower()][:] = cat[name][s]

    # Loop over redmapper randoms and put in h5 file
#    for i in range(len(cats_redmapper_random)):
#        cat  = fitsio.FITS(rmp_filebase+file+'_'+cats_redmapper_random[i]+'.'+file_ext)[1].read()
#        cols = [name for name in cat.dtype.names]
#        total_length = fitsio.FITS(rmp_filebase+file+'_'+cats_redmapper_random[i]+'.'+file_ext)[1].read_header()['NAXIS2']
#        s = np.argsort(hp.ang2pix(16384, np.pi/2.-np.radians(cat['DEC']),np.radians(cat['RA']), nest=True))
#        for name in cols:
#            f.create_dataset( 'randoms/redmapper/'+cats_redmapper_random_table[i]+'/'+name.lower(), maxshape=(total_length,), shape=(total_length,), dtype=cat.dtype[name], chunks=(total_length,) )
#            f['randoms/redmapper/'+cats_redmapper_random_table[i]+'/'+name.lower()][:] = cat[name][s]

    # Make combined catalog version and add in new h5 table
    if make_combined:
        #        print 'combined redmagic'
        binedges_all = np.unique(np.hstack(combined_dict['binedges']))
        # combined_mask
        for i in range(len(combined_dict['samples'])):
            zmax_cut = combined_dict['binedges'][i][-1]
            if i == 0:
                mask_master = fitsio.FITS(
                    rmg_filebase + file + '_' + combined_dict['samples'][i][:-3] + '_vlim_zmask.' + file_ext)[1].read()
                mask_master['HPIX'] = hp.ring2nest(4096, mask_master['HPIX'])
                select_zmax = (mask_master['ZMAX'] > zmax_cut)
                mask_master = mask_master[select_zmax]
            else:
                mask = fitsio.FITS(
                    rmg_filebase + file + '_' + combined_dict['samples'][i][:-3] + '_vlim_zmask.' + file_ext)[1].read()
                mask['HPIX'] = hp.ring2nest(4096, mask['HPIX'])
                select_zmax = (mask['ZMAX'] > zmax_cut)
                badpix = mask['HPIX'][~select_zmax]
                select_badpix = np.in1d(mask_master['HPIX'], badpix)
                mask_master = mask_master[~select_badpix]

        select_fracdet = (mask_master['FRACGOOD'] > combined_dict['fracgood'])
        mask_master = mask_master[select_fracdet]
        cols = [name for name in mask_master.dtype.names]
        total_length = len(mask_master)
        s = np.argsort(mask_master['HPIX'])
        for name in cols:
            f.create_dataset('masks/redmagic/' + combined_dict['label'] + '/' + name.lower(), maxshape=(
                total_length,), shape=(total_length,), dtype=mask_master.dtype[name], chunks=(100000,))
            f['masks/redmagic/' + combined_dict['label'] +
                '/' + name.lower()][:] = mask_master[name][s]

        # combined catalog
        for i in range(len(combined_dict['samples'])):
            cat_sample = fitsio.FITS(
                rmg_filebase + file + '_' + combined_dict['samples'][i] + '.' + file_ext)[1].read()
            ran_sample_ = fitsio.FITS(
                rmg_filebase + file + '_' + combined_dict['samples'][i] + '_randoms.' + file_ext)[1].read()
            binedges = combined_dict['binedges'][i]
            select_zrange = (
                cat_sample['ZREDMAGIC'] >= binedges[0]) * (cat_sample['ZREDMAGIC'] < binedges[-1])
            cat_sample = cat_sample[select_zrange]
            select_zrange = (
                ran_sample_['Z'] >= binedges[0]) * (ran_sample_['Z'] < binedges[-1])
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
            4096, np.pi / 2. - np.radians(ran_sample['DEC']), np.radians(ran_sample['RA']), nest=True)
        select_inmask = np.in1d(catpix, mask_master['HPIX'])
        ran_sample = ran_sample[select_inmask]

        catpix = hp.ang2pix(
            4096, np.pi / 2. - np.radians(cat['DEC']), np.radians(cat['RA']), nest=True)
        select_inmask = np.in1d(catpix, mask_master['HPIX'])

        # apply ZLUM cut
        select_zlum = (cat['ZLUM'] < combined_dict['zlum'])

        # remove dupes
        seen = {}
        dupes = []
        for item in cat['COADD_OBJECTS_ID']:
            if item in seen:
                dupes.append(item)
            seen[item] = 1

        print('removing', len(dupes), 'duplicates')
        select_keep = np.ones(len(cat['COADD_OBJECTS_ID'])).astype('bool')
        for d in dupes:
            # location of all objects with this id
            loc = np.where(cat['COADD_OBJECTS_ID'] == d)[0]
            dupe_z = cat['ZREDMAGIC'][loc]
            loc_remove = loc[dupe_z != dupe_z.max()]
            select_keep[loc_remove] = False  # set all but last value to keep

        cat = cat[select_zlum * select_inmask * select_keep]

        cols = [name for name in cat.dtype.names]
        total_length = len(cat)
        s = np.argsort(hp.ang2pix(16384, np.pi / 2. -
                                  np.radians(cat['DEC']), np.radians(cat['RA']), nest=True))
        for name in cols:
            if name.lower() == 'coadd_objects_id':
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
            16384, np.pi / 2. - np.radians(ran_sample['DEC']), np.radians(ran_sample['RA']), nest=True))
        for name in ran_sample.dtype.names:
            f.create_dataset('randoms/redmagic/' + combined_dict['label'] + '/' + name.lower(), maxshape=(
                len(ran_sample),), shape=(len(ran_sample),), dtype=ran_sample.dtype[name], chunks=(1000000,))
            f['randoms/redmagic/' + combined_dict['label'] +
                '/' + name.lower()][:] = ran_sample[name][s]
    f.close()

    return rmg_filebase + file + '.h5'


def assign_jk_regions(mastercat, regionsfile, nside=512):

    f = h5py.File(mastercat, 'r+')

    centers = fitsio.read(regionsfile)

    # assign healpix cells to regions
    regionmap = np.zeros(12 * nside**2)
    pixra, pixdec = hp.pix2ang(nside, np.arange(
        12 * nside**2), nest=True, lonlat=True)
    pixcenters = np.vstack([pixra, pixdec]).T
    _, jk_idx = spatial.cKDTree(centers).query(pixcenters)

    gold_size = len(f['catalog/gold/coadd_object_id'])
    rmg_size = len(f['catalog/redmagic/combined_sample_fid/coadd_object_id'])
    rand_size = len(f['randoms/redmagic/combined_sample_fid/ra'])

    cat_size = [gold_size, rmg_size, rand_size]

    for i, cat in enumerate(['catalog/gold', 'catalog/redmagic/combined_sample_fid', 'randoms/redmagic/combined_sample_fid']):
        ra = f[cat + '/ra'][:]
        dec = f[cat + '/dec'][:]
        pix = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)
        jk_region = jk_idx[pix]
        print(jk_region)
        try:
            f.create_dataset('regions/' + cat + '/region', maxshape=(
                cat_size[i],), shape=(cat_size[i],), dtype=int, chunks=(1000000,))
        except:
            pass
        f['regions/' + cat + '/region'][:] = jk_region

    f.create_dataset('regions/centers/ra', maxshape=(len(centers),), shape=(len(centers),), dtype=centers.dtype, chunks=(len(centers),))
    f['regions/centers/ra'][:] = centers[:, 0]

    f.create_dataset('regions/centers/dec', maxshape=(len(centers),), shape=(len(centers),), dtype=centers.dtype, chunks=(len(centers),))
    f['regions/centers/dec'][:] = centers[:, 1]

    f.create_dataset('regions/centers/dist', maxshape=(len(centers),), shape=(len(centers),), dtype=centers.dtype, chunks=(len(centers),))
    f['regions/centers/dist'][:] = centers[:, 2]

    f.create_dataset('regions/centers/dist', maxshape=(len(centers),), shape=(1,), dtype='>i8', chunks=(len(centers),))
    f['regions/centers/number'][:] = len(centers)

    f.close()


def make_master_bcc(outfile='./Y3_mastercat_v2_6_20_18.h5',
                    shapefile='y3v02-mcal-002-blind-v1.h5', goldfile='Y3_GOLD_2_2.h5',
                    bpzfile='Y3_GOLD_2_2_BPZ.h5', rmfile='y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22.h5',
                    mapfile='Y3_GOLD_2_2_1_maps.h5', maskfile=None):
    """
    Create master h5 file that links the individual catalog h5 files and
    outfile='./Y3_mastercat_v1_6_20_18.h5'; shapefile='y3v02-mcal-002-blind-v1.h5'; goldfile='Y3_GOLD_2_2.h5'; rmfile='y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22.h5'; bpzfile='Y3_GOLD_2_2_BPZ.h5'; dnffile='Y3_GOLD_2_2_DNF.h5'; mapfile='Y3_GOLD_2_2_1_maps.h5'
    """

    # Open catalog h5 files for sorting by healpix id
    f = h5py.File(goldfile, 'r+')
    b = h5py.File(bpzfile, 'r+')
    m = h5py.File(shapefile, 'r+')

    # Sort by healpix id and loop over all columns in catalogs, reordering
    s = np.argsort(f['catalog']['gold']['hpix_16384'][:])
    for col in f['catalog']['gold'].keys():
        print(col)
        c = f['catalog']['gold'][col][:]
        f['catalog']['gold'][col][:] = c[s]

    for col in m['catalog']['unsheared']['metacal'].keys():
        c = m['catalog']['unsheared']['metacal'][col][:]
        m['catalog']['unsheared']['metacal'][col][:] = c[s]

    for col in b['catalog']['bpz'].keys():
        c = b['catalog']['bpz'][col][:]
        b['catalog']['bpz'][col][:] = c[s]

    # Close h5 files to dump cache
    f.close()
    b.close()
    m.close()

    mask = hp.read_map(maskfile)
    mask = mask == 1
    hpix = np.where(mask)[0].astype(int)
    with h5py.File(goldfile, 'r+') as fp:
        fp.create_dataset('masks/gold/hpix', maxshape=(np.sum(mask),),
                          shape=(np.sum(mask),), dtype=hpix.dtype, chunks=(1000000,))
        fp['masks/gold/hpix'][:] = hpix

    # Create master h5 file and softlink all external data tables inside it
    f = h5py.File(outfile, mode='w')
    f['/catalog/metacal/unsheared'] = h5py.ExternalLink(
        shapefile, "/catalog/unsheared/metacal")
    f['/catalog/gold'] = h5py.ExternalLink(goldfile, "/catalog/gold")
    f['/catalog/bpz/unsheared'] = h5py.ExternalLink(bpzfile, "/catalog/bpz")
    f['/catalog/redmagic'] = h5py.ExternalLink(rmfile,    "/catalog/redmagic")
    f['/catalog/redmapper'] = h5py.ExternalLink(
        rmfile,    "/catalog/redmapper")
    f['/randoms'] = h5py.ExternalLink(rmfile,    "/randoms")
    f['/masks/gold'] = h5py.ExternalLink(goldfile,  "/masks/gold")
    f['/masks/redmagic'] = h5py.ExternalLink(rmfile,    "/masks/redmagic")
#    f['/maps']              = h5py.ExternalLink(mapfile,   "/maps")

    # include index coadd id array in master file
    coadd = f['catalog/gold/coadd_object_id'][:]
    # Need sorted by coadd id for matching to redmagic later
    s_ = np.argsort(coadd)
    total_length = len(coadd)
    f.create_dataset('index/coadd_object_id', maxshape=(total_length,),
                     shape=(total_length,), dtype=int, chunks=(1000000,))
    f['index/coadd_object_id'][:] = coadd

    # construct indices to map gold onto the shape catalog
    idx = np.where(f['catalog/metacal/unsheared/flags'][:] < 2**28)[0]
    f.create_dataset('index/metacal/match_gold', maxshape=(len(idx),),
                     shape=(len(idx),), dtype=int, chunks=(1000000,))
    f['index/metacal/match_gold'][:] = idx

    gpix = f['catalog']['gold']['hpix_16384'][:]

    # construct indices to map gold onto the photoz catalogs
    for x in ['bpz']:
        s = np.arange(len(coadd))
        f.create_dataset('index/' + x + '/match_gold', maxshape=(len(s),),
                         shape=(len(s),), dtype=int, chunks=(1000000,))
        f['index/' + x + '/match_gold'][:] = s

    # construct gold level selection flags (in index form)
    goldflag = f['catalog/gold/flags_gold'][:]
#    mask = np.in1d(gpix // (hp.nside2npix(16384) // hp.nside2npix(4096)), f['index/mask/hpix'][:], assume_unique=False)
    c = np.where((goldflag == 0))[0]
    f.create_dataset('index/gold/select', maxshape=(len(c),),
                     shape=(len(c),), dtype=int, chunks=(1000000,))
    f['index/gold/select'][:] = c

    mask = np.in1d(f['masks/redmagic/combined_sample_fid/hpix']
                   [:], f['masks/gold/hpix'][:])
    s = np.argsort(f['masks/redmagic/combined_sample_fid/hpix'][:][mask])
    for col in f['masks/redmagic/combined_sample_fid/'].keys():
        c = f['masks/redmagic/combined_sample_fid/' + col][:][mask][s]
        f.create_dataset('index/mask/' + col, maxshape=(len(s),),
                         shape=(len(s),), dtype=c.dtype, chunks=(1000000,))
        f['index/mask/' + col][:] = c

    # construct indices to map gold onto the redmagic catalogs
    for table in f['catalog/redmagic'].keys():
        # del(f['index/redmagic/'+table+'/match_gold'])
        s = coadd.searchsorted(
            f['catalog/redmagic/' + table + '/coadd_object_id'][:], sorter=s_)
        f.create_dataset('index/redmagic/' + table + '/match_gold',
                         maxshape=(len(s),), shape=(len(s),), dtype=int, chunks=(10000,))
        f['index/redmagic/' + table + '/match_gold'][:] = s_[s]

    gpix = f['catalog']['gold']['hpix_16384'][:]
    # Add masking from joint mask to redmagic
    for table in f['catalog/redmagic'].keys():
        mask = np.in1d(gpix[f['index/redmagic/' + table + '/match_gold'][:]] // (hp.nside2npix(
            16384) // hp.nside2npix(4096)), f['index/mask/hpix'][:], assume_unique=False)
        f.create_dataset('index/redmagic/' + table + '/select',  maxshape=(
            np.sum(mask),), shape=(np.sum(mask),), dtype=int, chunks=(100000,))
        f['index/redmagic/' + table + '/select'][:] = np.where(mask)[0]

    for table in f['randoms/redmagic'].keys():
        rpix = hp.ang2pix(16384, np.pi / 2. - np.radians(f['randoms/redmagic/' + table + '/dec'][:]), np.radians(
            f['randoms/redmagic/' + table + '/ra'][:]), nest=True)
        mask = np.in1d(rpix // (hp.nside2npix(16384) // hp.nside2npix(4096)),
                       f['index/mask/hpix'][:], assume_unique=False)
        f.create_dataset('index/redmagic/' + table + '/random_select',  maxshape=(
            np.sum(mask),), shape=(np.sum(mask),), dtype=int, chunks=(100000,))
        f['index/redmagic/' + table + '/random_select'][:] = np.where(mask)[0]

    # construct gold-pz level selection flags (in index form)
    for x in ['bpz']:
        c = np.where((goldflag == 0))[0]
        f.create_dataset('index/' + x + '/select', maxshape=(len(c),),
                         shape=(len(c),), dtype=int, chunks=(1000000,))
        f['index/' + x + '/select'][:] = c

    # construct shape catalog selection flags for default expected selection (in index form), both with and without gold flags
    flags = f['catalog/metacal/unsheared/flags'][:]
    for table, suffix in tuple(zip(['unsheared'], [''])):
        idx = (flags == 0)
        f.create_dataset('index/metacal/' + table + '/select', maxshape=(
            np.sum(idx),), shape=(np.sum(idx),), dtype=int, chunks=(1000000,))
        f['index/metacal/' + table + '/select'][:] = np.where(idx)[0]
        idx = (goldflag == 0) & idx
        f.create_dataset('index/select' + suffix, maxshape=(np.sum(idx),),
                         shape=(np.sum(idx),), dtype=int, chunks=(1000000,))
        f['index/select' + suffix][:] = np.where(idx)[0]

    mask = hp.read_map(maskfile)
    mask = mask == 1
    hpix = np.where(mask)[0].astype(int)
    with h5py.File(goldfile, 'r+') as fp:
        fp.create_dataset('masks/gold/hpix', maxshape=(np.sum(mask),),
                          shape=(np.sum(mask),), dtype=hpix.dtype, chunks=(1000000,))
        fp['masks/gold/hpix'][:] = hpix

    f.close()


if __name__ == '__main__':

    # '/global/cscratch1/sd/jderose/BCC/Chinchilla/Herd/Chinchilla-3/sampleselection/Y3/Y3_mastercat_w_rmg_b3_v1.9.2.h5'
    outfile = sys.argv[1]
    # '/global/cscratch1/sd/jderose/BCC/Chinchilla/Herd/Chinchilla-3/sampleselection/Y3/Buzzard_v1.9.2_Y3_shape.h5'
    mcalfile = sys.argv[2]
    # '/global/cscratch1/sd/jderose/BCC/Chinchilla/Herd/Chinchilla-3/sampleselection/Y3/Buzzard_v1.9.2_Y3_gold.h5'
    goldfile = sys.argv[3]
    # '/global/cscratch1/sd/jderose/BCC/Chinchilla/Herd/Chinchilla-3/sampleselection/Y3/Buzzard_v1.9.2_Y3_bpz.h5'
    bpzfile = sys.argv[4]
    # '/global/cscratch1/sd/jderose/buzzard-3_1.6_y3_run_redmapper_v6.4.20new.h5'
    rmfile = sys.argv[5]
    rmg_filebase = sys.argv[6]
    rmp_filebase = sys.argv[7]
    # ' /global/cscratch1/sd/jderose/BCC/Chinchilla/Herd/Chinchilla-3/sampleselection/Y3/jk_regions.fits'
    regionfile = sys.argv[7]
    # '/global/homes/j/jderose/des/jderose/SkyFactory-config//SampleSelection/y3a2_footprint_griz_1exp_v2.0.fits.gz'
    maskfile = sys.argv[8]

    h5rmfile = convert_rm_to_h5(rmg_filebase=rmg_filebase, rmp_filebase=rmp_filebase,
                                file=rmfile)

    make_master_bcc(outfile=outfile, shapefile=mcalfile, goldfile=goldfile, bpzfile=bpzfile, rmfile=h5rmfile,
                    maskfile=maskfile)

    assign_jk_regions(outfile, regionfile)

from mpi4py import MPI
import numpy as np
import fitsio as fio
import healpy as hp
import h5py
import glob
import yaml
import os
import sys

'''
'''


class buzzard_flat_cat(object):

    def __init__(
            self,
            rootdir='/project/projectdirs/des/jderose/addgals/catalogs/Buzzard/Catalog_v1.1/',
            obsdir='des_obs_rotated/Y1A1/',
            pzdir='des_obs_rotated/Y1A1/bpz/',
            truthdir='truth_rotated/',
            obsname='Y1A1.',
            truthname='truth.',
            pzname='Y1A1_bpz.',
            simname='Buzzard_v1.1',
            debug=True,
            nzcut=True,
            simnum=0,
            loop=False,
            merge=False,
            already_merged=False,
            merge_with_bpz=False):

        self.maxrows = 100000000

        self.rootdir = rootdir
        self.truthdir = truthdir
        self.truthname = truthname
        self.obsdir = obsdir
        self.obsname = obsname
        self.pzdir = pzdir
        self.pzname = pzname
        self.simnum = str(simnum)
        self.simname = simname
        self.nzcut = nzcut
        self.already_merged = already_merged
        self.odir = self.obsdir

        if loop:
            self.loop_cats(debug=debug)

        if merge:
            self.merge_rank_files(merge_with_bpz=merge_with_bpz)

    def loop_cats(self, debug=False):

        if debug:
            gold = np.zeros(self.maxrows, dtype=[('coadd_objects_id', 'i8')]
                            + [('ra', 'f4')]
                            + [('dec', 'f4')]
                            + [('redshift', 'f4')]
                            + [('mag_g', 'f4')]
                            + [('mag_r', 'f4')]
                            + [('mag_i', 'f4')]
                            + [('mag_z', 'f4')]
                            + [('ivar_g', 'f4')]
                            + [('ivar_r', 'f4')]
                            + [('ivar_i', 'f4')]
                            + [('ivar_z', 'f4')]
                            + [('flags_badregion', 'i8')]
                            + [('flags_gold', 'i8')]
                            + [('hpix', 'i8')]
                            + [('lss-sample', 'i8')]
                            + [('wl-sample', 'i8')])

        else:
            gold = np.zeros(self.maxrows, dtype=[('coadd_objects_id', 'i8')]
                            + [('ra', 'f4')]
                            + [('dec', 'f4')]
                            + [('mag_r', 'f4')]
                            + [('redshift', 'f4')]
                            + [('flags_badregion', 'i8')]
                            + [('flags_gold', 'i8')]
                            + [('hpix', 'i8')]
                            + [('lss-sample', 'i8')]
                            + [('wl-sample', 'i8')])

        if debug:
            shape = np.zeros(self.maxrows, dtype=[('coadd_objects_id', 'i8')]
                             + [('ra', 'f4')]
                             + [('dec', 'f4')]
                             + [('e1', 'f4')]
                             + [('e2', 'f4')]
                             + [('g1', 'f4')]
                             + [('g2', 'f4')]
                             + [('kappa', 'f4')]
                             + [('m1', 'f4')]
                             + [('m2', 'f4')]
                             + [('c1', 'f4')]
                             + [('c2', 'f4')]
                             + [('weight', 'f4')]
                             + [('size', 'f4')]
                             + [('flags', 'i8')])
        else:
            shape = np.zeros(self.maxrows, dtype=[('coadd_objects_id', 'i8')]
                             + [('ra', 'f4')]
                             + [('dec', 'f4')]
                             + [('e1', 'f4')]
                             + [('e2', 'f4')]
                             + [('g1', 'f4')]
                             + [('g2', 'f4')]
                             + [('kappa', 'f4')]
                             + [('m1', 'f4')]
                             + [('m2', 'f4')]
                             + [('c1', 'f4')]
                             + [('c2', 'f4')]
                             + [('weight', 'f4')]
                             + [('flags', 'i8')])

        photoz = np.zeros(self.maxrows, dtype=[('coadd_objects_id', 'i8')]
                          + [('mean-z', 'f8')]
                          + [('mc-z', 'f8')]
                          + [('mode-z', 'f8')]
                          + [('redshift', 'f8')]
                          + [('weight', 'f8')]
                          + [('flags', 'f8')])

        lenst = 0

        for ifile, filename in enumerate(glob.glob(self.rootdir + '/' + self.obsdir + '*' + self.obsname + '*[0-9].fits')):

            gout = fio.FITS(
                self.simname + '_{}'.format(self.obsdir[:-1]) + '_gold.fits', 'rw')
            sout = fio.FITS(
                self.simname + '_{}'.format(self.obsdir[:-1]) + '_shape.fits', 'rw')
            pout = fio.FITS(
                self.simname + '_{}'.format(self.obsdir[:-1]) + '_pz.fits', 'rw')

            tname = filename.replace(self.obsname, self.truthname).replace(
                self.obsdir, self.truthdir)
            pzname = filename.replace(self.obsname, self.pzname).replace(
                self.obsdir, self.pzdir).replace('.fits', '.BPZ.fits')

            truth = fio.FITS(
                tname)[-1].read(columns=['ID', 'GAMMA1', 'GAMMA2', 'KAPPA', 'Z'])
            if not debug:
                obs = fio.FITS(filename)[-1].read(columns=['RA', 'DEC',
                                                           'EPSILON1', 'EPSILON2', 'LSS_FLAG', 'WL_FLAG', 'MAG_R'])
            else:
                obs = fio.FITS(filename)[-1].read(columns=['RA', 'DEC', 'EPSILON1', 'EPSILON2', 'LSS_FLAG',
                                                           'WL_FLAG', 'MAG_G', 'MAG_R', 'MAG_I', 'MAG_Z', 'IVAR_G', 'IVAR_R', 'IVAR_I', 'IVAR_Z', 'SIZE'])

            pz = fio.FITS(
                pzname)[-1].read(columns=['MEAN_Z', 'Z_MC', 'MODE_Z'])

            sflag = np.zeros(len(obs), dtype=int)
            sflag[obs['LSS_FLAG'] == 1] = 1
            sflag[obs['WL_FLAG'] == 1] = 2

            if self.nzcut:
                sflag[obs['MAG_R'] > (
                    20.8755386 + 2.88949793 * truth['Z'])] = 0

            truth = truth[sflag > 0]
            obs = obs[sflag > 0]
            pz = pz[sflag > 0]
            sflag = sflag[sflag > 0]

            # insert selection function here to mask truth/obs (if can be run on individual files)

            print(ifile, len(truth), filename)

            gold['coadd_objects_id'][lenst:lenst + len(truth)] = truth['ID']
            gold['ra'][lenst:lenst + len(truth)] = obs['RA']
            gold['dec'][lenst:lenst + len(truth)] = obs['DEC']
            gold['redshift'][lenst:lenst + len(truth)] = truth['Z']
            gold['hpix'][lenst:lenst + len(truth)] = hp.ang2pix(
                16384, np.pi / 2. - np.radians(obs['DEC']), np.radians(obs['RA']), nest=True)
            gold['lss-sample'][lenst:lenst + len(truth)] = obs['LSS_FLAG']
            gold['wl-sample'][lenst:lenst + len(truth)] = obs['WL_FLAG']
            gold['mag_r'][lenst:lenst + len(truth)] = obs['MAG_R']

            if debug:
                gold['mag_g'][lenst:lenst + len(truth)] = obs['MAG_G']
                gold['mag_i'][lenst:lenst + len(truth)] = obs['MAG_I']
                gold['mag_z'][lenst:lenst + len(truth)] = obs['MAG_Z']
                gold['ivar_g'][lenst:lenst + len(truth)] = obs['IVAR_G']
                gold['ivar_r'][lenst:lenst + len(truth)] = obs['IVAR_R']
                gold['ivar_i'][lenst:lenst + len(truth)] = obs['IVAR_I']
                gold['ivar_z'][lenst:lenst + len(truth)] = obs['IVAR_Z']

            shape['ra'][lenst:lenst + len(truth)] = truth['RA']
            shape['dec'][lenst:lenst + len(truth)] = truth['DEC']
            shape['coadd_objects_id'][lenst:lenst + len(truth)] = truth['ID']
            shape['e1'][lenst:lenst + len(truth)] = obs['EPSILON1']
            shape['e2'][lenst:lenst + len(truth)] = obs['EPSILON2']
            shape['g1'][lenst:lenst + len(truth)] = truth['GAMMA1']
            shape['g2'][lenst:lenst + len(truth)] = truth['GAMMA2']
            shape['kappa'][lenst:lenst + len(truth)] = truth['KAPPA']
            shape['m1'][lenst:lenst + len(truth)] += 1.
            shape['m2'][lenst:lenst + len(truth)] += 1.
            shape['weight'][lenst:lenst + len(truth)] += 1.
            if debug:
                shape['size'][lenst:lenst + len(truth)] = obs['SIZE']

            photoz['coadd_objects_id'][lenst:lenst + len(truth)] = truth['ID']
            photoz['mean-z'][lenst:lenst + len(truth)] = pz['MEAN_Z']
            photoz['mc-z'][lenst:lenst + len(truth)] = pz['Z_MC']
            photoz['mode-z'][lenst:lenst + len(truth)] = pz['MODE_Z']
            photoz['redshift'][lenst:lenst + len(truth)] = truth['Z']
            photoz['weight'][lenst:lenst + len(truth)] += 1.

            if ifile == 0:
                gout.write(gold[lenst:lenst + len(truth)])
                sout.write(shape[lenst:lenst + len(truth)])
                pout.write(photoz[lenst:lenst + len(truth)])
            else:
                gout[-1].append(gold[lenst:lenst + len(truth)])
                sout[-1].append(shape[lenst:lenst + len(truth)])
                pout[-1].append(photoz[lenst:lenst + len(truth)])

            sout.close()
            pout.close()
            gout.close()

            lenst += len(truth)

        return

    def process_single_file(self, obs, truth, pz, sample, rank, debug=False):

        if debug:
            gold = np.zeros(len(obs), dtype=[('coadd_objects_id', 'i8')]
                            + [('ra', 'f4')]
                            + [('dec', 'f4')]
                            + [('redshift', 'f4')]
                            + [('mag_g', 'f4')]
                            + [('mag_r', 'f4')]
                            + [('mag_i', 'f4')]
                            + [('mag_z', 'f4')]
                            + [('magerr_g', 'f4')]
                            + [('magerr_r', 'f4')]
                            + [('magerr_i', 'f4')]
                            + [('magerr_z', 'f4')]
                            + [('ivar_g', 'f4')]
                            + [('ivar_r', 'f4')]
                            + [('ivar_i', 'f4')]
                            + [('ivar_z', 'f4')]
                            + [('flags_badregion', 'i8')]
                            + [('flags_gold', 'i8')]
                            + [('hpix', 'i8')]
                            + [('lss-sample', 'i8')]
                            + [('wl-sample', 'i8')])

        else:
            gold = np.zeros(len(obs), dtype=[('coadd_objects_id', 'i8')]
                            + [('ra', 'f4')]
                            + [('dec', 'f4')]
                            + [('mag_r', 'f4')]
                            + [('redshift', 'f4')]
                            + [('flags_badregion', 'i8')]
                            + [('flags_gold', 'i8')]
                            + [('hpix', 'i8')]
                            + [('lss-sample', 'i8')]
                            + [('wl-sample', 'i8')])

        if debug:
            shape = np.zeros(len(obs), dtype=[('coadd_objects_id', 'i8')]
                             + [('ra', 'f4')]
                             + [('dec', 'f4')]
                             + [('e1', 'f4')]
                             + [('e2', 'f4')]
                             + [('g1', 'f4')]
                             + [('g2', 'f4')]
                             + [('kappa', 'f4')]
                             + [('m1', 'f4')]
                             + [('m2', 'f4')]
                             + [('c1', 'f4')]
                             + [('c2', 'f4')]
                             + [('weight', 'f4')]
                             + [('size', 'f4')]
                             + [('flags', 'i8')])
        else:
            shape = np.zeros(len(obs), dtype=[('coadd_objects_id', 'i8')]
                             + [('ra', 'f4')]
                             + [('dec', 'f4')]
                             + [('e1', 'f4')]
                             + [('e2', 'f4')]
                             + [('g1', 'f4')]
                             + [('g2', 'f4')]
                             + [('kappa', 'f4')]
                             + [('m1', 'f4')]
                             + [('m2', 'f4')]
                             + [('c1', 'f4')]
                             + [('c2', 'f4')]
                             + [('weight', 'f4')]
                             + [('flags', 'i8')])

#        altlens = np.zeros(dtype=[('coadd_objects_id', 'i8')]
#                           + [('ra', 'f4')]
#                           + [('dec', 'f4')]
#                           + [('mag_i', 'f4')]
#                           + [('magerr_i', 'f4')]
#                           + [('flux_i', 'f4')]
#                           + [('ivar_i', 'f4')]
#                           + [('weight', 'f4')]
#                           + [('flags', 'i8')])

        photoz = np.zeros(len(obs), dtype=[('coadd_objects_id', 'i8')]
                          + [('mean-z', 'f8')]
                          + [('mc-z', 'f8')]
                          + [('mode-z', 'f8')]
                          + [('redshift', 'f8')]
                          + [('weight', 'f8')]
                          + [('z_cos'), 'f4']
                          + [('flags', 'f8')])

        lenst = 0

        if os.path.exists(self.odir + '/' + self.simname + '_{}'.format(self.obsdir[:-1]) + '_gold.{}.fits'.format(rank)):
            ifile = 1
        elif not os.path.exists(self.odir):
            try:
                os.makedirs(self.odir)
            except:
                pass
            ifile = 0
        else:
            ifile = 0

        gout = fio.FITS(self.odir + '/' + self.simname +
                        '_{}'.format(self.obsdir[:-1]) + '_gold.{}.fits'.format(rank), 'rw')
        sout = fio.FITS(self.odir + '/' + self.simname +
                        '_{}'.format(self.obsdir[:-1]) + '_shape.{}.fits'.format(rank), 'rw')
        pout = fio.FITS(self.odir + '/' + self.simname +
                        '_{}'.format(self.obsdir[:-1]) + '_pz.{}.fits'.format(rank), 'rw')
#        alout = fio.FITS(self.odir + '/' + self.simname +
#                         '_{}'.format(self.obsdir[:-1]) + '_alt_lens.{}.fits'.format(rank), 'rw')

        sflag = np.zeros(len(obs), dtype=int)
        sflag[sample['LSS_FLAG'] == 1] = 1
        sflag[sample['WL_FLAG'] == 1] = 2

#        if self.nzcut:
#            sflag[obs['MAG_R'] > (20.8755386 + 2.88949793 * truth['Z'])] = 0

        truth = truth[sflag > 0]
        obs = obs[sflag > 0]
        if pz is not None:
            pz = pz[sflag > 0]

        sample = sample[sflag > 0]

        if self.nzcut:
            gold = gold[sflag > 0]
            shape = shape[sflag > 0]
            photoz = photoz[sflag > 0]

        # insert selection function here to mask truth/obs (if can be run on individual files)

        gold['coadd_objects_id'] = truth['ID']
        gold['ra'] = obs['RA']
        gold['dec'] = obs['DEC']
        gold['redshift'] = truth['Z']

        gold['hpix'] = hp.ang2pix(
            16384, np.pi / 2. - np.radians(obs['DEC']), np.radians(obs['RA']), nest=True)
        gold['lss-sample'] = sample['LSS_FLAG']
        gold['wl-sample'] = sample['WL_FLAG']
        gold['mag_r'] = obs['MAG_R']

        if debug:
            gold['mag_g'] = obs['MAG_G']
            gold['mag_i'] = obs['MAG_I']
            gold['mag_z'] = obs['MAG_Z']
            gold['ivar_g'] = obs['IVAR_G']
            gold['ivar_r'] = obs['IVAR_R']
            gold['ivar_i'] = obs['IVAR_I']
            gold['ivar_z'] = obs['IVAR_Z']

        shape['coadd_objects_id'] = truth['ID']
        shape['ra'] = obs['RA']
        shape['dec'] = obs['DEC']
        shape['e1'] = obs['EPSILON1']
        shape['e2'] = obs['EPSILON2']
        shape['g1'] = truth['GAMMA1']
        shape['g2'] = truth['GAMMA2']
        shape['kappa'] = truth['KAPPA']
        shape['m1'] += 1.
        shape['m2'] += 1.
        shape['weight'] += 1.
        if debug:
            shape['size'] = obs['SIZE']

#        altlens['coadd_objects_id'] = truth['ID']
#        altlens['ra'] = obs['RA']
#        altlens['dec'] = obs['DEC']
#        altlens['mag_i'] = obs['MAG_I']
#        altlens['magerr_i'] = obs['MAGERR_I']
#        altlens['flux_i'] = obs['FLUX_I']
#        altlens['ivar_i'] = obs['IVAR_I']
#        altlens['weight'] = 1
#        altlens['flags'] = 0

        photoz['coadd_objects_id'] = truth['ID']
        if pz is not None:
            photoz['mean-z'] = pz['MEAN_Z']
            photoz['mc-z'] = pz['Z_MC']
            photoz['mode-z'] = pz['MODE_Z']

        photoz['redshift'] = truth['Z']
        photoz['z_cos'] = truth['Z_COS']
        photoz['weight'] += 1.

        if ifile == 0:
            gout.write(gold)
            sout.write(shape)
            pout.write(photoz)
#            alout.write(altlens)
        else:
            gout[-1].append(gold)
            sout[-1].append(shape)
            pout[-1].append(photoz)
#            alout[-1].append(altlens)

        sout.close()
        pout.close()
        gout.close()
#        alout.close()

        return

    def process_single_file_all_detections(self, obs, truth, pz, rank, debug=False):

        gold = np.zeros(len(obs), dtype=[('coadd_objects_id', 'i8')]
                        + [('ra', 'f4')]
                        + [('dec', 'f4')]
                        + [('tra', 'f4')]
                        + [('tdec', 'f4')]
                        + [('px', 'f4')]
                        + [('py', 'f4')]
                        + [('pz', 'f4')]
                        + [('vx', 'f4')]
                        + [('vy', 'f4')]
                        + [('vz', 'f4')]
                        + [('rhalo', 'f4')]
                        + [('r200', 'f4')]
                        + [('m200', 'f4')]
                        + [('central', 'i4')]
                        + [('haloid', 'i8')]
                        + [('mag_g', 'f4')]
                        + [('mag_r', 'f4')]
                        + [('mag_i', 'f4')]
                        + [('mag_z', 'f4')]
                        + [('mag_g_lensed', 'f4')]
                        + [('mag_r_lensed', 'f4')]
                        + [('mag_i_lensed', 'f4')]
                        + [('mag_z_lensed', 'f4')]
                        + [('mag_g_true', 'f4')]
                        + [('mag_r_true', 'f4')]
                        + [('mag_i_true', 'f4')]
                        + [('mag_z_true', 'f4')]
                        + [('magerr_g', 'f4')]
                        + [('magerr_r', 'f4')]
                        + [('magerr_i', 'f4')]
                        + [('magerr_z', 'f4')]
                        + [('flux_g', 'f4')]
                        + [('flux_r', 'f4')]
                        + [('flux_i', 'f4')]
                        + [('flux_z', 'f4')]
                        + [('ivar_g', 'f4')]
                        + [('ivar_r', 'f4')]
                        + [('ivar_i', 'f4')]
                        + [('ivar_z', 'f4')]
                        + [('sdss_sedid', 'i8')]
                        + [('flags_badregion', 'i8')]
                        + [('flags_gold', 'i8')]
                        + [('hpix', 'i8')])

        shape = np.zeros(len(obs), dtype=[('coadd_objects_id', 'i8')]
                         + [('e1', 'f4')]
                         + [('e2', 'f4')]
                         + [('g1', 'f4')]
                         + [('g2', 'f4')]
                         + [('kappa', 'f4')]
                         + [('m1', 'f4')]
                         + [('m2', 'f4')]
                         + [('c1', 'f4')]
                         + [('c2', 'f4')]
                         + [('weight', 'f4')]
                         + [('size', 'f4')]
                         + [('flags', 'i8')])

        photoz = np.zeros(len(obs), dtype=[('coadd_objects_id', 'i8')]
                          + [('mean-z', 'f8')]
                          + [('mc-z', 'f8')]
                          + [('redshift', 'f8')]
                          + [('redshift_cos', 'f8')]
                          + [('weight', 'f8')]
                          + [('flags', 'f8')])

        lenst = 0

        if os.path.exists(self.odir + '/' + self.simname + '_{}'.format(self.obsdir[:-1]) + '_gold.{}.fits'.format(rank)):
            ifile = 1
        elif not os.path.exists(self.odir):
            try:
                os.makedirs(self.odir)
            except:
                pass
            ifile = 0
        else:
            ifile = 0

        gout = fio.FITS(self.odir + '/' + self.simname +
                        '_{}'.format(self.obsdir[:-1]) + '_gold.{}.fits'.format(rank), 'rw')
        sout = fio.FITS(self.odir + '/' + self.simname +
                        '_{}'.format(self.obsdir[:-1]) + '_shape.{}.fits'.format(rank), 'rw')
        pout = fio.FITS(self.odir + '/' + self.simname +
                        '_{}'.format(self.obsdir[:-1]) + '_pz.{}.fits'.format(rank), 'rw')

        # insert selection function here to mask truth/obs (if can be run on individual files)

        gold['coadd_objects_id'] = truth['ID']
        gold['ra'] = obs['RA']
        gold['dec'] = obs['DEC']
        gold['tra'] = truth['TRA']
        gold['tdec'] = truth['TDEC']
        gold['px'] = truth['PX']
        gold['py'] = truth['PY']
        gold['pz'] = truth['PZ']
        gold['vx'] = truth['VX']
        gold['vy'] = truth['VY']
        gold['vz'] = truth['VZ']

        gold['hpix'] = hp.ang2pix(
            16384, np.pi / 2. - np.radians(obs['DEC']), np.radians(obs['RA']), nest=True)

        gold['mag_r'] = obs['MAG_R']
        gold['mag_g'] = obs['MAG_G']
        gold['mag_i'] = obs['MAG_I']
        gold['mag_z'] = obs['MAG_Z']

        gold['mag_r_lensed'] = truth['LMAG'][:, 1]
        gold['mag_g_lensed'] = truth['LMAG'][:, 0]
        gold['mag_i_lensed'] = truth['LMAG'][:, 2]
        gold['mag_z_lensed'] = truth['LMAG'][:, 3]

        gold['mag_r_true'] = truth['TMAG'][:, 1]
        gold['mag_g_true'] = truth['TMAG'][:, 0]
        gold['mag_i_true'] = truth['TMAG'][:, 2]
        gold['mag_z_true'] = truth['TMAG'][:, 3]

        gold['magerr_r'] = obs['MAGERR_R']
        gold['magerr_g'] = obs['MAGERR_G']
        gold['magerr_i'] = obs['MAGERR_I']
        gold['magerr_z'] = obs['MAGERR_Z']

        gold['flux_r'] = obs['FLUX_R']
        gold['flux_g'] = obs['FLUX_G']
        gold['flux_i'] = obs['FLUX_I']
        gold['flux_z'] = obs['FLUX_Z']

        gold['ivar_g'] = obs['IVAR_G']
        gold['ivar_r'] = obs['IVAR_R']
        gold['ivar_i'] = obs['IVAR_I']
        gold['ivar_z'] = obs['IVAR_Z']

        gold['rhalo'] = truth['RHALO']
        gold['r200'] = truth['R200']
        gold['m200'] = truth['M200']
        gold['haloid'] = truth['HALOID']
        gold['central'] = truth['CENTRAL']

        gold['sdss_sedid'] = truth['SEDID']

        shape['coadd_objects_id'] = truth['ID']
        shape['e1'] = obs['EPSILON1']
        shape['e2'] = obs['EPSILON2']
        shape['g1'] = truth['GAMMA1']
        shape['g2'] = truth['GAMMA2']
        shape['kappa'] = truth['KAPPA']
        shape['m1'] += 1.
        shape['m2'] += 1.
        shape['weight'] += 1.
        shape['size'] = obs['SIZE']

        photoz['coadd_objects_id'] = truth['ID']
        if pz is not None:
            photoz['mean-z'] = pz['MEAN_Z']
            photoz['mc-z'] = pz['Z_MC']

        photoz['redshift'] = truth['Z']
        photoz['redshift_cos'] = truth['Z_COS']
        photoz['weight'] += 1.

        if ifile == 0:
            gout.write(gold)
            sout.write(shape)
            pout.write(photoz)
        else:
            gout[-1].append(gold)
            sout[-1].append(shape)
            pout[-1].append(photoz)

        sout.close()
        pout.close()
        gout.close()

        return

    def merge_rank_files(self, merge_with_bpz=False):

        gout = fio.FITS(self.odir + '/' + self.simname +
                        '_{}'.format(self.obsdir[:-1]) + '_gold.fits', 'rw')
        sout = fio.FITS(self.odir + '/' + self.simname +
                        '_{}'.format(self.obsdir[:-1]) + '_shape.fits', 'rw')
        pout = fio.FITS(self.odir + '/' + self.simname +
                        '_{}'.format(self.obsdir[:-1]) + '_pz.fits', 'rw')

        gfiles = glob.glob(self.odir + '/' +
                           self.simname + '_{}'.format(self.obsdir[:-1]) +
                           '_gold*[0-9].fits')
        size = len(gfiles)
        count = 0
        for i in range(size):
            try:
                gold = fio.read(self.odir + '/' +
                                self.simname + '_{}'.format(self.obsdir[:-1]) + '_gold.{}.fits'.format(i))
                shape = fio.read(self.odir + '/' +
                                 self.simname + '_{}'.format(self.obsdir[:-1]) + '_shape.{}.fits'.format(i))
                photoz = fio.read(self.odir + '/' +
                                  self.simname + '_{}'.format(self.obsdir[:-1]) + '_pz.{}.fits'.format(i))
            except OSError as e:
                print('File rank {} has no galaxies in it'.format(i))
                continue

            idx = ((gold['mag_g'] < 99) & (gold['mag_r'] < 99) & (gold['mag_i'] < 99) & (gold['mag_z'] < 99) &
                   np.isfinite(gold['mag_g']) & np.isfinite(
                       gold['mag_r']) & np.isfinite(gold['mag_i'])
                   & np.isfinite(gold['mag_z']))

            if merge_with_bpz:
                bpz = fio.read(
                    self.simname + '_{}'.format(self.obsdir[:-1]) + '_gold.{}.BPZ.fits'.format(i))
                photoz['mean-z'] = bpz['MEAN_Z']
                photoz['mc-z'] = bpz['Z_MC']
                photoz['mode-z'] = bpz['MODE_Z']

            if count == 0:
                gout.write(gold[idx])
                sout.write(shape[idx])
                pout.write(photoz[idx])
            else:
                gout[-1].append(gold[idx])
                sout[-1].append(shape[idx])
                pout[-1].append(photoz[idx])
            count += 1

    def merge_rank_files_h5(self, merge_with_bpz=False):

        mcal_inc = {'coadd_objects_id': 'coadd_object_id',
                    'flags': 'flags',
                    'weight': 'mask_frac',
                    'ra': 'ra',
                    'dec': 'dec',
                    'e1': 'e1',
                    'e2': 'e2',
                    'g1': 'g1',
                    'g2': 'g2',
                    'kappa': 'kappa',
                    'size': 'size',
                    }

        gold_inc = {'coadd_objects_id': 'coadd_object_id',
                    'hpix': 'hpix_16384',
                    'flags_gold': 'flags_gold',
                    'ra': 'ra',
                    'dec': 'dec',
                    'px': 'px',
                    'py': 'py',
                    'pz': 'pz',
                    'vx': 'vx',
                    'vy': 'vy',
                    'vz': 'vz',
                    'mag_g': 'mag_g',
                    'magerr_g': 'mag_err_g',
                    'mag_r': 'mag_r',
                    'magerr_r': 'mag_err_r',
                    'mag_i': 'mag_i',
                    'magerr_i': 'mag_err_i',
                    'mag_z': 'mag_z',
                    'magerr_z': 'mag_err_z',
                    'flux_g': 'flux_g',
                    'flux_r': 'flux_r',
                    'flux_i': 'flux_i',
                    'flux_z': 'flux_z',
                    'mag_g_true': 'mag_g_true',
                    'mag_r_true': 'mag_r_true',
                    'mag_i_true': 'mag_i_true',
                    'mag_z_true': 'mag_z_true',
                    'mag_g_lensed': 'mag_g_lensed',
                    'mag_r_lensed': 'mag_r_lensed',
                    'mag_i_lensed': 'mag_i_lensed',
                    'mag_z_lensed': 'mag_z_lensed',
                    'ivar_g': 'ivar_g',
                    'ivar_r': 'ivar_r',
                    'ivar_i': 'ivar_i',
                    'ivar_z': 'ivar_z',
                    'sdss_sedid': 'sdss_sedid',
                    'rhalo': 'rhalo',
                    'r200': 'r200',
                    'm200': 'm200',
                    'haloid': 'haloid'}

        bpz_inc = {'coadd_objects_id': 'coadd_object_id',
                   'mc-z': 'zmc_sof',
                   'mean-z': 'zmean_sof',
                   'redshift': 'z',
                   'redshift_cos': 'redshift_cos'}

        gout = h5py.File(self.odir + '/' + self.simname +
                         '_{}'.format(self.obsdir[:-1]) + '_gold.h5', 'w')
        sout = h5py.File(self.odir + '/' + self.simname +
                         '_{}'.format(self.obsdir[:-1]) + '_shape.h5', 'w')
        pout = h5py.File(self.odir + '/' + self.simname +
                         '_{}'.format(self.obsdir[:-1]) + '_bpz.h5', 'w')

        if not self.already_merged:
            gfiles = glob.glob(self.odir + '/' +
                               self.simname + '_{}'.format(self.obsdir[:-1]) +
                               '_gold*[0-9].fits')
        else:
            gfiles = [(self.odir + '/' + self.simname + '_{}'.format(self.obsdir[:-1]) +
                       '_gold.fits')]

        size = len(gfiles)
        total_length = 0
        iter_end = 0

        for i in range(size):
            try:
                if not self.already_merged:
                    hdr = fio.read_header(self.odir + '/' +
                                          self.simname + '_{}'.format(self.obsdir[:-1]) +
                                          '_gold.{}.fits'.format(i), 1)
                else:
                    hdr = fio.read_header(self.odir + '/' +
                                          self.simname + '_{}'.format(self.obsdir[:-1]) +
                                          '_gold.fits', 1)

                total_length += hdr['NAXIS2']

            except OSError as e:
                continue

        for i in range(size):
            try:
                if not self.already_merged:
                    gold = fio.read(self.odir + '/' +
                                    self.simname + '_{}'.format(self.obsdir[:-1]) +
                                    '_gold.{}.fits'.format(i))
                    shape = fio.read(self.odir + '/' +
                                     self.simname + '_{}'.format(self.obsdir[:-1]) +
                                     '_shape.{}.fits'.format(i))
                    bpz = fio.read(self.odir + '/' +
                                   self.simname + '_{}'.format(self.obsdir[:-1]) +
                                   '_pz.{}.fits'.format(i))
                else:
                    gold = fio.read(self.odir + '/' +
                                    self.simname + '_{}'.format(self.obsdir[:-1]) +
                                    '_gold.fits')
                    shape = fio.read(self.odir + '/' +
                                     self.simname + '_{}'.format(self.obsdir[:-1]) +
                                     '_shape.fits')
                    bpz = fio.read(self.odir + '/' +
                                   self.simname + '_{}'.format(self.obsdir[:-1]) +
                                   '_pz.fits')

            except OSError as e:
                print('File rank {} has no galaxies in it'.format(i))
                continue

            lencat = len(gold)

            for name in gold_inc:
                if i == 0:
                    gout.create_dataset('catalog/gold/' + gold_inc[name], maxshape=(total_length,),
                                        shape=(total_length,
                                               ), dtype=gold.dtype[name],
                                        chunks=(1000000,))
                gout['catalog/gold/' + gold_inc[name]
                     ][iter_end:iter_end + lencat] = gold[name]

            for name in mcal_inc:
                # old fits files don't have ra/dec in shape files
                try:
                    if i == 0:
                        sout.create_dataset('catalog/unsheared/metacal/' + mcal_inc[name], maxshape=(total_length,),
                                            shape=(total_length,
                                                   ), dtype=shape.dtype[name],
                                            chunks=(1000000,))
                    sout['catalog/unsheared/metacal/' + mcal_inc[name]
                         ][iter_end:iter_end + lencat] = shape[name]
                except KeyError as e:
                    if i == 0:
                        sout.create_dataset('catalog/unsheared/metacal/' + mcal_inc[name], maxshape=(total_length,),
                                            shape=(total_length,
                                                   ), dtype=gold.dtype[name],
                                            chunks=(1000000,))
                    sout['catalog/unsheared/metacal/' + mcal_inc[name]
                         ][iter_end:iter_end + lencat] = gold[name]

            for name in bpz_inc:
                try:
                    if i == 0:
                        pout.create_dataset('catalog/bpz/' + bpz_inc[name], maxshape=(total_length,),
                                            shape=(total_length,
                                                   ), dtype=bpz.dtype[name],
                                            chunks=(1000000,))
                    pout['catalog/bpz/' + bpz_inc[name]
                         ][iter_end:iter_end + lencat] = bpz[name]
                except KeyError as e:
                    if name == 'z_cos':
                        if i == 0:
                            pout.create_dataset('catalog/bpz/' + bpz_inc[name], maxshape=(total_length,),
                                                shape=(
                                                    total_length,), dtype=bpz.dtype['redshift'],
                                                chunks=(1000000,))
                        pout['catalog/bpz/' + bpz_inc[name]
                             ][iter_end:iter_end + lencat] = bpz['redshift']
                    else:
                        raise(e)

            iter_end += lencat

        gout.close()
        pout.close()
        sout.close()


if __name__ == '__main__':

    cfgfile = sys.argv[1]

    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp)

    if 'merge' in cfg:
        cfg = cfg['merge']

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        cat = buzzard_flat_cat(**cfg)
        cat.merge_rank_files_h5()

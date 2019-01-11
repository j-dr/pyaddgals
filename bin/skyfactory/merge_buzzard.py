from mpi4py import MPI
import numpy as np
import fitsio as fio
import healpy as hp
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
            odir='',
            debug=True,
            nzcut=True,
            simnum=0,
            loop=False,
            merge=False,
            merge_with_bpz=False):

        self.maxrows = 100000000

        self.rootdir = rootdir
        self.odir = odir
        self.truthdir = truthdir
        self.truthname = truthname
        self.obsdir = obsdir
        self.obsname = obsname
        self.pzdir = pzdir
        self.pzname = pzname
        self.simnum = str(simnum)
        self.simname = simname
        self.nzcut = nzcut

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
                4096, np.pi / 2. - np.radians(obs['DEC']), np.radians(obs['RA']), nest=True)
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

        photoz = np.zeros(len(obs), dtype=[('coadd_objects_id', 'i8')]
                          + [('mean-z', 'f8')]
                          + [('mc-z', 'f8')]
                          + [('mode-z', 'f8')]
                          + [('redshift', 'f8')]
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

        sflag = np.zeros(len(obs), dtype=int)
        sflag[sample['LSS_FLAG'] == 1] = 1
        sflag[sample['WL_FLAG'] == 1] = 2

        if self.nzcut:
            sflag[obs['MAG_R'] > (20.8755386 + 2.88949793 * truth['Z'])] = 0

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
            4096, np.pi / 2. - np.radians(obs['DEC']), np.radians(obs['RA']), nest=True)
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

        photoz['coadd_objects_id'] = truth['ID']
        if pz is not None:
            photoz['mean-z'] = pz['MEAN_Z']
            photoz['mc-z'] = pz['Z_MC']
            photoz['mode-z'] = pz['MODE_Z']

        photoz['redshift'] = truth['Z']
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

            if i == 0:
                gout.write(gold[idx])
                sout.write(shape[idx])
                pout.write(photoz[idx])
            else:
                gout[-1].append(gold[idx])
                sout[-1].append(shape[idx])
                pout[-1].append(photoz[idx])


if __name__ == '__main__':

    cfgfile = sys.argv[1]

    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp)

    if 'merge' in cfg:
        cfg = cfg['merge']

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        obj = buzzard_flat_cat(**cfg)
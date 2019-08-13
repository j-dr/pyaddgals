from __future__ import print_function, division
from mpi4py import MPI
from glob import glob
from copy import copy
from merge_buzzard import buzzard_flat_cat
import numpy as np
import healpix_util as hu
import fitsio
import sys
import yaml

MASKED_VAL = -9999


def read_partial_map(filename, ext=1, masked_val=MASKED_VAL,
                     pix_col='PIXEL', val_col='SIGNAL'):

    f = fitsio.FITS(filename)[ext]

    nside = int(f.read_header()['NSIDE'])
    hpix = hu.HealPix("ring", nside)
    m = masked_val * np.ones(hpix.npix)
    m_data = f.read()
    m[m_data[pix_col]] = m_data[val_col]

    return hu.Map("ring", m)


def sys_map_cuts(gal_data, sys_map_data=None, ra_col='ra',
                 dec_col='dec'):
    """Get systematic values, and cut data without sys data"""

    sys_map_vals = {}
    use = np.ones(len(gal_data), dtype='bool')

    for i, (name, m) in enumerate(sys_map_data.items()):

        sys_map_vals[name] = m.get_mapval(gal_data[ra_col],
                                          gal_data[dec_col])

        if i == 0:
            use = sys_map_vals[name] != MASKED_VAL
        else:
            use &= sys_map_vals[name] != MASKED_VAL

    return use, sys_map_vals


def gold_cuts(gal_data, ra_col='RA', dec_col='DEC',
              gold_fp_map=None, gold_br_map=None):

    sys.stdout.flush()

    sys.stdout.flush()

    use = gold_fp_map.get_mapval(gal_data[ra_col], gal_data[dec_col]) >= 1

    if gold_br_map is not None:
        use *= (gold_br_map.get_mapval(gal_data[ra_col],
                                       gal_data[dec_col]) == 0)
    sys.stdout.flush()
    return use.astype(bool)


def WL_cuts(obs, truth, pz, sys_map_vals,
            maglim_cut_factor, rgrp_cut, z_col, nz_cut):

    psf_size = 0.26 * 0.5 * sys_map_vals['psf_fwhm_r']
    mask = ((obs['MAGERR_R'] < maglim_cut_factor) & (obs['MAGERR_I'] < maglim_cut_factor) &
            (obs['MAGERR_Z'] < maglim_cut_factor))
    mask &= np.sqrt(obs['SIZE']**2 + psf_size**2) > rgrp_cut * psf_size

    del psf_size

    mask &= (np.isfinite(sys_map_vals['maglim_r'])
             * np.isfinite(sys_map_vals['psf_fwhm_r']))

    mask &= obs['MAG_I'] < (nz_cut[0] + nz_cut[1] * truth['Z'])

    return mask


def LSS_cuts(obs, truth, pz, sys_map_vals, zcol,
             nz_cut):

    if 'MEAN_Z' == zcol:
        z = pz[zcol]
    else:
        z = truth[zcol]

    mask = (obs['MAG_I'] > 17.5) & (obs['MAG_I'] < (nz_cut[0] + nz_cut[1] * z))

    return mask


def make_single_selection(obs, truth, pz, mask,
                          sys_map_data, cut_func, zcol):

    # mask based on systematics maps
    print('{}'.format(mask.any()))
    smask, sys_map_vals = sys_map_cuts(obs,
                                       sys_map_data=sys_map_data,
                                       ra_col='RA', dec_col='DEC')

    print('{}'.format(smask.any()))
    mask &= smask

    # apply galaxy property cuts
    mask &= cut_func(obs, truth, pz, sys_map_vals, zcol)
    print('{}'.format(mask.any()))

    return mask


def pair_files(ofiles, tfiles, pzfiles):

    opix = np.array([int(i.split('.')[-2]) for i in ofiles])
    tpix = np.array([int(i.split('.')[-2]) for i in tfiles])

    ssidx = np.in1d(tpix, opix)

    tfiles = tfiles[ssidx]
    tpix = tpix[ssidx]

    assert(len(tpix) == len(opix))

    oidx = opix.argsort()
    tidx = tpix.argsort()

    assert((tpix[tidx] == opix[oidx]).all())

    ofiles = ofiles[oidx]
    tfiles = tfiles[tidx]

    if pzfiles is not None:
        ppix = np.array([int(i.split('.')[-3]) for i in pzfiles])
        assert(len(tpix) == len(ppix))
        pidx = ppix.argsort()
        assert((tpix[tidx] == ppix[pidx]).all())
        pzfiles = pzfiles[pidx]
    else:
        pzfiles = [None] * len(ofiles)

    return ofiles, tfiles, pzfiles


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cfgfile = sys.argv[1]
    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp)
    print(cfg)
    sys.stdout.flush()

    use_hdf5 = cfg.pop('hdf5', False)

    # Read in gold masks
    if 'gold_footprint_fn' in list(cfg['gold'].keys()):
        gold_fp = hu.readMap(cfg['gold']['gold_footprint_fn'])
    else:
        gold_fp = None
    if 'gold_badreg_fn' in list(cfg['gold'].keys()):
        gold_br = hu.readMap(cfg['gold']['gold_badreg_fn'])
    else:
        gold_br = None

    pzpath = cfg['sim'].pop('pzpath', None)

    ofiles = np.array(glob(cfg['sim']['obspath']))
    tfiles = np.array(glob(cfg['sim']['truthpath']))
    if pzpath is not None:
        pzfiles = np.array(glob(pzpath))
    else:
        pzfiles = None

    print('pairing files')
    sys.stdout.flush()
    ofiles, tfiles, pzfiles = pair_files(ofiles, tfiles, pzfiles)

    sys_map_data = {}
    dtype_flag = np.dtype([('{0}_FLAG'.format(sample), 'i8')
                           for sample in cfg['samples']])

    flatcat = buzzard_flat_cat(simname=cfg['merge']['simname'], obsdir=cfg['merge']['obsdir'],
                               obsname=cfg['merge']['obsname'], nzcut=cfg['merge']['nzcut'])

    for of, tf, pz in zip(ofiles[rank::size], tfiles[rank::size], pzfiles[rank::size]):

        print("working on files {}, {}, {}".format(of, tf, pz))
        sys.stdout.flush()
        obsf = fitsio.FITS(of, 'r')

        obs = obsf[-1].read()
        idx = ((obs['MAG_G'] < 99) | (obs['MAG_R'] < 99) |
               (obs['MAG_I'] < 99) | (obs['MAG_Z'] < 99))

        obs = obs[idx]

        truthf = fitsio.FITS(tf, 'r')
        if pz is not None:
            pf = fitsio.FITS(pz, 'r')
            pz = pf[-1].read(columns=['MEAN_Z', 'Z_MC', 'MODE_Z'])
            pz = pz[idx]

        truth = truthf[-1].read(columns=['ID', 'GAMMA1',
                                         'GAMMA2', 'KAPPA', 'Z',
                                         'Z_COS', 'PX', 'PY', 'PZ',
                                         'VX', 'VY', 'VZ', 'TRA', 'TDEC',
                                         'SEDID', 'LMAG', 'TMAG',
                                         'M200', 'R200', 'RHALO', 'HALOID',
                                         'CENTRAL'])
        truth = truth[idx]

        sys.stdout.flush()
        mask = gold_cuts(obs, gold_fp_map=gold_fp,
                         gold_br_map=gold_br)

        obs = obs[mask]
        truth = truth[mask]
        if pz is not None:
            pz = pz[mask]

        flatcat.process_single_file_all_detections(
            obs, truth, pz, rank, debug=cfg['merge']['debug'])

        obsf.close()
        truthf.close()
        if pz is not None:
            pf.close()

    comm.Barrier()

    if rank == 0:
        if use_hdf5:
            flatcat.merge_rank_files_h5()
        else:
            flatcat.merge_rank_files(
                merge_with_bpz=cfg['merge']['merge_with_bpz'])

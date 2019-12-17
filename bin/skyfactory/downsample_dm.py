from __future__ import print_function, division
from mpi4py import MPI
from glob import glob
import numpy as np
import healpy as hp

import fitsio
import pickle
import h5py
import yaml
import sys

from PyAddgals.config import parseConfig
from PyAddgals.cosmology import Cosmology
from pixlc.pixLC import read_radial_bin


def select_matter(r, rmin, rmax, nside, nd):
    farea = hp.nside2pixarea(nside) / (12 * hp.nside2pixarea(1))

    zidx = np.ones(len(r), dtype=np.bool)

    rbins = np.arange(rmin, rmax + 5, 5)
    vol = [farea * 4 * np.pi * (rbins[i + 1]**3 - rbins[i]**3) / 3 for i in
           range(len(rbins) - 1)]

    idx = np.digitize(r, bins=rbins)
    counts = np.zeros(len(rbins) - 1)

    for i in np.arange(1, len(rbins)):
        zbidx = (idx == i) & zidx
        znbidx = idx != i
        n = np.sum(zbidx)
        count = int(nd * vol[i - 1])
        counts[i - 1] = count

        cidx = np.random.choice(
            np.where(zbidx)[0], size=n - count, replace=False)
        zbidx[cidx] = False
        zidx &= (zbidx | znbidx)

    return zidx


def get_lightcone_files(nside, pix, radii, filebase):
    """
    Get the lightcone files corresponding to the current
    jackknife region
    """

    files = []

    for r in radii:
        r = int(r)
        # read in default file to get nside of this radial bin
        hdr, idx = read_radial_bin('{}_{}_{}'.format(filebase, r, 0))
        file_nside = hdr[2]

        if nside == file_nside:
            file_pix = [pix]
        elif nside < file_nside:
            umap = hp.ud_grade(np.arange(12 * nside**2), file_nside,
                               order_in='NESTED', order_out='NESTED')
            if not hasattr(pix, '__iter__'):
                pix = [pix]

            for i, p in enumerate(pix):
                if i == 0:
                    file_pix, = np.where(umap == p)
                else:
                    fp = np.where(umap == p)
                    file_pix = np.hstack([file_pix, fp])
        else:
            umap = hp.ud_grade(np.arange(12 * file_nside**2), nside,
                               order_in='NESTED', order_out='NESTED')
            file_pix = [umap[pix]]

        files.extend(['{}_{}_{}'.format(filebase, r, p) for p in file_pix])

    return files


def read_file_downsample(filename, nd=1e-2, read_vel=False):
    """
    Read in particles and downsample.

    inputs
    ------
    filename - str
      Name of the file to read
    f_down - float
      Factor to downsample the particles by
    read_vel - bool
      True if you would also like velocities (for RSD)

    outputs
    -------
    pos - array
      (N,3) array of particle positions in comoving Mpc/h
    vel - array
      (N,3) array of velocities in km/s
    """
    if read_vel:
        hdr, idx, pos, vel = read_radial_bin(filename, read_pos=True,
                                             read_vel=True)
        pos = pos.reshape((len(pos) // 3, 3))
        vel = vel.reshape((len(vel) // 3, 3))
        file_nside = hdr[2]

    else:
        hdr, idx, pos = read_radial_bin(filename, read_pos=True)
        pos = pos.reshape((len(pos) // 3, 3))
        file_nside = hdr[2]

    rbin = int(filename.split('_')[-2])
    r = np.sqrt(np.sum(pos**2, axis=1))
    idx_down = select_matter(r, 25 * rbin, 25 * (rbin + 1), file_nside, nd)

    if read_vel:
        return pos[idx_down], vel[idx_down]
    else:
        return pos[idx_down]


def read_files_lightcone(files, z_min, z_max, cosmology,
                         nside_jack, pix_jack,
                         nd=1e-2, rsd=True):

    pdtype = np.dtype([('px', np.float), ('py', np.float), ('pz', np.float),
                       ('vx', np.float), ('vy', np.float), ('vz', np.float),
                       ('z_cos', np.float)])

    for i, f in enumerate(files):
        if i == 0:
            if rsd:
                pos, vel = read_file_downsample(f, nd=nd,
                                                read_vel=True)
            else:
                pos = read_file_downsample(f, nd=nd)
        else:
            if rsd:
                p, v = read_file_downsample(f, nd=nd,
                                            read_vel=True)
                vel = np.vstack([vel, v])
            else:
                p = read_file_downsample(f, nd=nd)

            pos = np.vstack([pos, p])

    parts = np.zeros(len(pos), dtype=pdtype)

    # cut to min/max redshifts
    # temporarily store radii in z column
    parts['z_cos'] = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)

    parts['z_cos'] = cosmo.zofR(parts['z_cos'])

    pix = hp.vec2pix(nside_jack, pos[:, 0], pos[:, 1], pos[:, 2], nest=True)
    idx = ((z_min < parts['z_cos']) & (parts['z_cos'] <= z_max)
           & (pix == pix_jack))
    pos = pos[idx]
    vel = vel[idx]
    parts = parts[idx]
    parts['px'] = pos[:, 0]
    parts['py'] = pos[:, 1]
    parts['pz'] = pos[:, 2]
    parts['vx'] = vel[:, 0]
    parts['vy'] = vel[:, 1]
    parts['vz'] = vel[:, 2]

    del vel, pos

    return parts


def mask_parts(part, mask_pix, rot):
    vec = np.zeros((len(part), 3))

    vec[:, 0] = part['px']
    vec[:, 1] = part['py']
    vec[:, 2] = part['pz']

    rvec = np.dot(vec, np.linalg.inv(rot))
    dec, ra = hp.vec2ang(rvec)
    pix = hp.ang2pix(4096, dec, ra, nest=True)

    dec = -dec * 180. / np.pi + 90
    ra = ra * 180. / np.pi

    pidx = np.in1d(pix, mask_pix)
    part = part[pidx]

    return part


def save_downsampled_particles(outpath, part, rank):
    """
    Save a downsampled particle catalog with masking if given
    """
    op = outpath.format(rank)
    fits = fitsio.FITS(op, 'rw')

    if len(fits) > 1:
        fits[-1].append(part)
    else:
        fits.write(part)


def write_parts_to_mastercat(outbase, mastercat_file):

    partfiles = glob(outbase + '/downsampled_particles*')

    with h5py.File(mastercat_file, 'r+') as fp:

        for i, f in enumerate(partfiles):

            parts = fitsio.read(f)
            cols = parts.dtype.names
            nparts = len(parts)

            if i == 0:
                for name in cols:
                    try:
                        fp.create_dataset('catalog/downsampled_dm/' + name,
                                          maxshape=(None,), shape=(nparts,),
                                          dtype=parts.dtype[name],
                                          chunks=True)
                    except:
                        del fp['catalog/downsampled_dm/' + name]
                        fp.create_dataset('catalog/downsampled_dm/' + name,
                                          maxshape=(None,), shape=(nparts,),
                                          dtype=parts.dtype[name],
                                          chunks=True)                        
                    fp['catalog/downsampled_dm/' + name][:] = parts[name]
            else:
                for name in cols:
                    ds_size = fp['catalog/downsampled_dm/' + name].shape[0]
                    fp['catalog/downsampled_dm/' + name].resize(ds_size + len(parts),
                                                                axis=0)
                    fp['catalog/downsampled_dm/' + name][ds_size:] = parts[name]


def downsample(cosmo, lightcone_base, mask_pix, rot, rank, outbase,
               zrange=[0.0, 1.5], comoving_nd=1e-2, nside_jack=4):

    # rotate mask centers to simulation coordinates
    vec = np.array(hp.pix2vec(4096, mask_pix, nest=True)).T
    vec = np.dot(vec, rot)

    pix = hp.vec2pix(int(nside_jack), vec[:, 0], vec[:, 1], vec[:, 2],
                     nest=True)
    upix = np.unique(pix)

    if rank == 0:
        print('Total number of pix: {}'.format(len(upix)))

    for i, p in enumerate(upix[rank::size]):
        print('{}: Working on pixel {}'.format(rank, p))
        sys.stdout.flush()

        rrange = cosmo.rofZ(np.array(zrange))

        r_min_idx = rrange[0] // 25
        r_max_idx = rrange[1] // 25
        ridx = np.arange(r_min_idx, r_max_idx + 1)
        print('Radial cells to read: {}'.format(ridx))

        # read in particles in this pixel/redshift range
        files_lightcone = get_lightcone_files(nside_jack, p, ridx,
                                              lightcone_base)
        part = read_files_lightcone(files_lightcone, zrange[0], zrange[1],
                                    cosmo, nside_jack, p,
                                    nd=comoving_nd)

        part = mask_parts(part, mask, rot)

        save_downsampled_particles(outbase+'/downsampled_particles.{}.fits',
                                   part, rank)


if __name__ == '__main__':

    addgals_cfg_file = sys.argv[1]
    footprint_cfg_file = sys.argv[2]
    partpath = sys.argv[3]
    outbase = sys.argv[4]

    if len(sys.argv) > 5:
        mastercat_file = sys.argv[5]
        write_to_mastercat = True
    else:
        write_to_mastercat = False

    if len(sys.argv) > 6:
        only_mastercat = bool(sys.argv[6])
    else:
        only_mastercat = False

    addgals_config = parseConfig(addgals_cfg_file)

    with open(footprint_cfg_file, 'r') as fp:
        footprint_cfg = yaml.load(fp)

    cc = addgals_config['Cosmology']
    cosmo = Cosmology(**cc)

    mask_file = footprint_cfg['DepthFile']
    mask = fitsio.read(mask_file)
    mask = mask['HPIX']

    rotation_file = footprint_cfg['MatPath']

    with open(rotation_file, 'rb') as fp:
        rot_matrix = pickle.load(fp, encoding='latin1')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if not only_mastercat:
        downsample(cosmo, partpath, mask, rot_matrix, rank, outbase)

    comm.Barrier()

    if rank == 0:
        if write_to_mastercat:
            write_parts_to_mastercat(outbase, mastercat_file)

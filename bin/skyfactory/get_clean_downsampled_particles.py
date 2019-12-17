#!/usr/bin/env python
from __future__ import print_function, division
from astropy.cosmology import FlatLambdaCDM
from mpi4py import MPI
from glob import glob
import healpy as hp
import numpy  as np
import fitsio
from pixlc import pixLC
import sys

def select_matter(r, rmin, rmax):

    farea = hp.nside2pixarea(2)/(12*hp.nside2pixarea(1))
    cos  = FlatLambdaCDM(100, 0.286)

    zidx = np.ones(len(r), dtype=np.bool)
    
    rbins = np.arange(rmin,rmax+5,5)
    vol   = [farea*4*np.pi*(rbins[i+1]**3 - rbins[i]**3) / 3 for i in range(len(rbins)-1)]
    idx   = np.digitize(r, bins=rbins)
    counts = np.zeros(len(rbins)-1)

    for i in np.arange(1,len(rbins)):
        zbidx = (idx == i) & zidx
        znbidx = idx!=i
        n     = np.sum(zbidx)
        count = int(1e-2 * vol[i-1])
        counts[i-1] = count

        cidx  = np.random.choice(np.where(zbidx)[0], size=n-count, replace=False)
        zbidx[cidx] = False
        zidx &= (zbidx | znbidx)

    return zidx

def generate_z_of_r_table(omegam, omegal, zmax=2.0, npts=1000):

    c = 2.9979e5
    da = (1.0 - (1.0/(zmax+1)))/npts
    dtype = np.dtype([('r', np.float), ('z', np.float)])
    z_of_r_table = np.ndarray(npts, dtype=dtype)
    Thisa = 1.0
    z_of_r_table['z'][0] = 1.0/Thisa - 1.
    z_of_r_table['r'][0] = 0.0
    for i in range(1, npts):
        Thisa = 1. - da*float(i)
        ThisH = 100.*np.sqrt(omegam/Thisa**3 + omegal)
        z_of_r_table['z'][i] = 1./Thisa - 1
        z_of_r_table['r'][i] = z_of_r_table['r'][i-1] + 1./(ThisH*Thisa*Thisa)*da*c


    return z_of_r_table



def z_of_r(r, table):

    npts = len(table)
    try:
        nz = len(r)
    except:
        nz = 1

    zred = np.zeros(nz)-1

    if nz==1:
        for i in range(1, npts):
            if (table['r'][i] > r): break
        slope = (table['z'][i] - table['z'][i-1])/(table['r'][i]-table['r'][i-1])
        zred = table['z'][i-1] + slope*(r-table['r'][i-1])
    else:
        for i in range(1, npts):
            ii, = np.where((r >= table['r'][i-1]) & (r < table['r'][i]))
            count = len(ii)
            if (count == 0): continue
            slope = (table['z'][i] - table['z'][i-1])/(table['r'][i]-table['r'][i-1])
            zred[ii] = table['z'][i-1] + slope*(r[ii]-table['r'][i-1])

    return zred

def get_lightcone_files(nside, pix, radii, filebase):
    """
    Get the lightcone files corresponding to the current
    jackknife region
    """

    files = []
    print('input pix: {}'.format(pix))
    if not hasattr(pix, '__iter__'):
        pix = [pix]
    
    for r in radii:
        r = int(r)
        #read in default file to get nside of this radial bin
        hdr, idx   = pixLC.read_radial_bin('{}_{}_{}'.format(filebase,r,0))
        file_nside = hdr[2]

        if nside==file_nside:
            file_pix = pix
        elif nside < file_nside:
            umap = hp.ud_grade(np.arange(12*nside**2), file_nside,
                                order_in='NESTED', order_out='NESTED')

            for i,p in enumerate(pix):
                if i==0:
                    file_pix, = np.where(umap==p)
                else:
                    fp        = np.where(umap==p)
                    file_pix  = np.hstack([file_pix, fp])
        else:
#            print(file_nside)
#            print(nside)
#            print(pix)

            umap      = hp.ud_grade(np.arange(12*file_nside**2), nside,
                                      order_in='NESTED', order_out='NESTED')
            file_pix  = umap[pix]

        print('reading pixels: {}'.format(file_pix))

        files.extend(['{}_{}_{}'.format(filebase,r,p) for p in file_pix])

    return files


if __name__=='__main__':

    gpath  = sys.argv[1]
    ppath  = sys.argv[2]
    opath  = sys.argv[3]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    tfiles = np.array(glob(gpath))
    pix    = np.array([int(f.split('.')[-2]) for f in tfiles])
    print(pix)
    #get nside=2 pixels which fully lie within the two octants of the catalog
    dmap   = hp.ud_grade(np.arange(12*2**2), 8, order_in='NESTED', order_out='NESTED')
    dmap[~np.in1d(np.arange(12*8**2), pix)] = -99
    c,e = np.histogram(dmap[dmap!=-99],bins=np.arange(np.min(dmap[dmap!=-99]), np.max(dmap)+2))
    pix_footprint = e[:-1][c==16]
    print(pix_footprint)

    table = generate_z_of_r_table(0.286, 0.714, zmax=3.0, npts=5000)
    
    pix_rank = pix_footprint[rank::size]
    if not hasattr(pix_rank, '__iter__'):
        pix_rank = [pix_rank]

    print('rank {}: assigned {}'.format(rank,pix_rank))

    pdtype = np.dtype([('px',np.float),('py',np.float),('pz',np.float),('vx',np.float),
                       ('vy',np.float),('vz',np.float),('polar_ang',np.float),
                       ('azim_ang',np.float),('redshift',np.float)])
        
    for pix in pix_rank:
        pofile = fitsio.FITS('{}_matter.{}.fits'.format(opath, pix), 'rw')
        count = 0

        dumped  = True

        for i in range(0,16):
            lcfiles = get_lightcone_files(2, pix, range(i*10,(i+1)*10), ppath)
            print('rank {} working on set {}'.format(rank, i))

            for j, lf in enumerate(lcfiles):
                hdr, idx, p, v = pixLC.read_radial_bin(lf, read_pos=True,
                                                       read_vel=True)

                p = p.reshape((len(p)//3, 3))
                v = v.reshape((len(v)//3, 3))

                fpix = hp.vec2pix(2, p[:,0], p[:,1], p[:,2], nest=True)
                p   = p[fpix==pix]
                v   = v[fpix==pix]
                
                r   = np.sqrt(np.sum(p**2, axis=1))
                z   = z_of_r(r, table)

                if j==0:
                    pos      = p
                    vel      = v
                    radius   = r
                    redshift = z
                    dumped   = False
                else:
                    pos      = np.vstack([pos,p])
                    vel      = np.vstack([vel,v])
                    radius   = np.hstack([radius,r])
                    redshift = np.hstack([redshift,z])


            lidx = select_matter(radius, i*10*25, (i+1)*10*25)

            dec, ra = hp.vec2ang(pos[lidx])
            dec     = -dec * 180. / np.pi + 90.
            ra      = ra * 180. / np.pi
                
            out       = np.zeros(np.sum(lidx), dtype=pdtype)
            out['px'] = pos[lidx,0]
            out['py'] = pos[lidx,1]
            out['pz'] = pos[lidx,2]
            out['vx'] = vel[lidx,0]
            out['vy'] = vel[lidx,1]
            out['vz'] = vel[lidx,2]
            out['redshift']  = redshift[lidx]
            out['polar_ang'] = dec
            out['azim_ang']  = ra

            if count==0:
                pofile.write(out,clobber=True)
            else:
                pofile[-1].append(out)

            count += 1

        pofile.close()

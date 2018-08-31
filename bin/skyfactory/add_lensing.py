from __future__ import print_function, division
from mpi4py import MPI
from glob import glob
import numpy.lib.recfunctions as rf
import healpy as hp
import numpy as np
import fitsio
import yaml
import sys

def compute_lensing(g, shear, halos=False):


    lensfields = ['GAMMA1', 'GAMMA2', 'KAPPA', 'W', 'MU', 'TRA', 'TDEC', 'RA', 'DEC']

    if not halos:
        lensfields.append("LMAG")

    nmimg  = 0
    adtype = []
    fields = []
    data   = []

    for f in lensfields:
        if f not in g.dtype.names:
            fields.append(f)
            print(f)
            sys.stdout.flush()
            if f == 'RA':
                adtype.append(np.dtype([(f,np.float)]))
                theta, phi = hp.vec2ang(g[['PX','PY','PZ']].view((g.dtype['PX'],3)))
                dec, ra =  -np.degrees(theta-np.pi/2.), np.degrees(np.pi*2.-phi)
                data.append(ra)
            elif f == 'DEC':
                adtype.append(np.dtype([(f,np.float)]))
                theta, phi = hp.vec2ang(g[['PX','PY','PZ']].view((g.dtype['PX'],3)))
                dec, ra =  -np.degrees(theta-np.pi/2.), np.degrees(np.pi*2.-phi)
                data.append(dec)
            elif f == 'LMAG':
                adtype.append(np.dtype([(f,(np.float,5))]))
                data.append(np.zeros(len(g)))
            else:
                adtype.append(np.dtype([(f,np.float)]))
                data.append(np.zeros(len(g)))

    if len(fields)>0:
        g = rf.append_fields(g, fields, data=data,
                             dtypes=adtype, usemask=False)

    print("Number of rows in shear catalog: {0}".format(len(shear)))
    print("Number of rows in galaxy catalog: {0}".format(len(g)))
    sys.stdout.flush()

    #Make array long enough for multiply imaged galaxies
    uidx, sidx = np.unique(shear['index'], return_index=True)

    #unique indicies should have same length as galaxy catalog
    assert(len(sidx)==len(g))

    #Get indices of multiply imaged galaxies
    miidx = np.arange(len(shear))
    miidx = miidx[~np.in1d(miidx,sidx)]
    nmi = len(miidx)

    #number of multiply imaged galaxies should be difference in lengths
    #of shear catalog and galaxy catalog
    assert(nmi == (len(shear)-len(g)))

    print('Number of multiply imaged galaxies: {0}'.format(nmi))
    sys.stdout.flush()

    g = np.hstack([g, g[shear['index'][miidx]]])

    shear['index'][miidx] = np.arange(nmi) + len(sidx)

    g = g[shear['index']]

    g['TRA'] = g['RA']
    g['TDEC'] = g['DEC']

    g['RA'] = shear['ra']
    g['DEC'] = shear['dec']

    g['GAMMA1'] = 0.5*(shear['A11'] - shear['A00'])
    g['GAMMA2'] = -0.5*(shear['A01'] + shear['A10'])
    g['KAPPA'] = 1.0 - 0.5*(shear['A00'] + shear['A11'])
    g['W'] = 0.5*(shear['A10'] - shear['A01'])

    #compute mu = 1/detA
    g['MU'] = 1./(shear['A11']*shear['A00'] - shear['A01']*shear['A10'])

    if not halos:
        #lens the size and magnitudes
        g['SIZE'] = g['TSIZE'] * np.sqrt(g['MU'])
        for im in range(g['AMAG'].shape[1]):
            g['LMAG'][:,im] = g['TMAG'][:,im] - 2.5*np.log10(g['MU'])

        #get intrinsic shape
        epss = g['TE'][:,0] + 1j * g['TE'][:,1]

        #;;get reduced complex shear g = (gamma1 + i*gamma2)/(1-kappa)
        gquot = (g['GAMMA1'] + 1j * g['GAMMA2']) / (1.-g['KAPPA'])

        #;;lens the shapes - see Bartelmann & Schneider (2001), Section 4.2
        lidx = np.abs(gquot) < 1
        eps = (1.0 + ( gquot * epss.conjugate())) / (epss.conjugate() + gquot.conjugate())
        eps[lidx] = (epss[lidx] + gquot[lidx]) / (1.0 + ( epss[lidx] * gquot[lidx].conjugate() ) )

        g['EPSILON'][:,0] = np.real(eps)
        g['EPSILON'][:,1] = np.imag(eps)

    return g


def add_lensing(gfiles, sfiles):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert(len(gfiles)==len(sfiles))

    gpix = np.array([int(gf.split('.')[-2]) for gf in gfiles])
    spix = np.array([int(sf.split('.')[-3].split('_')[0]) for sf in sfiles])

    gidx = gpix.argsort()
    sidx = spix.argsort()
    gpix = gpix[gidx]
    spix = spix[sidx]

    assert(all(gpix==spix))

    gfiles = gfiles[gidx]
    sfiles = sfiles[sidx]

    for i, f in enumerate(gfiles):
        if ('halo' in f) and ('halo' not in sfiles[i]):
            if ((i+1)<len(gpix)) and (gpix[i]==spix[i+1]):
                temp = sfiles[i+1]
                sfiles[i+1] = sfiles[i]
                sfiles[i] = temp
            elif gpix[i]==spix[i-1]:
                temp = sfiles[i-1]
                sfiles[i-1] = sfiles[i]
                sfiles[i] = temp

            assert(('halo' in f) and ('halo' in sfiles[i]))


    for gf,sf in zip(gfiles[rank::size],sfiles[rank::size]):

        print("[{1}] Lensing {0}".format(gf, rank))
        print("[{1}] Using  {0}".format(sf, rank))

        gfs = gf.split('/')
        gbase = "/".join(gfs[:-1])


        g     = fitsio.read(gf)
        shear = fitsio.read(sf)

        if 'halo' in gf:
            g     = compute_lensing(g, shear, halos=True)
        else:
            g     = compute_lensing(g, shear)

        gfss = gfs[-1].split('.')
        oname = "{0}/{1}_lensed.{2}.fits".format(gbase, gfss[0], gfss[1])

        print("[{1}] Writing lensed catalog to {0}".format(oname, rank))

        fits = fitsio.FITS(oname, 'rw')
        fits.write(g)


if __name__=='__main__':

    cfgfile = sys.argv[1]

    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp)

    snames = np.loadtxt(cfg['LensGalsList'], dtype=str)
    gnames = np.loadtxt(cfg['TruthGalsList'], dtype=str)

    add_lensing(gnames, snames)

from __future__ import print_function, division
from glob import glob
from mpi4py import MPI
from rot_mock_tools import rot_mock_file
import numpy.lib.recfunctions as rf
import numpy as np
import healpy as hp
import fitsio
import pickle
import yaml
import sys
import os

models = {
    'DR8':
        {'maglims':[20.425,21.749,21.239,20.769,19.344],
        'exptimes' : [21.00,159.00,126.00,99.00,15.00],
        'lnscat' : [0.284,0.241,0.229,0.251,0.264]
         },

    'STRIPE82':
        {
        'maglims' : [22.070,23.432,23.095,22.649,21.160],
        'exptimes' : [99.00,1172.00,1028.00,665.00,138.00],
        'lnscat' : [0.280,0.229,0.202,0.204,0.246]
        },

    'CFHTLS':
        {
        'maglims' : [24.298,24.667,24.010,23.702,22.568],
        'exptimes' : [2866.00,7003.00,4108.00,3777.00,885.00],
        'lnscat' : [0.259,0.244,0.282,0.258,0.273]
        },
    'DEEP2':
        {
        'maglims' : [24.730,24.623,24.092],
        'exptimes' : [7677.00,8979.00,4402.00],
        'lnscat' : [0.300,0.293,0.300]
        },
    'FLAMEX':
        {
        'maglims' : [21.234,20.929],
        'exptimes' : [259.00,135.00],
        'lnscat' : [0.300,0.289]
        },
    'IRAC':
        {
        'maglims' : [19.352,18.574],
        'exptimes' : [8.54,3.46],
        'lnscat' : [0.214,0.283]
        },
    'NDWFS':
        {
        'maglims' : [25.142,23.761,23.650],
        'exptimes' : [6391.00,1985.00,1617.00],
        'lnscat' : [0.140,0.294,0.272]
        },

    'RCS':
        {
        'maglims' : [23.939,23.826,23.067,21.889],
        'exptimes' : [2850.00,2568.00,1277.00,431.00],
        'lnscat' : [0.164,0.222,0.250,0.271]
        },
    'VHS':
        {
        'maglims' : [20.141,19.732,19.478],
        'exptimes' : [36.00,31.00,23.00],
        'lnscat' : [0.097,0.059,0.069]
        },
    'VIKING':
        {
        'maglims' : [21.643,20.915,20.768,20.243,20.227],
        'exptimes' : [622.00,246.00,383.00,238.00,213.00],
        'lnscat' : [0.034,0.048,0.052,0.040,0.066]
        },
    'DC6B':
        {
        'maglims' : [24.486,23.473,22.761,22.402],
        'exptimes' : [2379.00,1169.00,806.00,639.00],
        'lnscat' : [0.300,0.300,0.300,0.300]
        },

    'DES':
        {
        'maglims' : [24.956,24.453,23.751,23.249,21.459],
        'exptimes' : [14467.00,12471.00,6296.00,5362.00,728.00],
        'lnscat' : [0.2,0.2,0.2,0.2,0.2]
        },

    'BCS_LO':
        {
        'maglims' : [23.082,22.618,22.500,21.065],
        'exptimes' : [809.00,844.00,641.00,108.00],
        'lnscat' : [0.277,0.284,0.284,0.300]
        },

    'BCS':
        {
        'maglims' : [23.360,23.117,22.539,21.335],
        'exptimes' : [838.00,1252.00,772.00,98.00],
        'lnscat' : [0.276,0.272,0.278,0.279]
        },

    'DES_SV':
    {
	'maglims' : [23.621,23.232,23.008,22.374,20.663],
	'exptimes' : [4389.00,1329.00,1405.00,517.00,460.00],
	'lnscat' : [0.276,0.257,0.247,0.241,0.300]
        },

    'DES_SV_OPTIMISTIC':
        {
        'maglims' : [23.621+0.5,23.232+0.5,23.008,22.374,20.663],
        'exptimes' : [4389.00,1329.00,1405.00,517.00,460.00],
        'lnscat' : [0.276,0.257,0.247,0.241,0.300]
        },
    'WISE':
        {
        'maglims' : [19.352,18.574],
        'exptimes' : [8.54,3.46],
        'lnscat' : [0.214,0.283]
        },

    'DECALS':
        {
       'maglims' : [23.3,23.3,22.2,20.6,19.9],
       'exptimes' : [1000,3000,2000,1500,1500],
       'lnscat' : [0.2,0.2,0.2,0.2,0.2]
       },

}

def calc_nonuniform_errors(exptimes,limmags,mag_in,nonoise=False,zp=22.5,
                            nsig=10.0,fluxmode=False,lnscat=None,b=None,
                            inlup=False,detonly=False):

    f1lim = 10**((limmags-zp)/(-2.5))
    fsky1 = ((f1lim**2)*exptimes)/(nsig**2) - f1lim
    fsky1[fsky1<0.001] = 0.001

    if inlup:
        bnmgy = b*1e9
        tflux = exptimes*2.0*bnmgy*np.sinh(-np.log(b)-0.4*np.log(10.0)*mag_in)
    else:
        tflux = exptimes*10**((mag_in - zp)/(-2.5))

    noise = np.sqrt(fsky1*exptimes + tflux)

    if lnscat is not None:
        noise = np.exp(np.log(noise) + lnscat*np.random.randn(len(mag_in)))

    if nonoise:
        flux = tflux
    else:
        flux = tflux + noise*np.random.randn(len(mag_in))
        
    #convert to nanomaggies
        flux = flux/exptimes 
        noise = noise/exptimes

        flux  = flux * 10 ** ((zp - 22.5)/-2.5)
        noise = noise * 10 ** ((zp - 22.5)/-2.5)

    if fluxmode:
        mag = flux
        mag_err = noise
    else:
        if b is not None:
            bnmgy = b*1e9
            flux_new = flux
            noise_new = noise
            mag = 2.5*np.log10(1.0/b) - asinh2(0.5*flux_new/(bnmgy))/(0.4*np.log(10.0))

            mag_err = 2.5*noise_new/(2.*bnmgy*np.log(10.0)*np.sqrt(1.0+(0.5*flux_new/(bnmgy))**2.))

        else:
            mag = 22.5-2.5*np.log10(flux)
            mag_err = (2.5/np.log(10.))*(noise/flux)

            #temporarily changing to cut to 10-sigma detections in i,z
            bad = np.where((np.isfinite(mag)==False))
            nbad = len(bad)

            if detonly:
                mag[bad]=99.0
                mag_err[bad]=99.0
                
    return mag, mag_err


def calc_uniform_errors(model, tmag, maglims, exptimes, lnscat, zp=22.5):

    nmag=len(maglims)
    ngal=len(tmag)

    tmag = tmag.reshape(len(tmag),nmag)

    #calculate fsky1 -- sky in 1 second
    flux1_lim = 10**((maglims-zp)/(-2.5))
    flux1_lim[flux1_lim < 120/exptimes] = 120/exptimes[flux1_lim < 120/exptimes]
    fsky1 = (flux1_lim**2*exptimes)/100. - flux1_lim

    oflux=np.zeros((ngal, nmag))
    ofluxerr=np.zeros((ngal, nmag))
    omag=np.zeros((ngal, nmag))
    omagerr=np.zeros((ngal, nmag))
    offset = 0.0

    for i in range(nmag):
        tflux = exptimes[i] * 10**((tmag[:,i]-offset-zp)/(-2.5))
        noise = np.exp(np.log(np.sqrt(fsky1[i]*exptimes[i] + tflux))
                    + lnscat[i]*np.random.randn(ngal))

        flux = tflux + noise*np.random.randn(ngal)

        oflux[:,i] = flux / exptimes[i]
        ofluxerr[:,i] = noise/exptimes[i]

        oflux[:,i]    *= 10 ** ((zp - 22.5) / -2.5)
        ofluxerr[:,i] *= 10 ** ((zp - 22.5) / -2.5)
        

        omag[:,i] = 22.5-2.5*np.log10(oflux[:,i])
        omagerr[:,i] = (2.5/np.log(10.))*(ofluxerr[:,i]/oflux[:,i])

        bad,=np.where(~np.isfinite(omag[:,i]))
        nbad = len(bad)
        if (nbad > 0) :
            omag[bad,i] = 99.0
            omagerr[bad,i] = 99.0


    return omag, omagerr, oflux, ofluxerr

def make_output_structure(ngals, dbase_style=False, bands=None, nbands=None,
                             all_obs_fields=True, blind_obs=False):

    if all_obs_fields & dbase_style:
        if bands is None:
            raise(ValueError("Need names of bands in order to use database formatting!"))

        fields = [('ID', np.int), ('RA', np.float), ('DEC', np.float),
                    ('EPSILON1',np.float), ('EPSILON2', np.float),
                    ('SIZE',np.float), ('PHOTOZ_GAUSSIAN', np.float)]

        for b in bands:
            fields.append(('MAG_{0}'.format(b.upper()),np.float))
            fields.append(('MAGERR_{0}'.format(b.upper()),np.float))
            fields.append(('FLUX_{0}'.format(b.upper()),np.float))
            fields.append(('IVAR_{0}'.format(b.upper()),np.float))

    if all_obs_fields & (not dbase_style):

        fields = [('ID', np.int), ('RA', np.float), ('DEC', np.float),
                    ('EPSILON1',np.float), ('EPSILON2', np.float),
                    ('SIZE',np.float), ('PHOTOZ_GAUSSIAN', np.float),
                    ('MAG',(np.float,nbands)), ('FLUX',(np.float,nbands)),
                    ('MAGERR',(np.float,nbands)),('IVAR',(np.float,nbands))]

    if (not all_obs_fields) & dbase_style:
        fields = [('ID',np.int)]
        for b in bands:
            fields.append(('MAG_{0}'.format(b.upper()),np.float))
            fields.append(('MAGERR_{0}'.format(b.upper()),np.float))
            fields.append(('FLUX_{0}'.format(b.upper()),np.float))
            fields.append(('IVAR_{0}'.format(b.upper()),np.float))

    if (not all_obs_fields) & (not dbase_style):
        fields = [('ID', np.int), ('MAG',(np.float,nbands)),
                    ('FLUX',(np.float,nbands)),('MAGERR',(np.float,nbands)),
                    ('IVAR',(np.float,nbands))]

    if blind_obs:
        fields.extend([('M200', np.float), ('Z', np.float),
                        ('CENTRAL', np.int)])

    odtype = np.dtype(fields)

    out = np.zeros(ngals, dtype=odtype)

    return out


def apply_nonuniform_errormodel(g, oname, d, dhdr,
                                survey, magfile=None, usemags=None,
                                nest=False, bands=None, all_obs_fields=True,
                                dbase_style=True, use_lmag=True,
                                sigpz=0.03, blind_obs=False, filter_obs=True,
                                refbands=None, zp=22.5):
    
    if magfile is not None:
        mags = fitsio.read(magfile)
        if use_lmag:
            if ('LMAG' in mags.dtype.names) and (mags['LMAG']!=0).any():
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
            if ('LMAG' in g.dtype.names) and (g['LMAG']!=0).any():
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
        mnames  = ['MAG_{0}'.format(b.upper()) for b in bands]
        menames = ['MAGERR_{0}'.format(b.upper()) for b in bands]
        fnames  = ['FLUX_{0}'.format(b.upper()) for b in bands]
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
    oname = "{0}/{1}_obs.{2}.fits".format(odir,obase,fs[-2])

    #get mags to use
    if usemags is None:
        nmag = omag.shape[1]
        usemags = range(nmag)
    else:
        nmag = len(usemags)

    #make output structure
    obs = make_output_structure(len(g), dbase_style=dbase_style, bands=bands,
                                nbands=len(usemags),
                                all_obs_fields=all_obs_fields,
                                blind_obs=blind_obs)

    if ("Y1" in survey) | ("Y3" in survey) | (survey=="DES") | (survey=="SVA") | (survey=='Y3'):
        mindec = -90.
        maxdec = 90
        minra = 0.0
        maxra = 360.

    elif survey=="DR8":
        mindec = -20
        maxdec = 90
        minra = 0.0
        maxra = 360.

    maxtheta=(90.0-mindec)*np.pi/180.
    mintheta=(90.0-maxdec)*np.pi/180.
    minphi=minra*np.pi/180.
    maxphi=maxra*np.pi/180.

    #keep pixels in footprint
    #theta, phi = hp.pix2ang(dhdr['NSIDE'],d['HPIX'])
    #infp = np.where(((mintheta < theta) & (theta < maxtheta)) & ((minphi < phi) & (phi < maxphi)))
    #d = d[infp]

    #match galaxies to correct pixels of depthmap

    theta = (90-g['DEC'])*np.pi/180.
    phi   = (g['RA']*np.pi/180.)

    pix   = hp.ang2pix(dhdr['NSIDE'],theta, phi, nest=nest)

    guse = np.in1d(pix, d['HPIX'])
    guse, = np.where(guse==True)

    if not any(guse):
        print("No galaxies in this pixel are in the footprint")
        return

    pixind = d['HPIX'].searchsorted(pix[guse],side='right')
    pixind -= 1

    oidx = np.zeros(len(omag), dtype=bool)
    oidx[guse] = True

    for ind,i in enumerate(usemags):

        flux, fluxerr = calc_nonuniform_errors(d['EXPTIMES'][pixind,ind],
                                               d['LIMMAGS'][pixind,ind],
                                               omag[guse,i], fluxmode=True,
                                               zp=zp)

        if not dbase_style:

            obs['OMAG'][:,ind] = 99
            obs['OMAGERR'][:,ind] = 99

            obs['FLUX'][guse,ind] = flux
            obs['IVAR'][guse,ind] = 1/fluxerr**2
            obs['OMAG'][guse,ind] = 22.5 - 2.5*np.log10(flux)
            obs['OMAGERR'][guse,ind] = 1.086*fluxerr/flux

            bad = (flux<=0)

            obs['OMAG'][guse[bad],ind] = 99.0
            obs['OMAGERR'][guse[bad],ind] = 99.0

            r = np.random.rand(len(pixind))

            if len(d['FRACGOOD'].shape)>1:
                bad = r>d['FRACGOOD'][pixind,ind]
            else:
                bad = r>d['FRACGOOD'][pixind]

            if len(bad)>0:
                obs['OMAG'][guse[bad],ind] = 99.0
                obs['OMAGERR'][guse[bad],ind] = 99.0

            if filter_obs and (ind in refnames):
                oidx &= obs['OMAG'][:,ind] < d['LIMMAGS'][pixind,ind]

        else:
            obs[mnames[ind]]  = 99.0
            obs[menames[ind]] = 99.0

            obs[fnames[ind]][guse]  = flux
            obs[fenames[ind]][guse] = 1/fluxerr**2
            obs[mnames[ind]][guse]  = 22.5 - 2.5*np.log10(flux)
            obs[menames[ind]][guse] = 1.086*fluxerr/flux

            bad = (flux<=0)

            obs[mnames[ind]][guse[bad]] = 99.0
            obs[menames[ind]][guse[bad]] = 99.0

            #Set fluxes, magnitudes of non detections to zero, 99
            ntobs = ~np.isfinite(flux)
            obs[fnames[ind]][guse[ntobs]]  = 0.0
            obs[fenames[ind]][guse[ntobs]] = 0.0
            obs[mnames[ind]][guse[ntobs]] = 99.0
            obs[mnames[ind]][guse[ntobs]] = 99.0

            r = np.random.rand(len(pixind))

            if len(d['FRACGOOD'].shape)>1:
                bad = r>d['FRACGOOD'][pixind,ind]
            else:
                bad = r>d['FRACGOOD'][pixind]
            if any(bad):
                obs[mnames[ind]][guse[bad]]  = 99.0
                obs[menames[ind]][guse[bad]] = 99.0

            if (filter_obs) and (mnames[ind] in refnames):
                oidx[guse] &= obs[mnames[ind]][guse] < d['LIMMAGS'][pixind,ind]
                

    obs['RA']              = g['RA']
    obs['DEC']             = g['DEC']
    obs['ID']              = g['ID']
    obs['EPSILON1']        = g['EPSILON'][:,0]
    obs['EPSILON2']        = g['EPSILON'][:,1]
    obs['SIZE']            = g['SIZE']
    obs['PHOTOZ_GAUSSIAN'] = g['Z'] + sigpz * (1 + g['Z']) * (np.random.randn(len(g)))

    if blind_obs:
        obs['M200']    = g['M200']
        obs['CENTRAL'] = g['CENTRAL']
        obs['Z']       = g['Z']

    fitsio.write(oname, obs, clobber=True)

    if filter_obs:
        soname = oname.split('.')
        soname[-3] += '_rmp'
        roname = '.'.join(soname)
        fitsio.write(roname, obs[oidx], clobber=True)

    return oidx


def apply_uniform_errormodel(g, oname, survey, magfile=None, usemags=None,
                              bands=None, all_obs_fields=True,
                              dbase_style=True, use_lmag=True,
                              sigpz=0.03, blind_obs=False, filter_obs=True,
                              refbands=None, zp=22.5):
    
    if magfile is not None:
        mags = fitsio.read(magfile)
        if use_lmag:
            if ('LMAG' in mags.dtype.names) and (mags['LMAG']!=0).any():
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
            if ('LMAG' in g.dtype.names) and (g['LMAG']!=0).any():
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
        mnames  = ['MAG_{0}'.format(b.upper()) for b in bands]
        menames = ['MAGERR_{0}'.format(b.upper()) for b in bands]
        fnames  = ['FLUX_{0}'.format(b.upper()) for b in bands]
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
    oname = "{0}/{1}_obs.{2}.fits".format(odir,obase,fs[-2])

    #get mags to use
    if usemags is None:
        nmag = omag.shape[1]
        usemags = range(nmag)
    else:
        nmag = len(usemags)

    #make output structure
    obs = make_output_structure(len(g), dbase_style=dbase_style, bands=bands,
                                nbands=len(usemags),
                                all_obs_fields=all_obs_fields,
                                blind_obs=blind_obs)

    if ("Y1" in survey) | (survey=="DES") | (survey=="SVA"):
        mindec = -90.
        maxdec = 90
        minra = 0.0
        maxra = 360.

    elif survey=="DR8":
        mindec = -20
        maxdec = 90
        minra = 0.0
        maxra = 360.

    maxtheta=(90.0-mindec)*np.pi/180.
    mintheta=(90.0-maxdec)*np.pi/180.
    minphi=minra*np.pi/180.
    maxphi=maxra*np.pi/180.

    maglims = np.array(models[model]['maglims'])
    exptimes = np.array(models[model]['exptimes'])
    lnscat = np.array(models[model]['lnscat'])

    oidx = np.ones(len(omag), dtype=bool)

    for ind,i in enumerate(usemags):

        _, _, flux, fluxerr = calc_uniform_errors(model, omag[:,i],
                                                  np.array([maglims[ind]]), 
                                                  np.array([exptimes[ind]]),
                                                  np.array([lnscat[ind]]),
                                                  zp=zp)

        flux    = flux.reshape(len(flux))
        fluxerr = fluxerr.reshape(len(fluxerr))

        if not dbase_style:

            obs['FLUX'][:,ind] = flux
            obs['IVAR'][:,ind] = 1/fluxerr**2
            obs['OMAG'][:,ind] = 22.5 - 2.5*np.log10(flux)
            obs['OMAGERR'][:,ind] = 1.086*fluxerr/flux

            bad = (flux<=0)

            obs['OMAG'][bad,ind] = 99.0
            obs['OMAGERR'][bad,ind] = 99.0

            if filter_obs and (ind in refnames):
                oidx &= obs['OMAG'][:,ind] < maglims[ind]

        else:
            obs[mnames[ind]]  = 99.0
            obs[menames[ind]] = 99.0

            obs[fnames[ind]]  = flux
            obs[fenames[ind]] = 1/fluxerr**2
            obs[mnames[ind]]  = 22.5 - 2.5*np.log10(flux)
            obs[menames[ind]] = 1.086*fluxerr/flux

            bad = (flux<=0)

            obs[mnames[ind]][bad] = 99.0
            obs[menames[ind]][bad] = 99.0

            if (filter_obs) and (mnames[ind] in refnames):
                print('filtering {}'.format(mnames[ind]))
                oidx &= obs[mnames[ind]] < maglims[ind]
            else:
                print('mnames[ind]: {}'.format(mnames[ind]))
                
    print('filter_obs: {}'.format(filter_obs))
    print('refnames: {}'.format(refnames))
    print('maglims: {}'.format(maglims))
    print('oidx.any(): {}'.format(oidx.any()))

    obs['RA']              = g['RA']
    obs['DEC']             = g['DEC']
    obs['ID']           = g['ID']
    obs['EPSILON1']        = g['EPSILON'][:,0]
    obs['EPSILON2']        = g['EPSILON'][:,1]
    obs['SIZE']            = g['SIZE']
    obs['PHOTOZ_GAUSSIAN'] = g['Z'] + sigpz * (1 + g['Z']) * (np.random.randn(len(g)))

    if blind_obs:
        obs['M200']    = g['M200']
        obs['CENTRAL'] = g['CENTRAL']
        obs['Z']       = g['Z']

    fitsio.write(oname, obs, clobber=True)

    if filter_obs:
        soname = oname.split('.')
        soname[-3] += '_rmp'
        roname = '.'.join(soname)
        fitsio.write(roname, obs[oidx], clobber=True)


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cfgfile = sys.argv[1]
    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp)

    gpath = cfg['GalPath']
    model = cfg['Model']
    odir  = cfg['OutputDir']
    obase = cfg['OutputBase']

    fnames = np.array(glob(gpath))

    if 'DepthFile' in cfg.keys():
        dfile = cfg['DepthFile']
        uniform = False
        if 'Nest' in cfg.keys():
            nest = bool(cfg['Nest'])
        else:
            nest = False

        d,dhdr = fitsio.read(dfile, header=True)
        pidx = d['HPIX'].argsort()
        d = d[pidx]
    else:
        uniform = True

    if ('MagPath' in cfg.keys()) and (cfg['MagPath'] is not None):
        mnames  = np.array(glob(cfg['MagPath']))

        fpix = np.array([int(f.split('.')[-2]) for f in fnames])
        mpix = np.array([int(f.split('.')[-2]) for f in mnames])

        fidx = fpix.argsort()
        midx = mpix.argsort()

        assert((fpix[fidx]==mpix[midx]).all())

        fnames = fnames[fidx]
        mnames = mnames[midx]

    else:
        mnames = [None]*len(fnames)

    if 'UseMags' in cfg.keys():
        usemags = cfg['UseMags']
    else:
        usemags = None

    if ('DataBaseStyle' in cfg.keys()) & (cfg['DataBaseStyle']==True):
        if ('Bands' in cfg.keys()):
            dbstyle = True
            bands   = cfg['Bands']
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

    truth_only = cfg.pop('TruthOnly',False)

    if rank==0:
        try:
            os.makedirs(odir)
        except Exception as e:
            pass

    if ('RotOutDir' in cfg.keys()):
        if ('MatPath' in cfg.keys()):
            rodir = cfg['RotOutDir']
            robase = cfg['RotBase']
            rpath = cfg['MatPath']
            with open(rpath, 'r') as fp:
                rot    = pickle.load(fp)
            try:
                os.makedirs(rodir)
            except Exception as e:
                pass
        else:
            raise(KeyError("No Matrix path!"))

    else:
        rodir = None
        rpath = None
        rot   = None
        robase= None


    print("Rank {0} assigned {1} files".format(rank, len(fnames[rank::size])))

    for fname, mname in zip(fnames[rank::size],mnames[rank::size]):
        if rodir is not None:
            p = fname.split('.')[-2]
            nfname = "{0}/{1}.{2}.fits".format(rodir,robase,p)
            g = rot_mock_file(fname,rot,nfname,
                    footprint=d,nside=dhdr['NSIDE'],nest=nest)

            #if returns none, no galaxies in footprint
            if g is None: continue
        else:
            g = fitsio.read(fname)

        fs = fname.split('.')
        fp = fs[-2]

        oname = "{0}/{1}.{3}.fits".format(odir, obase, model, fp)

        if truth_only: continue

        if uniform:
            apply_uniform_errormodel(g, oname, model, magfile=mname,
                                     usemags=usemags,
                                     bands=bands,
                                     all_obs_fields=all_obs_fields,
                                     dbase_style=dbstyle,
                                     use_lmag=use_lmag,
                                     blind_obs=blind_obs,
                                     filter_obs=filter_obs,
                                     refbands=refbands,
                                     zp=zp)

        else:
            oidx = apply_nonuniform_errormodel(g, oname, d, dhdr,
                                               model, magfile=mname,
                                               usemags=usemags,
                                               nest=nest, bands=bands,
                                               all_obs_fields=all_obs_fields,
                                               dbase_style=dbstyle,
                                               use_lmag=use_lmag,
                                               blind_obs=blind_obs,
                                               filter_obs=filter_obs,
                                               refbands=refbands,
                                               zp=zp)

    if rank==0:
        print("*******Rotation and error model complete!*******")

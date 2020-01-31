from __future__ import print_function, division
from helpers import SimulationAnalysis
from astropy.cosmology import FlatLambdaCDM
try:
    from halotools.sim_manager import TabularAsciiReader
    noht = False
except:
    noht = True
import astropy.constants as const
import numpy as np
import warnings
import fitsio
import os

from .abundancematch import abundanceMatchSnapshot
from .rdelmag        import fitSnapshot

class Simulation(object):

    def __init__(self, name, boxsize, snapdirs, 
                   hfiles, rnnfiles, outdir, 
                   h, omega_m, zs=None, zmin=None, zmax=None, 
                   zstep=None, nz=None, shamfiles=None,
                   compressed_hlist=False, simtype='LGADGET2',
                   scaleh=False, scaler=False, strscale=None,
                   snapnums=None, ctrees_version=1):

        self.name     = name
        self.boxsize  = boxsize
        self.h        = h
        self.omegam   = omega_m
        self.cos      = FlatLambdaCDM(self.h * 100, self.omegam)
        self.snapdirs= np.array(snapdirs)
        self.hfiles   = np.array(hfiles)
        self.rnnfiles = np.array(rnnfiles)
        self.outdir   = outdir
        self.simtype  = simtype
        self.compressed_hlist = compressed_hlist
        self.scaleh = scaleh
        self.scaler = scaler
        self.strscale  = strscale
        self.snapnums  = snapnums
        self.files_associated = False
        self.ctrees_version = ctrees_version

        if shamfiles is not None:
            self.shamfiles = np.array(shamfiles)
        else:
            self.shamfiles = shamfiles

        self.lums = np.linspace(-30, -10, 100)

        if zs is not None:
            self.zs = zs
        elif (zmin is not None) & (zmax is not None) & (zstep is not None):
            self.zs = np.arange(zmin, zmax+zstep, zstep)
        elif (zmin is not None) & (zmax is not None) & (nz is not None):
            self.zs = np.linspace(zmin, zmax, nz)

        self.unitmap = {'mag':'magh', 'phi':'hmpc3dex'}


    def associateFiles(self):
        if self.files_associated: return 

        hfn = np.array([int(h.split('_')[-1].split('.')[0]) for h in self.hfiles])
        hz = self.zs[self.snapnums.searchsorted(hfn)]
        shz = self.strscale[self.snapnums.searchsorted(hfn)]

        crn = np.array([int(c.split('_')[-1]) for c in self.rnnfiles])
        cz = self.zs[self.snapnums.searchsorted(crn)]

        if len(hfn) > len(crn):
            inz = np.in1d(hfn, crn)
            self.hfiles = self.hfiles[inz]
            hfn = hfn[inz]
            hz  = hz[inz]
            shz = shz[inz]
            inz = np.in1d(crn, hfn)
            self.rnnfiles = self.rnnfiles[inz]
            crn = crn[inz]
            cz  = cz[inz]

        else:
            inz = np.in1d(crn, hfn)
            self.rnnfiles = self.rnnfiles[inz]
            crn = crn[inz]
            cz  = cz[inz]
            inz = np.in1d(hfn, crn)
            self.hfiles = self.hfiles[inz]
            hfn = hfn[inz]
            hz  = hz[inz]
            shz = shz[inz]

        assert(len(self.hfiles)==len(self.rnnfiles))

        hidx = hfn.argsort()
        cidx = crn.argsort()

        self.hfiles   = self.hfiles[hidx[::-1]]
        self.rnnfiles = self.rnnfiles[cidx[::-1]]
        self.zs       = hz[hidx[::-1]]
        self.strscale    = shz[hidx[::-1]]
        self.nums     = hfn[hidx[::-1]]

        print(self.hfiles)
        print(self.rnnfiles)
        print(self.zs)
        self.files_associated = True

    def getSHAMFileName(self, hfname, alpha, scatter, lfname, ind):

        fs = hfname.split('/')

        if self.nums[ind]<10:
            num  = '00{}'.format(self.nums[ind])
        else:
            num = '0{}'.format(self.nums[ind])

        fs[-1] = 'sham_{}_{}_{}_{}_{}'.format(self.name, 
                                                lfname,
                                                alpha,
                                                scatter,
                                                fs[-1])

        print(fs[-1])
        print(num)
        fn = fs[-1].replace(num, self.strscale[ind])
        fn = fn.replace('.list', '.fits')

        return fn

    def abundanceMatch(self, lf, alpha=[0.7], scatter=[0.17], debug=False,
                       parallel=False, startat=None, endat=None, grid=True):
        """
        Abundance match all of the hlists
        """
        print('amatch')
        self.associateFiles()        
        odtype = np.dtype([('PX', np.float),
                            ('PY', np.float),
                            ('PZ', np.float),
                            ('AMPROXY', np.float),
                            ('VMAX', np.float),
                            ('MVIR', np.float),
                            ('MPEAK', np.float),
                            ('LUMINOSITY', np.float),
                            ('CENTRAL', np.int),
                            ('UPID', np.int),
                            ('ID',np.int),
                            ('RVIR', np.float)])

        if startat is None:
            startat = 0
        if endat is None:
            endat = len(self.hfiles)

        hfiles = self.hfiles[startat:endat]
        zs = self.zs[startat:endat]

        if not hasattr(alpha, '__iter__'):
            alpha = [alpha]
        if not hasattr(scatter, '__iter__'):
            scatter = [scatter]

        if parallel:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            
            if grid:
                scatter = scatter[rank::size]
            else:
                hfiles = hfiles[rank::size]
                zs     = zs[rank::size]

        else: 
            rank = 0
            hfiles = self.hfiles
            zs = self.zs


        if rank == 0:
            try:
                os.makedirs('{0}/sham/'.format(self.outdir))
            except OSError as e:
                warnings.warn('Directory {0}/sham/ already exists!'.format(self.outdir), Warning)
                pass

        for i, hf in enumerate(hfiles):
            if not grid:
                ind = startat + rank + i * size
            else:
                ind = startat + i

            
            if not self.compressed_hlist:

                if self.ctrees_version == 1:
                    try:
                        fields = {'vmax':(71,'f4'),
                                  'mvir':(55,'f4'),
                                  'mvir_now':(10,'f4'),
                                  'mpeak_scale':(66,'f4'),
                                  'upid':(6,'i4'),
                                  'x':(17,'f4'),
                                  'y':(18,'f4'),
                                  'z':(19,'f4'),
                                  'rvir':(11,'f4'),
                                  'id':(1,'i4')}
                
                        reader = TabularAsciiReader(hf, fields)
                        halos  = reader.read_ascii()
                    except:
                        fields = {'vmax':(66,'f4'),
                                  'mvir':(50,'f4'),
                                  'mvir_now':(10,'f4'),
                                  'mpeak_scale':(61,'f4'),
                                  'upid':(6,'i4'),
                                  'x':(17,'f4'),
                                  'y':(18,'f4'),
                                  'z':(19,'f4'),
                                  'rvir':(11,'f4'),
                                  'id':(1,'i4')}
                
                        reader = TabularAsciiReader(hf, fields)
                        halos  = reader.read_ascii()
                elif self.ctrees_version == 2:
                        fields = {'vmax':(77,'f4'),
                                  'mvir':(61,'f4'),
                                  'mvir_now':(10,'f4'),
                                  'mpeak_scale':(72,'f4'),
                                  'upid':(6,'i4'),
                                  'x':(17,'f4'),
                                  'y':(18,'f4'),
                                  'z':(19,'f4'),
                                  'rvir':(11,'f4'),
                                  'id':(1,'i4')}

                elif self.ctrees_version == 3:
                        fields = {'vmax':(69,'f4'),
                                  'mvir':(55,'f4'),
                                  'mvir_now':(10,'f4'),
                                  'mpeak_scale':(64,'f4'),
                                  'upid':(6,'i4'),
                                  'x':(17,'f4'),
                                  'y':(18,'f4'),
                                  'z':(19,'f4'),
                                  'rvir':(11,'f4'),
                                  'id':(1,'i4')}
                
                        reader = TabularAsciiReader(hf, fields)
                        halos  = reader.read_ascii()
                    


            else:
                #compressed hlists have vmax@vpeak stored as vmax already
                halos = fitsio.read(hf, columns=['vmax', 'mpeak',
                                                 'mpeak_scale',
                                                 'upid', 'id'
                                                 'x','y','z'])

            Delta = self.calc_Delta_vir(self.omegam, a=halos['mpeak_scale'])
            rho_c = self.cos.critical_density0.to('M_sun/km3').value
            G     = const.G.to('km3/(s2*M_sun)').value

            print(halos[0])

            vvir  = (4 / 3 * np.pi * Delta * (rho_c * G ** 3)) ** (1/6) * halos['mvir'] ** (1/3)
            print(vvir)
            out = np.zeros(len(vvir), dtype=odtype)

            z = zs[i]
            lz = lf.genLuminosityFunctionZ(self.lums, z)

            for k in lf.unitmap:
                if lf.unitmap[k] == self.unitmap[k]:
                    continue
                else:
                    conv = getattr(self, '{0}2{1}'.format(lf.unitmap[k], self.unitmap[k]))
                    lz[k] = conv(lz[k])

            for a in alpha:
                for s in scatter:

                    sfname = self.getSHAMFileName(hf, a, s,
                                                  lf.name, ind)

                    oname = '{0}/sham/{1}'.format(self.outdir, sfname)

                    if os.path.isfile(oname):
                        print('{} exists! Skipping this snapshot'.format(oname))
                        continue 

                    proxy = vvir * (halos['vmax'] / vvir) ** a
                    out['PX'] = halos['x']
                    out['PY'] = halos['y']
                    out['PZ'] = halos['z']
                    out['VMAX'] = halos['vmax']
                    out['MPEAK'] = halos['mvir']
                    out['MVIR'] = halos['mvir_now']
                    out['CENTRAL'][halos['upid']==-1] = 1
                    out['CENTRAL'][halos['upid']!=-1] = 0
                    out['AMPROXY'] = proxy
                    out['RVIR'] = halos['rvir']
                    out['UPID'] = halos['upid']
                    out['ID']   = halos['id']


                    out['LUMINOSITY'] = abundanceMatchSnapshot(proxy,
                                                               s,
                                                               lz,
                                                               self.boxsize,
                                                               debug=debug)

                    try:
                        fitsio.write('{0}/sham/{1}'.format(self.outdir, sfname), out)
                    except IOError as e:
                        print('File {} already exists, not writing new one!'.format('{0}/sham/{1}'.format(self.outdir, sfname)))
                

    def getSHAMFiles(self, lf, alpha=0.7, scatter=0.17):
        
        shamfiles = []
        
        for i, hf in enumerate(self.hfiles):
            shamfiles.append('{}/sham/{}'.format(self.outdir, self.getSHAMFileName(hf, alpha, scatter, lf.name, i)))

        self.shamfiles = np.array(shamfiles)

    def rdelMagDist(self, lf, debug=False, 
                      startat=None, endat=None,
                      parallel=False, alpha=0.7, 
                      scatter=0.17):
        """
        Compute rdel-magnitude distribution in SHAMs
        """
        print('rdel')
        self.associateFiles()
        if self.shamfiles is None:
            self.getSHAMFiles(lf, alpha=alpha, scatter=scatter)

        if startat is None:
            startat = 0
        if endat is None:
            endat = len(self.hfiles)

        shamfiles = self.shamfiles[startat:endat]
        rnnfiles = self.rnnfiles[startat:endat]

        if parallel:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            comm.Barrier()

            shamfiles = shamfiles[rank::size]
            rnnfiles  = rnnfiles[rank::size]

            print('Rank {} assigned {} files'.format(rank, len(shamfiles)))
        else:
            rank = 0


        if rank == 0:
            try:
                os.makedirs('{}/rdel'.format(self.outdir))
            except OSError as e:
                warnings.warn('Directory {}/rdel already exists!'.format(self.outdir), Warning)

        for sf, rf in zip(shamfiles, rnnfiles):
            fitSnapshot(sf, rf, '{}/rdel/'.format(self.outdir), self.boxsize, debug=debug)

        #read in all models, fit global model


    def mag2magh(self, mag):

        return mag - 5 * np.log10(self.h)

    def mpc3dex2hmpc3dex(self, phi):

        return phi / self.h ** 3

    def calc_Delta_vir(self,Omega_M0, a=1.0):
        x = 1.0/((1.0/Omega_M0-1.0)*a*a*a + 1.0) - 1.0
        return 18.0*np.pi*np.pi + (82.0-39.0*x)*x

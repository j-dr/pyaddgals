from __future__ import print_function, division
from itertools import product
from scipy.interpolate import interp1d
from scipy.integrate import quad
from copy import copy
import healpy as hp
import numpy as np

from . import luminosityFunction


class Domain(object):
    """Object containing information about the domain a particular task
    is responsible for.
    """

    def __init__(self, cosmo, fmt='BCCLightcone', nside=4, nest=True,
                 rmin=None, rmax=None, nrbins=None, lbox=None, nbox=None,
                 pixlist=None, n_snaps=None, snaplist=None,
                 luminosityFunctionConfig=None, n_octs=2, **kwargs):

        self.fmt = fmt
        self.cosmo = cosmo

        if (fmt == 'BCCLightcone') | (fmt == 'FastPMLightcone'):

            self.subbox = None
            self.n_octs = n_octs

            if pixlist is not None:
                self.pixlist = [int(p) for p in pixlist]
            else:
                self.pixlist = pixlist

            if rmin is None:
                raise(ValueError("rmin must be defined for Lightcone domain"))

            if not lbox:
                raise(ValueError("lbox must be defined for Lightcone domain"))

            if rmax is None:
                raise(ValueError("rmax must be defined for Lightcone domain"))

            if nrbins is None:
                raise(ValueError("nrbins must be defined for Lightcone"))

            if nside is None:
                raise(ValueError("nside must be defined for Lightcone domain"))

            if nest is None:
                raise(ValueError("nest must be defined for Lightcone domain"))

            if luminosityFunctionConfig is None:
                self.numberDensityDomainDecomp = False
            else:
                self.numberDensityDomainDecomp = True
                lf_type = luminosityFunctionConfig['modeltype']

                self.luminosityFunction = getattr(luminosityFunction, lf_type)
                self.luminosityFunction = self.luminosityFunction(cosmo, **luminosityFunctionConfig)

            if isinstance(lbox, str) | isinstance(lbox, (int, float, complex)):
                self.lbox = [int(lbox)]
            else:
                self.lbox = lbox

            if isinstance(rmin, str) | isinstance(rmin, (int, float, complex)):
                self.rmin = [float(rmin)]
            else:
                self.rmin = rmin

            if isinstance(rmax, str) | isinstance(rmax, (int, float, complex)):
                self.rmax = [float(rmax)]
            else:
                self.rmax = rmax

            if isinstance(nrbins, str) | isinstance(nrbins, (int, float, complex)):
                self.nrbins = [int(nrbins)]
            else:
                self.nrbins = nrbins

            self.nside = nside
            self.nest = nest

        if fmt == 'Snapshot':

            self.nside = None
            self.nest = None

            if not lbox:
                raise(ValueError("lbox must be defined for Snapshot domain"))
            if not nbox:
                raise(ValueError("subbox must be defined for Snapshot domain"))
            if not n_snaps:
                raise(ValueError("nsnaps must be defined for Snapshot domain"))

            if isinstance(lbox, str) | isinstance(lbox, (int, float, complex)):
                self.lbox = [float(lbox)]

            if isinstance(n_snaps, str) | isinstance(n_snaps, (int, float, complex)):
                self.n_snaps = [int(n_snaps)]
            else:
                self.n_snaps = n_snaps

            if isinstance(nbox, str) | isinstance(nbox, (int, float, complex)):
                self.nbox = [int(nbox)]
            else:
                self.nbox = nbox

            if snaplist is not None:
                self.snaplist = [int(p) for p in snaplist]
            else:
                self.snaplist = snaplist

    def getRadialLimits(self):
        """Get the radial limits of the lightcone given the redshift limits

        Returns
        -------
        rmin : float
            Minimum radius

        rmax : float
            Maximum radius

        """

        rmin = self.cosmo.rofZ(self.zmin)
        rmax = self.cosmo.rofZ(self.zmax)

        return rmin, rmax

    def octVert(self, octant):
        """Get the coordinates of the vertices enclosing an octant
            on the unit sphere.

        Parameters
        ----------
        octant : int
            The octant number

        Returns
        -------
        vert : list
            A list of three tuples, containing the cartesian coordinates
            of the vertices describing a particular octant.

        """

        if octant == 0:
            vert = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        elif octant == 1:
            vert = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        elif octant == 2:
            vert = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        elif octant == 3:
            vert = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        elif octant == 4:
            vert = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        elif octant == 5:
            vert = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
        elif octant == 6:
            vert = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        elif octant == 7:
            vert = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

        return vert

    def decomp(self, comm, rank, ntasks):

        for i, lb in enumerate(self.lbox):
            if self.fmt == 'BCCLightcone':
                if i > 0:
                    assert(self.rmin[i] == self.rmax[i - 1])

            self.decompSingleBox(comm, rank, ntasks, i)

        self.domains_task.extend(self.domains[self.rank::self.ntasks])
        self.domains_boxnum_task.extend(self.domains_boxnum[self.rank::self.ntasks])
        self.ndomains_task += len(self.domains[self.rank::self.ntasks])

    def decompSingleBox(self, comm, rank, ntasks, boxnum):
        """Perform domain decomposition, creating domain objects for each process. Store information within object.

        Parameters
        ----------
        comm : MPI.Intracomm
            Communicator for tasks
        rank : int
            This task's rank.
        ntasks : int
            Total number of tasks being used.
        i : int
            Index of the box that is currently being decomposed
        Returns
        -------
        None

        """

        self.rank = rank
        self.ntasks = ntasks
        self.comm = comm
        self.domain_counter = 0

        if (self.fmt == 'BCCLightcone') | (self.fmt == 'FastPMLightcone'):

            if boxnum == 0:
                self.rbins = []
                self.domains = []
                self.domains_boxnum = []
                self.domains_task = []
                self.domains_boxnum_task = []
                self.ndomains_task = 0

            # get the pixels that overlap with the octants that we're using
            allpix = []

            # high resolution nside that we'll use to calculate areas of pixels
            # that overlap with the octants in question.
            hrnside = 2048
            # for now only use two octants
            for i in range(self.n_octs):
                vec = self.octVert(i)

                # only want pixels whose centers fall within the octants
                allpix.append(hp.query_polygon(hrnside, vec,
                                               inclusive=False,
                                               nest=self.nest))

            allpix = np.hstack(allpix)
            allpix = np.unique(allpix)

            # get pixels at actual nside we want to use
            if self.nest:
                ud_map = hp.ud_grade(np.arange(12 * self.nside**2), hrnside,
                                     order_in='NESTED', order_out='NESTED')
            else:
                ud_map = hp.ud_grade(np.arange(12 * self.nside**2), hrnside)

            allpix = ud_map[allpix]

            # calculate fraction of area
            pcounts, e = np.histogram(
                allpix, np.arange(12 * self.nside**2 + 1))
            self.allpix = np.unique(allpix)
            self.allpix.sort()

            self.fracarea = pcounts[self.allpix] / 2 ** (2 * np.log2(hrnside /
                                                                     self.nside))

            if self.pixlist is not None:
                idx = np.in1d(self.allpix, self.pixlist)
                assert(idx.any())

                self.allpix = self.allpix[idx]
                self.fracarea = self.fracarea[idx]

            # get radial bins s.t. each bin has equal volume or equal
            # number of gals if given a luminosity function

            if not self.numberDensityDomainDecomp:
                vtot = 4 * np.pi / 3 * (np.array(self.rmax)**3 - np.array(self.rmin)**3)
                v = 4 * np.pi / 3 * (np.array(self.rmax)**3 - np.array(self.rmin)**3) / np.array(self.nrbins)
                vmin = np.hstack([[0], vtot])

                cumvol = np.arange(self.nrbins[boxnum] + 1) * v[boxnum] + vmin[boxnum]
                rbins = (cumvol / (4 * np.pi / 3)) ** (1 / 3)
                rbins[-1] = self.rmax[boxnum]
                rbins[0] = self.rmin[boxnum]
                self.rbins.append(rbins)
            else:
                zmin = self.cosmo.zofR(self.rmin[boxnum])
                zmax = self.cosmo.zofR(self.rmax[boxnum])
                z_bins, nd_cumu = self.luminosityFunction.redshiftCDF(zmin, zmax, self)
                nd_spl = interp1d(z_bins, nd_cumu)
                z_fine = np.linspace(zmin, zmax, 10000)
                nd = nd_spl(z_fine)
                cdf = nd / nd[-1]
                zbins_domain = z_fine[cdf.searchsorted(np.linspace(1 / self.nrbins[boxnum],
                                                                   1 - 1 / self.nrbins[boxnum],
                                                                   self.nrbins[boxnum] - 1))]
                rbins_domain = self.cosmo.rofZ(zbins_domain)
                rbins = np.hstack([[self.rmin[boxnum]], rbins_domain,
                                  [self.rmax[boxnum]]])
                self.rbins.append(rbins)

            # product of pixels and radial bins are all domains
            domains = list(product(np.arange(self.nrbins[boxnum],
                                   dtype=np.int),
                                   self.allpix))
            domains_boxnum = [boxnum] * len(domains)

            self.domains.extend(domains)
            self.domains_boxnum.extend(domains_boxnum)

        if self.fmt == 'Snapshot':
            self.domains = []
            self.domains_boxnum = []
            self.domains_task = []
            self.domains_boxnum_task = []
            self.ndomains_task = 0
            allsnaps = np.arange(self.n_snaps[boxnum])

            if self.snaplist is not None:
                idx = np.in1d(allsnaps, self.snaplist)
                assert(idx.any())

                allsnaps = allsnaps[idx]

            domains = list(product(allsnaps,
                                   np.arange(self.nbox[boxnum]**3)))

            domains_boxnum = [boxnum] * len(domains)

            self.domains.extend(domains)
            self.domains_boxnum.extend(domains_boxnum)

    def yieldDomains(self):

        for i in range(self.ndomains_task):
            d = copy(self)

            if (self.fmt == 'BCCLightcone') | (self.fmt == 'FastPMLightcone'):

                d.boxnum = self.domains_boxnum_task[i]
                d.rbin = self.domains_task[i][0]
                d.pix = self.domains_task[i][1]

                d.rmin = self.rbins[d.boxnum][d.rbin]
                d.rmax = self.rbins[d.boxnum][d.rbin + 1]

                d.zmin = self.cosmo.zofR(d.rmin) - 0.015
                d.zmax = self.cosmo.zofR(d.rmax) + 0.015

                if d.zmin < 0:
                    d.zmin = 1.e-4

                d.rmin = self.cosmo.rofZ(d.zmin)
                d.rmax = self.cosmo.rofZ(d.zmax)

                # volume weighted average radius
                d.rmean = 0.75 * (d.rmax - d.rmin)
                d.zmean = self.cosmo.zofR(d.rmean)

            elif self.fmt == 'Snapshot':
                d.boxnum = self.domains_boxnum_task[i]
                d.snapnum = self.domains_task[i][0]
                d.subbox = self.domains_task[i][1]
                d.pix = d.snapnum
                d.rmin = None
                d.rmax = None
                # will be filled in when files are read
                d.rmean = None
                d.zmean = None

            yield d

    def dummyDomain(self):
        d = copy(self)

        if (self.fmt == 'BCCLightcone') | (self.fmt == 'FastPMLightcone'):

            d.boxnum = self.domains_boxnum_task[0]
            d.rbin = self.domains_task[0][0]
            d.pix = self.domains_task[0][1]

            d.rmin = self.rbins[d.boxnum][d.rbin]
            d.rmax = self.rbins[d.boxnum][d.rbin + 1]

            d.zmin = self.cosmo.zofR(d.rmin) - 0.015
            d.zmax = self.cosmo.zofR(d.rmax) + 0.015

            if d.zmin < 0:
                d.zmin = 1.e-4

            d.rmin = self.cosmo.rofZ(d.zmin)
            d.rmax = self.cosmo.rofZ(d.zmax)

            # volume weighted average radius
            d.rmean = 0.75 * (d.rmax - d.rmin)
            d.zmean = self.cosmo.zofR(d.rmean)

        elif self.fmt == 'Snapshot':
            d.boxnum = self.domains_boxnum_task[0]
            d.snapnum = self.domains_task[0][0]
            d.subbox = self.domains_task[0][1]
            d.pix = d.snapnum
            d.rmin = None
            d.rmax = None
            # will be filled in when files are read
            d.rmean = None
            d.zmean = None

        return d

    def getArea(self):

        if self.fmt is 'Snapshot':
            raise(ValueError('No area associated with snapshot catalogs'))

        if hasattr(self, 'area'):
            return self.pixarea

        elif hasattr(self, 'pix'):
            pidx = self.allpix.searchsorted(self.pix)
            self.pixarea = hp.nside2pixarea(self.nside, degrees=True)
            self.pixarea *= self.fracarea[pidx]
        else:
            return 1

        return self.pixarea

    def getVolume(self):
        """Get the volume of this domain in comoving Mpc/h

        Returns
        -------
        volume : float
            The volume of the simulation domain
        """

        if hasattr(self, 'volume'):
            return self.volume

        if (self.fmt == 'BCCLightcone') | (self.fmt == 'FastPMLightcone'):
            if not hasattr(self, 'pix'):
                raise(ValueError('pix must be defined to calculate volume'))

            pidx = self.allpix.searchsorted(self.pix)
            pixarea = hp.nside2pixarea(self.nside, degrees=True)
            pixarea *= self.fracarea[pidx]

            self.volume = 4 * np.pi * (pixarea / 41253.) * (self.rmax ** 3 -
                                                            self.rmin ** 3) / 3
        elif self.fmt == 'Snapshot':

            self.volume = self.lbox[self.boxnum] ** 3 / self.nbox[self.boxnum] ** 3

        else:
            raise(ValueError('Cannot calculate volumes for fmt {}'.format(self.fmt)))

        return self.volume
    
    def getZeff(self):
        
        integrand = lambda z: z * self.cosmo.dVdz(z)
        zeff, _ = quad(integrand, self.zmin, self.zmax)

        integrand = lambda z: self.cosmo.dVdz(z)
        vol, _ = quad(integrand, self.zmin, self.zmax)        
        self.zeff = zeff / vol
        
        return zeff / vol
        
        

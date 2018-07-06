from __future__ import print_function, division
from itertools import product
from copy import copy
import healpy as hp
import numpy as np


class Domain(object):
    """Object containing information about the domain a particular task
    is responsible for.
    """

    def __init__(self, cosmo, fmt='BCCLightcone', nside=4, nest=True,
                 rmin=None, rmax=None, nrbins=None, lbox=None, nbox=None,
                 pixlist=None):

        self.fmt = fmt
        self.cosmo = cosmo

        if fmt == 'BCCLightcone':

            self.subbox = None

            if pixlist is not None:
                self.pixlist = [int(p) for p in pixlist]
            else:
                self.pixlist = pixlist

            if rmin is None:
                raise(ValueError("rmin must be defined for BCCLightcone domain"))

            if not lbox:
                raise(ValueError("lbox must be defined for BCCLightcone domain"))

            if rmax is None:
                raise(ValueError("rmax must be defined for BCCLightcone domain"))

            if nrbins is None:
                raise(ValueError("nrbins must be defined for BCCLightcone"))

            if nside is None:
                raise(ValueError("nside must be defined for BCCLightcone domain"))

            if nest is None:
                raise(ValueError("nest must be defined for BCCLightcone domain"))

            if isinstance(lbox, str) | isinstance(lbox, (int, float, complex)):
                self.lbox = [int(lbox)]
            if isinstance(rmin, str) | isinstance(rmin, (int, float, complex)):
                self.rmin = [float(rmin)]
            if isinstance(rmax, str) | isinstance(rmax, (int, float, complex)):
                self.rmax = [float(rmax)]
            if isinstance(nrbins, str) | isinstance(nrbins, (int, float, complex)):
                self.nrbins = [int(nrbins)]

            self.nside = nside
            self.nest = nest

        if fmt == 'Snapshot':

            self.nside = None
            self.nest = None

            if not lbox:
                raise(ValueError("lbox must be defined for Snapshot domain"))
            if not nbox:
                raise(ValueError("subbox must be defined for Snapshot domain"))

            if isinstance(lbox, str) | isinstance(lbox, (int, float, complex)):
                self.lbox = [float(lbox)]

            self.nbox = nbox

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
            if i > 0:
                assert(self.rmin[i] == self.rmax[i - 1])

            self.decompSingleBox(comm, rank, ntasks, i)

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

        if boxnum == 0:
            self.domains = []
            self.domains_boxnum = []
            self.domains_task = []
            self.domains_boxnum_task = []
            self.ndomains_task = 0

        if self.fmt == 'BCCLightcone':

            # get the pixels that overlap with the octants that we're using
            allpix = []

            # high resolution nside that we'll use to calculate areas of pixels
            # that overlap with the octants in question.
            hrnside = 2048
            # for now only use two octants
            for i in range(2):
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

            # get radial bins s.t. each bin has equal volume
            dl = self.rmax[boxnum] - self.rmin[boxnum]
            r1 = dl / (self.nrbins[boxnum])**(1 / 3)
            self.rbins = np.arange(self.nrbins[boxnum] + 1)**(1 / 3) * r1 + self.rmin[boxnum]

            # product of pixels and radial bins are all domains
            self.domains.extend(list(product(np.arange(self.nrbins[boxnum],
                                             dtype=np.int),
                                             self.allpix)))
            self.domains_boxnum.extend([boxnum] * len(self.domains))

            # divide up domains
            self.domains_task.extend(self.domains[self.rank::self.ntasks])
            self.domains_boxnum_task.extend(self.domains_boxnum[self.rank::self.ntasks])
            self.ndomains_task += len(self.domains_task)

        if self.fmt == 'Snapshot':
            self.domains.extend(np.arange(self.nbox**3))
            self.domains_boxnum.extend([boxnum] * len(self.domains))
            self.domains_task.extend(self.domains[self.rank::self.ntasks])
            self.domains_boxnum_task.extend(self.domains_boxnum[self.rank::self.ntasks])
            self.ndomains_task += len(self.domains_task)

    def yieldDomains(self):

        for i in range(self.ndomains_task):
            d = copy(self)

            if self.fmt == 'BCCLightcone':

                radial_buffer = 50.
                d.boxnum = self.domains_boxnum_task[i]
                d.rbin = self.domains_task[i][0]
                d.pix = self.domains_task[i][1]

                d.rmin = self.rbins[d.rbin] - radial_buffer
                d.rmax = self.rbins[d.rbin + 1] + radial_buffer

                if d.rmin < 0:
                    d.rmin = 1.

                print(d.rmin, d.rmax)

                d.zmin = self.cosmo.zofR(d.rmin)
                d.zmax = self.cosmo.zofR(d.rmax)
                # volume weighted average radius
                d.rmean = 0.75 * (d.rmax - d.rmin)
                d.zmean = self.cosmo.zofR(d.rmean)

            elif self.fmt == 'Snapshot':
                d.subbox = self.domains_task[i]

            yield d

    def getArea(self):

        if hasattr(self, 'area'):
            return self.pixarea

        else:
            pidx = self.allpix.searchsorted(self.pix)
            self.pixarea = hp.nside2pixarea(self.nside, degrees=True)
            self.pixarea *= self.fracarea[pidx]

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

        if self.fmt == 'BCCLightcone':
            if not hasattr(self, 'pix'):
                raise(ValueError('pix must be defined to calculate volume'))

            pidx = self.allpix.searchsorted(self.pix)
            pixarea = hp.nside2pixarea(self.nside, degrees=True)
            pixarea *= self.fracarea[pidx]

            self.volume = 4 * np.pi * (pixarea / 41253.) * (self.rmax ** 3 -
                                                            self.rmin ** 3) / 3
        elif self.fmt == 'Snapshot':

            self.volume = self.lbox ** 3 / self.nbox ** 3

        else:
            raise(ValueError('Cannot calculate volumes for fmt {}'.format(self.fmt)))

        return self.volume

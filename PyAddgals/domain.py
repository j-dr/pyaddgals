from __future__ import print_function, division
from itertools import product


class Domain(object):
    """Object containing information about the domain a particular task
    is responsible for.
    """


    def __init__(self, fmt='BCCLightcone', nside=4, nest=True,
                 rmin=None, rmax=None, rbins=None, lbox=None, nbox=None):

        self.fmt = fmt

        if fmt == 'BCCLightcone':

            self.subbox = None
            self.lbox = None

            if rmin is None:
                raise(ValueError("rmin must be defined for BCCLightcone domain"))

            if rmax is None:
                raise(ValueError("rmax must be defined for BCCLightcone domain"))

            if nrbins is None:
                raise(ValueError("nrbins must be defined for BCCLightcone"))

            if nside is None:
                raise(ValueError("nside must be defined for BCCLightcone domain"))

            if nest is None:
                raise(ValueError("nest must be defined for BCCLightcone domain"))

            self.rmin = rmin
            self.rmax = rmax
            self.nrbins = nrbins
            self.nside = nside
            self.nest = nest

        if fmt == 'Snapshot':

            self.nside = None
            self.nest = None

            if not lbox:
                raise(ValueError("lBox, must be defined for Snapshot domain"))
            if not nbox:
                raise(ValueError("subbox, must be defined for Snapshot domain"))

            self.lbox = lbox
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

        rmin = self.nbody.cosmo.rofZ(self.domain.zmin)
        rmax = self.nbody.cosmo.rofZ(self.domain.zmax)

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

        if octn==0:
            vert = [[1,0,0], [0,1,0], [0,0,1]]
        elif octn==1:
            vert = [[-1,0,0], [0,1,0], [0,0,1]]
        elif octn==2:
            vert = [[-1,0,0], [0,-1,0], [0,0,1]]
        elif octn==3:
            vert = [[1,0,0], [0,-1,0], [0,0,1]]
        elif octn==4:
            vert = [[1,0,0], [0,1,0], [0,0,-1]]
        elif octn==5:
            vert = [[-1,0,0], [0,1,0], [0,0,-1]]
        elif octn==6:
            vert = [[-1,0,0], [0,-1,0], [0,0,-1]]
        elif octn==7:
            vert = [[1,0,0], [0,-1,0], [0,0,-1]]

        return vert


    def decomp(self, comm, rank, ntasks):
        """Perform domain decomposition, creating domain objects for each process. Store information within object.

        Parameters
        ----------
        comm : MPI.Intracomm
            Communicator for tasks
        rank : int
            This task's rank.
        ntasks : int
            Total number of tasks being used.

        Returns
        -------
        None

        """

        self.rank = rank
        self.ntasks = ntasks
        self.comm = comm
        self.domain_counter = 0

        if self.fmt == 'BCCLightcone':
            #get the pixels that overlap with the octants that we're using
            self.allpix = []

            #for now only use two octants
            for i in range(2):
                self.allpix.extend(self.octVert(i))

            self.allpix = np.unique(self.allpix)

            #get radial bins s.t. each bin has equal volume
            dl = self.rmax - self.rmin
            r1 = dl / (self.nrbins)**(1/3)
            self.rbins = np.arange(self.nrbins+1)**(1/3) * r1 + self.rmin

            #product of pixels and radial bins are all domains
            self.domains = list(product(np.arange(self.nrbins,dtype=np.int),
                                        self.allpix))

            #divide up domains
            self.domains_task = self.domains[self.rank::self.ntasks]
            self.ndomains_task = len(self.domains_task)

        if self.fmt == 'Snapshot':
            self.domains = np.arange(self.nbox**3)
            self.domains_task = self.domains[self.rank::self.ntasks]
            self.ndomains_task = len(self.domains_task)

    def yieldDomains(self):

        for i in range(self.ndomains_task):
            d = copy(self)

            if self.fmt=='BCCLightcone':
                d.rbin = self.domains_task[i][0]
                d.pix  = self.domains_task[i][1]

                d.rmin = self.rbins[d.rbin]
                d.rmax = self.rbins[d.rmax]
                
            elif self.fmt=='Snapshot':
                d.subbox = self.domains_task[i]

            yield d

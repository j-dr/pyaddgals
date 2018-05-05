from __future__ import print_function, division


class Domain(object):
    """Object containing information about the domain a particular task
    is responsible for.
    """

    def __init__(self, fmt='BCCLightcone', subbox=3, nside=4, nest=True,
                 rmin=None, rmax=None, lBox=None):

        self.fmt = fmt

        if fmt == 'BCCLightcone':

            self.subbox = None
            self.lbox = None
            self.nside = nside
            self.nest = nest

            if not rmin:
                raise(ValueError("rmin, must be defined for BCCLightcone domain"))

            if not rmax:
                raise(ValueError("rmax, must be defined for BCCLightcone domain"))

            self.rmin = rmin
            self.rmax = rmax

        if fmt == 'Snapshot':

            self.nside = None
            self.nest = None

            if not lBox:
                raise(ValueError("lBox, must be defined for Snapshot domain"))
            if not subbox:
                raise(ValueError("subbox, must be defined for Snapshot domain"))

            self.lBox = lBox
            self.subbox = subbox

    def getRadialLimits(self):
        """Get the radial limits of the lightcone given the redshift limits

        Returns
        -------
        rmin : float
            Minimum radius

        rmax : float
            Maximum radius

        """

        rmin = ccl.comoving_radial_distance(cosmo, 1 / (1 + self.domain.zmin))
        rmax = ccl.comoving_radial_distance(cosmo, 1 / (1 + self.domain.zmax))

        return rmin, rmax

    def domainDecomp(self, comm, rank, ntasks):
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

        if fmt == 'BCCLightcone':
            pass

        if fmt == 'Snapshot':
            pass

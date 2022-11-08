from __future__ import print_function, division
from pixlc.pixLC import read_radial_bin, nest2peano
from collections import namedtuple
from nbodykit.lab import *
import healpy as hp
import numpy as np
import struct


class ParticleCatalog(object):

    __GadgetHeader_fmt = '6I6dddii6Iiiddddii6Ii'

    GadgetHeader = namedtuple('GadgetHeader',
                              'npart mass time redshift flag_sfr flag_feedback npartTotal flag_cooling num_files BoxSize Omega0 OmegaLambda HubbleParam flag_age flag_metals NallHW flag_entr_ics')

    def __init__(self, nbody, **kwargs):
        """Short summary.

        Parameters
        ----------
        nbody : NBody
            The nbody this particle catalog belongs to. Contains information about how to read data, and the domain decomposition.
        **kwargs : type
            Description of parameter `**kwargs`.

        Returns
        -------
        None

        """

        self.nbody = nbody

    def read(self):

        if self.nbody.domain.fmt == 'BCCLightcone':

            self.readBCCLightcone()

        elif self.nbody.domain.fmt == 'Snapshot':

            self.readSnapshot()
            
        elif self.nbody.domain.fmt == 'FastPMLightcone':
            
            self.readFastPMLightcone()

        else:
            raise(NotImplementedError(
                "Only BCCLightcone reading is currently implemented"))

    def delete(self):
        """Delete particle catalog

        Returns
        -------
        None

        """

        if not hasattr(self, 'catalog'):
            return

        keys = list(self.catalog.keys())

        if len(keys) == 0:
            return

        for k in keys:
            del self.catalog[k]

    def calculateOverdensity(self):

        dens = (len(self.catalog['pos']) *
                self.part_mass / self.nbody.domain.getVolume() /
                self.nbody.f_downsample[self.nbody.domain.boxnum])
        dens_mean = 3. * 100 ** 2 / \
            (8 * np.pi * 4.301e-9) * self.nbody.domain.cosmo.omega_m

        return dens / dens_mean

    def getFilePixels(self, r):
        """Given a healpix cell and radius for a given nside, figure out which
        lightcone pixels we need to read

        Parameters
        ----------
        r : int
            radial bin to read

        Returns
        -------
        pix_file : list
            List of the lightcone file pixels that need to be read
        peano_idx : list
            List of the peano indices of particles that need to be read
        """

        partpath = self.nbody.partpath[self.nbody.boxnum]
        nside = self.nbody.domain.nside
        pix = self.nbody.domain.pix

        header_fmt = ['Np', 'nside_index', 'nside_file', 'box_rmin',
                      'box_rmax', 'void', 'Lbox', 'Mpart', 'Omega_m', 'Omega_l', 'h']

        f = '{}/snapshot_Lightcone_{}_0'.format(partpath, r)
        hdr, idx = read_radial_bin(f)
        hdr = dict(zip(header_fmt, hdr))
        self.part_mass = hdr['Mpart'] * 1e10

        if not self.nbody.domain.nest:
            pix = hp.ring2nest(nside, pix)

        # this assumes that nside < nside_index which should always be true
        idxmap = hp.ud_grade(np.arange(12 * nside**2), hdr['nside_index'],
                             order_in='NESTED', order_out='NESTED')

        # get peano cells corresponding to pix
        # first get nside=nside_index, nest ordered cells corresponding to pix

        peano_idx = nest2peano(np.where(idxmap == pix)[0],
                               int(np.log2(hdr['nside_index'])))

        if nside < hdr['nside_file']:
            udmap = hp.ud_grade(np.arange(12 * nside**2), hdr['nside_file'],
                                order_in='NESTED', order_out='NESTED')
            pix_file, = np.where(udmap == pix)

        elif nside > hdr['nside_file']:
            udmap = hp.ud_grade(np.arange(12 * hdr['nside_file']**2), nside,
                                order_in='NESTED', order_out='NESTED')
            pix_file = [udmap[pix]]

        else:
            pix_file = [pix]

        return pix_file, peano_idx

    def getNpartAll(self):
        """Get number of particles to be read from nbody.

        Returns
        -------
        Npart : int
            Number of particles to read in this task's domain

        """
        Npart = 0

        if self.nbody.domain.fmt == 'BCCLightcone':
            partpath = self.nbody.partpath[self.nbody.boxnum]
            rpmin = int(self.nbody.domain.rmin // 25.)
            rpmax = int(self.nbody.domain.rmax // 25.)

            for r in range(rpmin, rpmax + 1):

                pix_file, peano_idx = self.getFilePixels(r)

                for p in pix_file:

                    f = '{}/snapshot_Lightcone_{}_{}'.format(partpath, r, p)
                    hdr, idx = read_radial_bin(f)

                    Npart += np.sum(idx[peano_idx])
        else:
            raise(ValueError("Only BCCLightcone reading is currently implemented"))

        return Npart

    def readPartialRadialBin(self, filename, peano_inds, read_pos=False,
                             read_vel=False, read_ids=False):
        """
        Read in a radial/hpix cell

        filename -- The name of the file to read, or a file object. If file
                    object, will not be closed upon function return. Instead
                    the pointer will be left at the location of the last
                    data read.
        peano_inds -- Peano indicies to read from the file
        read_xxx -- Whether or not to read xxx
        """

        hdrfmt = 'QIIffQfdddd'
        idxfmt = np.dtype('i8')
        to_read = np.array([read_pos, read_vel, read_ids])
        fmt = [np.dtype(np.float32), np.dtype(np.float32),
               np.dtype(np.uint64)]
        item_per_row = [3, 3, 1]
        data = []
        counter = 0  # keep track of where we are in the file

        opened = False

        if not hasattr(filename, 'read'):
            opened = True
            fp = open(filename, 'rb')
        else:
            fp = filename

        # read the header
        h = list(struct.unpack(hdrfmt,
                               fp.read(struct.calcsize(hdrfmt))))
        counter += struct.calcsize(hdrfmt)

        npart = h[0]
        indexnside = h[1]
        indexnpix = 12 * indexnside**2

        data.append(h)

        # read the peano index
        idx = np.fromstring(fp.read(idxfmt.itemsize * indexnpix), idxfmt)
        counter += idxfmt.itemsize * indexnpix

        data.append(idx)

        # make sure peano_inds are sorted
        peano_inds.sort()

        # figure out where in the file we need to read, and how many parts
        nPeano = len(peano_inds)
        cidx = np.cumsum(idx)
        # cumulative number of particles before each peano ind
        cidx = np.hstack([np.zeros(1), cidx])

        # number of particles in each peano ind to read
        npart_read = (cidx[peano_inds + 1] - cidx[peano_inds]).astype(int)
        # number of particles we need to seek before each peano ind we want to read

        npart_seek = np.hstack([np.array(cidx[peano_inds[0]]),
                                cidx[peano_inds[1:]] -
                                cidx[peano_inds[:-1] + 1]]).astype(int)

        # cumulative number of particles in peano inds we want to read
        npart_read_cum = np.hstack([np.zeros(1),
                                    np.cumsum(npart_read)]).astype(int)

        if to_read.any():

            for i, r in enumerate(to_read):

                if r:
                    d = np.zeros(int(npart_read_cum[-1] * item_per_row[i]))

                    for j, p in enumerate(peano_inds):
                        fp.seek(item_per_row[i] *
                                fmt[i].itemsize * npart_seek[j], 1)

                        d[npart_read_cum[j] * item_per_row[i]:npart_read_cum[j + 1] * item_per_row[i]] = \
                            np.fromstring(
                                fp.read(int(npart_read[j] * item_per_row[i] * fmt[i].itemsize)), fmt[i])

                    fp.seek(
                        int(npart * item_per_row[i] * fmt[i].itemsize) + counter, 0)
                    counter += int(npart * item_per_row[i] * fmt[i].itemsize)
                # fp.seek(
                #         int(item_per_row[i] * fmt[i].itemsize * (npart - npart_read_cum[-1])), 1)
                    data.append(d)
                else:
                    fp.seek(int(npart * item_per_row[i] * fmt[i].itemsize), 1)

                if not to_read[i + 1:].any():
                    break

        if opened:
            fp.close()

        return data, npart_read, npart_seek, npart_read_cum

    def readPartialPartRnn(self, filepath, peano_inds, npart_read, npart_seek,
                           npart_read_cum):
        """
        Read binary output from calcRnn code for particles
        into a numpy array

        """
        fmt = np.dtype(np.float32)

        with open(filepath, 'rb') as fp:
            # read header
            bytes = fp.read(4 * 5)
            head = struct.unpack('iiiii', bytes)
            npart = head[1]

            d = np.zeros(int(npart_read_cum[-1]))

            for j, p in enumerate(peano_inds):
                fp.seek(fmt.itemsize * npart_seek[j], 1)
                d[npart_read_cum[j]:npart_read_cum[j + 1]
                  ] = np.fromstring(fp.read(int(npart_read[j] * fmt.itemsize)), fmt)

            fp.seek(int(fmt.itemsize * (npart - npart_read_cum[-1])), 1)

        return d

    def readPartialHinfo(self, filepath, peano_inds, npart_read, npart_seek,
                         npart_read_cum, npart):
        """
        Read binary output from haloassoc code into numpy array
        into a numpy array

        """
        fmt = np.dtype([('haloid', np.int64),
                        ('rhalo', np.float64),
                        ('mass', np.float64),
                        ('radius', np.float64)])

        with open(filepath, 'rb') as fp:
            # read header
            d = np.zeros(int(npart_read_cum[-1]), dtype=fmt)

            for j, p in enumerate(peano_inds):
                fp.seek(fmt.itemsize * npart_seek[j], 1)
                d[npart_read_cum[j]:npart_read_cum[j + 1]
                  ] = np.fromstring(fp.read(int(npart_read[j] * fmt.itemsize)), fmt)

            fp.seek(int(fmt.itemsize * (npart - npart_read_cum[-1])), 1)

        return d

    def readBCCLightcone(self):
        """Read in particle information

        Returns
        -------
        None

        """

        partpath = self.nbody.partpath[self.nbody.boxnum]
        denspath = self.nbody.denspath[self.nbody.boxnum]
        hinfopath = self.nbody.hinfopath[self.nbody.boxnum]

        rmin = self.nbody.domain.rmin
        rmax = self.nbody.domain.rmax

        rpmin = int(rmin // 25)
        rpmax = int(rmax // 25)

        Npart = self.getNpartAll()

        count = 0
        hidtype = np.dtype([('haloid', np.int64),
                            ('rhalo', np.float64),
                            ('mass', np.float64),
                            ('radius', np.float64)])

        pos = np.zeros((Npart, 3))
        vel = np.zeros((Npart, 3))
        ids = np.zeros(Npart)
        rnn = np.zeros(Npart)
        hinfo = np.zeros(Npart, dtype=hidtype)

        for r in range(rpmin, rpmax + 1):

            pix_file, peano_idx = self.getFilePixels(r)

            for p in pix_file:

                fp = '{}/snapshot_Lightcone_{}_{}'.format(partpath, r, p)
                fr = '{}/rnn_snapshot_Lightcone_{}_{}'.format(denspath, r, p)
                fh = '{}/hinfo_snapshot_Lightcone_{}_{}'.format(
                    hinfopath, r, p)

                (hdr, idx, posi, veli, idsi), npart_read, npart_seek, npart_read_cum = self.readPartialRadialBin(fp, peano_idx,
                                                                                                                 read_pos=True, read_vel=True, read_ids=True)

                rnni = self.readPartialPartRnn(fr, peano_idx, npart_read, npart_seek,
                                               npart_read_cum)

                hinfoi = self.readPartialHinfo(fh, peano_idx, npart_read, npart_seek,
                                               npart_read_cum, np.sum(idx))

                pos[count:count + npart_read_cum[-1]] = posi.reshape(-1, 3)
                vel[count:count + npart_read_cum[-1]] = veli.reshape(-1, 3)
                ids[count:count + npart_read_cum[-1]] = idsi
                rnn[count:count + npart_read_cum[-1]] = rnni
                hinfo[count:count + npart_read_cum[-1]] = hinfoi

                count += npart_read_cum[-1]

        # store everything in a dict for easy access
        self.catalog = {}

        # calculate z from radii
        r = np.sqrt(np.sum(pos**2, axis=1))
        idx = (self.nbody.domain.rmin <= r) & (r < self.nbody.domain.rmax)

        # cut to first two octants
        ra, dec = hp.vec2ang(pos, lonlat=True)
        idx &= (ra <= 180) & (dec >= 0)
        r = r[idx]
        del ra, dec

        self.catalog['z'] = self.nbody.cosmo.zofR(r)
        del r
        self.catalog['pos'] = pos[idx]
        del pos
        self.catalog['vel'] = vel[idx]
        del vel
        self.catalog['id'] = ids[idx]
        del ids
        self.catalog['rnn'] = rnn[idx]
        del rnn
        self.catalog['haloid'] = hinfo['haloid'][idx]
        self.catalog['rhalo'] = hinfo['rhalo'][idx]
        self.catalog['mass'] = hinfo['mass'][idx]
        self.catalog['radius'] = hinfo['radius'][idx]
        del hinfo
        
    def readFastPMLightcone(self):
        """Read in particle information

        Returns
        -------
        None

        """
        pass
        #partpath = self.nbody.partpath[self.nbody.boxnum]

        #rmin = self.nbody.domain.rmin
        #rmax = self.nbody.domain.rmax
        #catalog = BigFileCatalog(self.nbody.halofile[self.nbody.boxnum], dataset="1")

        ## get the part of the catalog for this task
        #pos = catalog['Position'][:]
        #r = np.sqrt(pos**2, axis=1)
        #pix = hp.vec2pix(self.nbody.domain.nside, pos[:,0],
        #                 pos[:,1], pos[:,2],
        #                 nest=self.nbody.domain.nest)
        #idx = (self.nbody.domain.rmin < r) & (r <= self.nbody.domain.rmax)
        #idx = (self.nbody.domain.pix == pix) & idx
        #catalog = catalog[idx]
        #pos = pos[idx]
        #del r, idx
        ## store everything in a dict for easy access
        #self.catalog = {}

        ## calculate z from radii
        #self.catalog['z'] = (1/catalog['Aemit'] - 1).compute()
        #self.catalog['pos'] = pos.compute()
        #self.catalog['vel'] = catalog['Velocity'].compute()
        #self.catalog['id'] = catalog['ID'].compute()


    def readGadgetSnapshot(self, filename, read_pos=True, read_vel=True, read_id=False,
                           read_mass=False, print_header=False, single_type=-1,
                           lgadget=True):
        """
        This function reads the Gadget-2 snapshot file. Taken from Yao-Yuan Mao's
        helpers module.

        Parameters
        ----------
        filename : str
            path to the input file
        read_pos : bool, optional
            Whether to read the positions or not. Default is false.
        read_vel : bool, optional
            Whether to read the velocities or not. Default is false.
        read_id : bool, optional
            Whether to read the particle IDs or not. Default is false.
        read_mass : bool, optional
            Whether to read the masses or not. Default is false.
        print_header : bool, optional
            Whether to print out the header or not. Default is false.
        single_type : int, optional
            Set to -1 (default) to read in all particle types.
            Set to 0--5 to read in only the corresponding particle type.
        lgadget : bool, optional
            Set to True if the particle file comes from l-gadget.
            Default is false.

        Returns
        -------
        ret : tuple
            A tuple of the requested data.
            The first item in the returned tuple is always the header.
            The header is in the GadgetHeader namedtuple format.
        """
        blocks_to_read = (read_pos, read_vel, read_id, read_mass)
        ret = []
        with open(filename, 'rb') as f:
            f.seek(4, 1)
            h = list(struct.unpack(ParticleCatalog.__GadgetHeader_fmt,
                                   f.read(struct.calcsize(ParticleCatalog.__GadgetHeader_fmt))))
            if lgadget:
                h[30] = 0
                h[31] = h[18]
                h[18] = 0
                single_type = 1
            h = tuple(h)
            header = ParticleCatalog.GadgetHeader._make((h[0:6],) + (h[6:12],) +
                                                        h[12:16] + (h[16:22],) +
                                                        h[22:30] + (h[30:36],) +
                                                        h[36:])

            if print_header:
                print(header)
            if not any(blocks_to_read):
                return header
            ret.append(header)
            f.seek(256 - struct.calcsize(ParticleCatalog.__GadgetHeader_fmt), 1)
            f.seek(4, 1)
            #
            mass_npart = [0 if m else n for m,
                          n in zip(header.mass, header.npart)]
            if single_type not in set(range(6)):
                single_type = -1
            #
            for i, b in enumerate(blocks_to_read):
                fmt = np.dtype(np.float32)
                fmt_64 = np.dtype(np.float64)
                item_per_part = 1
                npart = header.npart
                #
                if i < 2:
                    item_per_part = 3
                elif i == 2:
                    fmt = np.dtype(np.uint32)
                    fmt_64 = np.dtype(np.uint64)
                elif i == 3:
                    if sum(mass_npart) == 0:
                        ret.append(np.array([], fmt))
                        break
                    npart = mass_npart
                #
                size_check = struct.unpack('I', f.read(4))[0]
                #
                block_item_size = item_per_part * sum(npart)
                if size_check != block_item_size * fmt.itemsize:
                    fmt = fmt_64
                if size_check != block_item_size * fmt.itemsize:
                    raise ValueError('Invalid block size in file!')
                size_per_part = item_per_part * fmt.itemsize
                #
                if not b:
                    f.seek(sum(npart) * size_per_part, 1)
                else:
                    if single_type > -1:
                        f.seek(sum(npart[:single_type]) * size_per_part, 1)
                        npart_this = npart[single_type]
                    else:
                        npart_this = sum(npart)
                    data = np.fromstring(
                        f.read(npart_this * size_per_part), fmt)
                    if item_per_part > 1:
                        data.shape = (npart_this, item_per_part)
                    ret.append(data)
                    if not any(blocks_to_read[i + 1:]):
                        break
                    if single_type > -1:
                        f.seek(sum(npart[single_type + 1:]) * size_per_part, 1)
                f.seek(4, 1)
        #
        return tuple(ret)

    def readPartRnn(self, filepath):
        """
        Read binary output from calcRnn code for particles
        into a numpy array
        """

        with open(filepath, 'rb') as fp:
            # read header
            bytes = fp.read(4 * 5)
            head = struct.unpack('iiiii', bytes)
            # read in densities
            bytes = fp.read()
            delta = struct.unpack('{0}f'.format(head[1]), bytes[:-4])
            delta = np.array(delta)

        return delta

    def getNPartSnapshot(self):
        snapnum = self.nbody.domain.snapnum
        if snapnum < 10:
            snapnum = '0{}'.format(snapnum)
        else:
            snapnum = '{}'.format(snapnum)

        partpath = '{}.{}'.format(self.nbody.partpath[self.nbody.boxnum].format(snapnum=snapnum), 0)
        hdr = self.readGadgetSnapshot(partpath, read_pos=False, read_vel=False)

        return hdr.npartTotal[1] + hdr.NallHW[1] * 2**32

    def readSnapshot(self):
        """Read particles and densities for snapshot catalog.

        Returns
        -------
        None
        """

        boxnum = self.nbody.boxnum
        snapnum = self.nbody.domain.snapnum
        if snapnum < 10:
            snapnum = '0{}'.format(snapnum)
        else:
            snapnum = '{}'.format(snapnum)

        nbox = self.nbody.domain.nbox[boxnum]

        npart_tot = self.getNPartSnapshot()
        npart_domain_max = int(1.3 * npart_tot // nbox**3)

        pos = np.zeros((npart_domain_max, 3))
        vel = np.zeros((npart_domain_max, 3))
        rnn = np.zeros(npart_domain_max)

        count = 0

        for i in range(self.nbody.n_blocks[boxnum]):
            partpath = '{}.{}'.format(self.nbody.partpath[boxnum].format(snapnum=snapnum), i)
            denspath = '{}.{}'.format(self.nbody.denspath[boxnum].format(snapnum=snapnum), i)

            hdr, posi, veli = self.readGadgetSnapshot(partpath)
            if i == 0:
                self.nbody.domain.zmean = hdr.redshift
                self.nbody.domain.zmin = hdr.redshift
                self.nbody.domain.zmax = hdr.redshift
                print(self.nbody.domain.zmean)
                self.part_mass = hdr.mass[1] * 10**10

            rnni = self.readPartRnn(denspath)

            idx = nbox * posi // self.nbody.domain.lbox[boxnum]
            idx = idx[:, 0] * nbox ** 2 + idx[:, 1] * nbox + idx[:, 2]
            idx = idx == self.nbody.domain.subbox

            posi = posi[idx]
            veli = veli[idx]
            rnni = rnni[idx]

            nparti = len(rnni)

            pos[count:count + nparti, :] = posi
            vel[count:count + nparti, :] = veli
            rnn[count:count + nparti] = rnni

            count += nparti

        self.catalog = {}
        self.catalog['pos'] = pos[:count]
        self.catalog['vel'] = vel[:count]
        self.catalog['rnn'] = rnn[:count]
        self.catalog['z'] = np.zeros_like(self.catalog['rnn']) + self.nbody.domain.zmean
        self.catalog['rhalo'] = np.zeros_like(self.catalog['rnn'])
        self.catalog['radius'] = np.zeros_like(self.catalog['rnn'])
        self.catalog['haloid'] = np.zeros_like(self.catalog['rnn'])
        self.catalog['mass'] = np.zeros_like(self.catalog['rnn'])

from __future__ import print_function, division
from pixlc.pixLC import read_radial_bin, nest2peano
from glob import glob
import healpy as hp
import pyccl as ccl
import numpy as np
import struct


class ParticleCatalog(object):

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
        else:
            raise(NotImplementedError(
                "Only BCCLightcone reading is currently implemented"))

    def delete(self):
        """Delete particle catalog

        Returns
        -------
        None

        """

        keys = list(self.catalog.keys())

        for k in keys:
            del self.catalog[k]

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

        partpath = self.nbody.partpath
        nside = self.nbody.domain.nside
        pix = self.nbody.domain.pix

        header_fmt = ['Np', 'nside_index', 'nside_file', 'box_rmin', 'box_rmax', 'void', 'Lbox', 'Mpart', 'Omega_m', 'Omega_l', 'h']

        f = '{}/snapshot_Lightcone_{}_0'.format(partpath, r)
        hdr, idx = read_radial_bin(f)
        hdr = dict(zip(header_fmt, hdr))

        nside_file = hdr['nside_file']
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
            partpath = self.nbody.partpath
            pix = self.nbody.domain.pix
            nside = self.nbody.domain.nside
            rpmin = int(self.nbody.domain.rmin//25.)
            rpmax = int(self.nbody.domain.rmax//25.)

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
                        int(item_per_row[i] * fmt[i].itemsize * (npart - npart_read_cum[-1])), 1)
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

        partpath = self.nbody.partpath
        denspath = self.nbody.denspath
        hinfopath = self.nbody.hinfopath
        pix = self.nbody.domain.pix
        nside = self.nbody.domain.nside

        rmin, rmax = self.nbody.domain.getRadialLimits()

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
                fh = '{}/hinfo_snapshot_Lightcone_{}_{}'.format(hinfopath, r, p)

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
        r = r[idx]

        self.catalog['pos'] = pos[idx]
        self.catalog['vel'] = vel[idx]
        self.catalog['id'] = ids[idx]
        self.catalog['rnn'] = rnn[idx]
        self.catalog['haloid'] = hinfo['haloid'][idx]
        self.catalog['rhalo'] = hinfo['rhalo'][idx]
        self.catalog['mass'] = hinfo['mass'][idx]
        self.catalog['radius'] = hinfo['radius'][idx]
        self.catalog['z'] = self.nbody.cosmo.zofR(r)

        del r

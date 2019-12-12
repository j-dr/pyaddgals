from glob import glob
import numpy as np
import fitsio
import sys


if __name__ == '__main__':

    inglob = sys.argv[1]
    outfile = sys.argv[2]

    infiles = glob(inglob)
    for i, f in enumerate(infiles):

        g = fitsio.read(f, columns=['ID', 'RA', 'DEC', 'FLUX_G', 'FLUX_R', 'FLUX_I', 'FLUX_Z',
                                    'MAGERR_G', 'MAGERR_R', 'MAGERR_I', 'MAGERR_Z', 'Z', 'Z_COS',
                                    'MAG_I'])

        idx = g['MAG_I'] < 23
        idx = np.random.choice(idx, size=len(idx) // 100)
        g = g[idx]

        train = np.zeros(len(idx), dtype=np.dtype([('ID', '>i8'), ('RA', '>f8'), ('DEC', '>f8'),
                                                   ('FLUX_G', '>f8'), ('FLUX_R', '>f8'), ('FLUX_I', '>f8'),
                                                   ('FLUX_Z', '>f8'), ('MAGERR_G', '>f8'), ('MAGERR_R', '>f8'),
                                                   ('MAGERR_I', '>f8'), ('MAGERR_Z', '>f8'), ('Z', '>f8'),
                                                   ('MAG_I', '>f8'), ('Z_COS', '>f8')]))

        for k in train.dtype.names:
            train[k] = g[k]

        if i == 0:
            out = fitsio.FITS(outfile)
            out.write(train)
        else:
            out = fitsio.FITS(outfile)
            out[-1].append(train)

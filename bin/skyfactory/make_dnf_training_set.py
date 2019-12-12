import h5py as h5
import numpy as np
import fitsio
import sys


if __name__ == '__main__':

    mastercatfile = sys.argv[1]
    outfile = sys.argv[2]

    mcat = h5.File(mastercatfile, 'r')

    training_columns_gold = dict(zip(['ID', 'RA', 'DEC', 'FLUX_G', 'FLUX_R', 'FLUX_I', 'FLUX_Z', 'MAGERR_G', 'MAGERR_R', 'MAGERR_I', 'MAGERR_Z'],
                                     ['coadd_object_id', 'ra', 'dec', 'flux_g', 'flux_r', 'flux_i', 'flux_z', 'magerr_g', 'magerr_r', 'magerr_i', 'magerr_z']))
    training_columns_z = dict(zip(['Z', 'Z_COS'], ['z', 'redshift_cos']))

    mag_i = mcat['catalog/gold/mag_i'][:]
    idx = np.where(mag_i < 23)
    idx = np.random.choice(idx, size=len(idx) // 100)

    train = np.zeros(len(idx), dtype=np.dtype([('ID', '>i8'), ('RA', '>f8'), ('DEC', '>f8'),
                                               ('FLUX_G', '>f8'), ('FLUX_R', '>f8'), ('FLUX_I', '>f8'), ('FLUX_Z', '>f8'), ('MAGERR_G', '>f8'), ('MAGERR_R', '>f8'), ('MAGERR_I', '>f8'), ('MAGERR_Z', '>f8'), ('Z', '>f8'), ('MAG_I', '>f8'), ('Z_COS', '>f8')]))

    for k in training_columns_gold.keys():
        train[k] = mcat['catalog/gold/{}'.format(training_columns_gold[k])][:][idx]

    for k in training_columns_z.keys():
        train[k] = mcat['catalog/bpz/unsheared/{}'.format(training_columns_z[k])][:][idx]

    fitsio.write(outfile, train)

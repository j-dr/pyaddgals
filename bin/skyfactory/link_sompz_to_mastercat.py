import numpy as np
import h5py as h5
import sys


def include_sompz(catfile, sompzfile):

    with h5.File(sompzfile, 'r') as fs:

        som_id = fs['catalog/sompz/unsheared/coadd_object_id'][:]
        cols = list(fs['catalog/sompz/unsheared/'].keys())

        with h5.File(catfile, 'r+') as f:

            f['catalog/sompz/pzdata'] = h5.ExternalLink(sompzfile, 'catalog/sompz/pzdata')

            coadd_id = f['catalog/gold/coadd_object_id'][:]
            total_length = len(coadd_id)

            iidx = np.argsort(coadd_id)
            cat_idx = coadd_id.searchsorted(som_id, sorter=iidx)

            sdtype = np.dtype([(c, fs['catalog/sompz/unsheared/' + c].dtype.name) for c in cols])

            sompzarray = np.zeros((total_length), dtype=sdtype)

            for name in cols:
                print(name)
                sompzarray[name][iidx[cat_idx]] = fs['catalog/sompz/unsheared/' + name][:]
                if name.lower() == 'id':
                    try:
                        f.create_dataset('catalog/sompz/unsheared/coadd_object_id',
                                         maxshape=(total_length,),
                                         shape=(total_length,),
                                         dtype=np.int64, chunks=(1000000,))
                    except Exception as e:
                        print(e)
                        pass
                    f['catalog/sompz/unsheared/coadd_object_id'][:] = sompzarray[name]
                else:
                    try:
                        f.create_dataset('catalog/sompz/unsheared/' + name.lower(),
                                         maxshape=(total_length,),
                                         shape=(total_length,),
                                         dtype=fs['catalog/sompz/unsheared/' + name].dtype,
                                         chunks=(1000000,))
                    except Exception as e:
                        print(e)
                        pass
                    f['catalog/sompz/unsheared/' + name.lower()][:] = sompzarray[name]


if __name__ == '__main__':

    mastercat_file = sys.argv[1]
    sompz_file = sys.argv[2]

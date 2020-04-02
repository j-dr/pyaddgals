import numpy as np
import h5py as h5
import sys

def include_sompz(catfile, sompzfile, bpzfile):

    with h5.File(sompzfile, 'r') as fs:

        som_id = fs['catalog/sompz/unsheared/coadd_object_id'][:]
        cols = list(fs['catalog/sompz/unsheared/'].keys())
        
        print(cols)
        sys.stdout.flush()

        with h5.File(catfile, 'r+') as f:

            f['catalog/sompz/pzdata'] = h5.ExternalLink(sompzfile, 'catalog/sompz/pzdata')
            print('linked file')
            sys.stdout.flush()

            coadd_id = f['catalog/gold/coadd_object_id'][:]
            total_length = len(coadd_id)
            print('read')
            sys.stdout.flush()
            
            iidx = np.argsort(coadd_id)
            print('sorted')
            sys.stdout.flush()            
            cat_idx = coadd_id.searchsorted(som_id, sorter=iidx)

            print('searched')
            sys.stdout.flush()

            sdtype = np.dtype([(c, fs['catalog/sompz/unsheared/' + c].dtype.name) for c in cols])

            sompzarray = np.zeros((total_length), dtype=sdtype)

            for name in cols:
                print(name)
                sys.stdout.flush()
                sompzarray[name][iidx[cat_idx]] = fs['catalog/sompz/unsheared/' + name][:]

                try:
                    f.create_dataset('catalog/sompz/unsheared/' + name.lower(),
                                     maxshape=(total_length,),
                                     shape=(total_length,),
                                     dtype=fs['catalog/sompz/unsheared/' + name].dtype,
                                     chunks=(1000000,))
                except Exception as e:
                    print(e)
                    sys.stdout.flush()
                    pass

                f['catalog/sompz/unsheared/' + name.lower()][:] = sompzarray[name]

            try:
                f['catalog/sompz/unsheared/z'] = h5py.ExternalLink(bpzfile, 'catalog/bpx/unsheared/z')
                f['catalog/sompz/unsheared/redshift_cos'] = h5py.ExternalLink(bpzfile, 'catalog/bpx/unsheared/redshift_cos')
            except:
                pass
                
                
if __name__ == '__main__':

    mastercat_file = sys.argv[1]
    sompz_file = sys.argv[2]
    bpz_file = sys.argv[3]

    include_sompz(mastercat_file, sompz_file, bpz_file)

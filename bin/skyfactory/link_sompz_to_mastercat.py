import numpy as np
import h5py as h5
import sys
import yaml


def include_sompz(catfile, sompzfile, bpzfile, sort=False):

    if not sort:
        with h5.File(catfile) as f:
            f['catalog/sompz'] = h5.ExternalLink(sompzfile, "/catalog/sompz")
            f['catalog/sompz/unsheared/z'] = h5.ExternalLink(bpzfile, "/catalog/bpz/z")
            f['catalog/sompz/unsheared/redshift_cos'] = h5.ExternalLink(bpzfile, "/catalog/bpz/redshift_cos")

        return

    else:
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
                    f['catalog/sompz/unsheared/z'] = h5.ExternalLink(bpzfile, 'catalog/bpx/unsheared/z')
                    f['catalog/sompz/unsheared/redshift_cos'] = h5.ExternalLink(bpzfile, 'catalog/bpx/unsheared/redshift_cos')
                except:
                    pass


def match_shape_noise(filename, mcalfilename, zbins, sigma_e_data):

    with h5.File(filename, 'r') as f:
        with h5.File(mcalfilename, 'r+') as mf:

            size_tot = len(f['catalog/metacal/unsheared/e1'][:])
            idx = f['index/select'][:]
            zmean = f['catalog/sompz/unsheared/bhat'][:][idx]
            e1 = f['catalog/metacal/unsheared/e1'][:][idx]
            e2 = f['catalog/metacal/unsheared/e2'][:][idx]

            try:
                mf.create_dataset('catalog/unsheared/metacal/e1_matched_se', maxshape=(
                    size_tot,), shape=(size_tot,), dtype=e1.dtype, chunks=(1000000,))
                mf.create_dataset('catalog/unsheared/metacal/e2_matched_se', maxshape=(
                    size_tot,), shape=(size_tot,), dtype=e1.dtype, chunks=(1000000,))
            except:
                del mf['catalog/unsheared/metacal/e1_matched_se'], mf['catalog/unsheared/metacal/e2_matched_se']

                mf.create_dataset('catalog/unsheared/metacal/e1_matched_se', maxshape=(
                    size_tot,), shape=(size_tot,), dtype=e1.dtype, chunks=(1000000,))
                mf.create_dataset('catalog/unsheared/metacal/e2_matched_se', maxshape=(
                    size_tot,), shape=(size_tot,), dtype=e1.dtype, chunks=(1000000,))

            e1_sn = np.zeros(len(f['catalog/metacal/unsheared/e1']))
            e2_sn = np.zeros(len(f['catalog/metacal/unsheared/e1']))

            for i in range(len(zbins) - 1):
                idxi = (zbins[i] < zmean) & (zmean < zbins[i + 1])
                sigma_e = np.std(e1[idxi])
                ds = np.sqrt((sigma_e_data[i]**2 - sigma_e**2))

                e1_sn[idx[idxi]] = e1[idxi] + ds * \
                    np.random.randn(np.sum(idxi))
                e2_sn[idx[idxi]] = e2[idxi] + ds * \
                    np.random.randn(np.sum(idxi))

                print(e1_sn[idx[idxi]])

            mf['catalog/unsheared/metacal/e1_matched_se'][:] = e1_sn
            mf['catalog/unsheared/metacal/e2_matched_se'][:] = e2_sn


if __name__ == '__main__':

    cfgfile = sys.argv[1]

    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp)

    mastercat_file = cfg['outfile']
    sompz_file = cfg['sompzfile']
    bpz_file = cfg['bpzfile']
    mcalfilename = cfg['mcalfile']
    zbins = cfg['zbins']
    sigma_e_data = cfg['sigma_e_data']

    include_sompz(mastercat_file, sompz_file, bpz_file)
    match_shape_noise(mastercat_file, mcalfilename, zbins, sigma_e_data)

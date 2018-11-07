#!/usr/bin/env python
from __future__ import print_function, division
from glob import glob
from mpi4py import MPI
import numpy as np
import os
import sys
import yaml
import subprocess

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cfgfile = sys.argv[1]
    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp)

    catfiles = np.array(glob(cfg['FilePath']))
    catfiles = catfiles[rank::size]

    if rank==0:
        try:
            os.mkdir(cfg['OPath'])
        except:
            pass

    for alg in cfg['Algorithms']:
        a = cfg['Algorithms'][alg]
        if alg=='BPZ':
            for f in catfiles:
                print(f)
                fs = f.split('/')
                fss = fs[-1].split('.')
                fbase = '.'.join(fss[:-1])
                opath = "{}/{}.BPZ.fits".format(cfg['OPath'], fbase)
                if os.path.exists(opath):
                    continue

                subprocess.call(['python', "{0}/redshift-wg/redshift_codes/photoz_codes/bpzv1/bpzv1.py".format(cfg['ExecPath']), a['CfgFile'], f])

def main_submany():

    cfgfile = sys.argv[1]
    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp)

    catfiles = np.array(glob(cfg['FilePath']))

    try:
        os.mkdir(cfg['OPath'])
    except:
        pass

    for alg in cfg['Algorithms']:
        a = cfg['Algorithms'][alg]
        if alg=='BPZ':
            for f in catfiles:
#                print(f)
                fs = f.split('/')
                fss = fs[-1].split('.')
                fbase = '.'.join(fss[:-1])
                opath = "{}/{}.BPZ.fits".format(cfg['OPath'], fbase)
                if os.path.exists(opath):
                    continue

                name = '.'.join(f.split('/')[-1].split('.')[:-1])

                print(opath)
                subprocess.call(['bsub', '-q', 'kipac-ibq', '-W', '24:00', '-n', '1', '-R', 'rusage[mem=8192]', '-oo', 'logs/pz_{}.oe'.format(name), '-J', '{}'.format(name),
                                 'python', "{0}/photoz-wg/redshift_codes/photoz_codes/bpzv1/bpzv1.py".format(cfg['ExecPath']), a['CfgFile'], f])


if __name__ == '__main__':

    if len(sys.argv)==2:
        main()
    else:
        main_submany()

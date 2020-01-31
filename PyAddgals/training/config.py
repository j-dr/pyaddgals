from __future__ import print_function, division
from glob import glob
import numpy as np
import yaml

from .simulation import Simulation
from .model      import Model
from .luminosityfunction import LuminosityFunction, DSGLuminosityFunction, BernardiLuminosityFunction, ReddickLuminosityFunction, BBGSLuminosityFunction


def readCfg(filename):

    with open(filename, 'r') as fp:
        cfg = yaml.load(fp)

    return cfg

def setLF(cfg):
    
    if cfg['LuminosityFunction']['type'] == 'DSG':

        lf = DSGLuminosityFunction()
    
    elif cfg['LuminosityFunction']['type'] == 'Bernardi':
        
        lf = BernardiLuminosityFunction(cfg['LuminosityFunction']['Q'])

    elif cfg['LuminosityFunction']['type'] == 'Reddick':
        
        lf = ReddickLuminosityFunction(cfg['LuminosityFunction']['Q'])

    elif cfg['LuminosityFunction']['type'] == 'BBGS':
        
        lf = BBGSLuminosityFunction(cfg['LuminosityFunction']['Q'], cfg['LuminosityFunction']['P'])

    return lf
        

def parseConfig(cfg):

    simcfg = cfg['Simulation']
    sims = []

    for i, s in enumerate(simcfg['hlistbase']):
        hlists = glob('{0}/hlist*_0[0-9]*'.format(s))
        rnn    = glob('{0}/snapdir*/rnn*[0-9]'.format(simcfg['rnnbase'][i]))
        snapdirs  = glob('{0}/snapdir*/'.format(simcfg['snapbase'][i]))

        print(hlists)
        print(rnn)
        print(snapdirs)

        ctrees_version = simcfg.pop('ctrees_version', 1)
        
        #snaptimes should always be provided in order that snapshots
        #were output in (in order of increasing time)
        if 'snaptimes' in simcfg:
            a = np.loadtxt(simcfg['snaptimes'][i])
            sa = np.loadtxt(simcfg['snaptimes'][i], dtype=str)
            zs = 1/a[:,1] - 1.
            sims.append(Simulation(simcfg['name'][i],
                                   simcfg['boxsize'][i],
                                   snapdirs,
                                   hlists,
                                   rnn,
                                   simcfg['outdir'],
                                   simcfg['h'][i],
                                   simcfg['omegam'][i],
                                   zs=zs,
                                   compressed_hlist=simcfg['compressed_hlist'],
                                   strscale=sa[:,1],
                                   snapnums=a[:,0],
                                   ctrees_version=ctrees_version))

        else:
            sims.append(Simulation(simcfg['name'][i],
                                   simcfg['boxsize'][i],
                                   snapdirs,
                                   hlists,
                                   rnn,
                                   simcfg['outdir'],
                                   simcfg['h'][i],
                                   simcfg['omegam'][i],
                                   zmin=simcfg['zmin'][i],
                                   zmax=simcfg['zmax'][i],
                                   nz=simcfg['nz'][i],
                                   compressed_hlist=simcfg['compressed_hlist'],
                                   ctrees_version=ctrees_version))


    lf = setLF(cfg)
    m  = Model(sims)

    return sims, lf, m

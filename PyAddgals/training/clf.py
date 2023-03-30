from halotools.empirical_models import NFWPhaseSpace, TrivialPhaseSpace, HodModelFactory
from scipy.optimize import minimize
from mpi4py import MPI
from glob import glob
import numpy as np
import fitsio
import sys

from ..CLF import CLFCens, CLFSats


DefaultCLFConfig =  {
      'CenOccParams': {'log_L_0': 9.557132570335106,
                      'log_M_1': 9.677826309588625,
                      'gamma_1': 85.26855242847066,
                      'gamma_2': 0.2221830261551097,
                      'sigma': 0.16923635878023224,
                      'M_0_sigma': 13798870703166.668,
                      'alpha_sigma': -0.10078033073027963},
      'SatOccParams': {'a_1': 1.204533927448177,
                        'a_2': 0.3270944122861654,
                        'log_M_2': 5.2617613662959775,
                        'b_0': 0.03775141313660356,
                        'b_1': 0.7923078487814424,
                        'b_2': -0.030878419266771373,
                        'delta_1': -1.0557363475245707,
                        'delta_2': 0.21909775439800422,
                        'log_L_0': 9.557132570335106,
                        'log_M_1': 9.677826309588625,
                        'gamma_1': 85.26855242847066,
                        'gamma_2': 0.2221830261551097,
                        'M_0_sigma': 13798870703166.668,
                        'alpha_sigma': -0.10078033073027963,
                        'alpha_12': 1.1637528173629659,
                        'alpha_22': -4.022577531406281,
                        'log_M_22': 13.0,
                        'phi_2': -0.9422093297426509}
}


def set_model_params(
        cens_occ_model,
        sats_occ_model,
        cens_prof_model,
        sats_prof_model,
        cen_clf_params,
        sat_clf_params,
        cen_prof_params,
        sat_prof_params
    ):
        
        for k in cen_clf_params:
            cens_occ_model.param_dict[k] = cen_clf_params[k]
            
        for k in sat_clf_params:
            sats_occ_model.param_dict[k] = sat_clf_params[k]            
            
        for k in cen_prof_params:
            cens_prof_model.param_dict[k] = cen_prof_params[k]
            
        for k in sat_prof_params:
            sats_prof_model.param_dict[k] = sat_prof_params[k]              
            
        model_instance = HodModelFactory(
            centrals_occupation=cens_occ_model,
            centrals_profile=cens_prof_model,
            satellites_occupation=sats_occ_model,
            satellites_profile=sats_prof_model,
        )

        return model_instance         

def setup_model(cen_occ_class, sat_occ_class, cen_clf_params, sat_clf_params, cen_prof_params, sat_prof_params, zeff, threshold=-14):
    threshold = (-0.4 * threshold - 4.76)

    cens_occ_model = cen_occ_class(
        prim_haloprop_key="halo_mvir", redshift=zeff,
        threshold=threshold
    )

    sats_occ_model = sat_occ_class(
        prim_haloprop_key="halo_mvir", redshift=zeff,
        threshold=threshold
    )

    cens_prof_model = TrivialPhaseSpace(
        prim_haloprop_key="halo_mvir", redshift=zeff
    )

    sats_prof_model = NFWPhaseSpace(
        prim_haloprop_key="halo_mvir",
        redshift=zeff,
        conc_mass_model="dutton_maccio14",
    )        

    model_instance = set_model_params(cens_occ_model,
                                       sats_occ_model,
                                       cens_prof_model,
                                       sats_prof_model,
                                       cen_clf_params,
                                       sat_clf_params,
                                       cen_prof_params,
                                       sat_prof_params
                                     )

    return model_instance


def loglike(x, clf_c=None, clf_s=None, var_clf_c=None, var_clf_s=None, 
            CLFModelConfig=None, z=None, nmbins=None,
            lmean=None, mmean=None):
    sat_par_idx = np.arange(18)
    cen_par_idx=[8,9,10,11,18,12,13]

    names_sat = ['a_1', 'a_2', 'log_M_2', 'b_0', 'b_1', 'b_2',
                 'delta_1', 'delta_2','log_L_0', 'log_M_1',
                 'gamma_1', 'gamma_2', 'M_0_sigma', 'alpha_sigma'
                 'alpha_12', 'alpha_22', 'log_M_22', 'phi_2']
    pars_sat = dict(zip(names_sat, x[sat_par_idx]))
                    
    names_cen = ['log_L_0', 'log_M_1','gamma_1', 'gamma_2', 'sigma', 'M_0_sigma', 'alpha_sigma']
    pars_cen = dict(zip(names_cen, x[cen_par_idx]))   
    
    CLFModelConfig['SatOccParams'].update(pars_sat)
    CLFModelConfig['CenOccParams'].update(pars_cen)
    
    clf_model = setup_model(CLFCens, CLFSats,
                            CLFModelConfig['CenOccParams'], CLFModelConfig['SatOccParams'],
                            CLFModelConfig['CenProfParams'], CLFModelConfig['SatProfParams'], z)
    cen_occ_model = clf_model.model_dictionary['centrals_occupation']
    sat_occ_model = clf_model.model_dictionary['satellites_occupation']    
    
    clf_c_mod = np.zeros_like(clf_c)
    clf_s_mod = np.zeros_like(clf_s)
    
    for i in range(nmbins):
        clf_c_mod[:,i] = cen_occ_model.clf(prim_galprop=10**(-0.4 * (lmean - 4.76)), prim_haloprop=mmean[i])
        clf_s_mod[:,i] = sat_occ_model.clf(prim_galprop=10**(-0.4 * (lmean - 4.76)), prim_haloprop=mmean[i])
        
    loss = np.nansum(((clf_c - clf_c_mod)**2/var_clf_c)[var_clf_c>np.percentile(var_clf_c[var_clf_c>0],5)]) + np.nansum(((clf_s - clf_s_mod)**2/var_clf_s)[var_clf_s>np.percentile(var_clf_s[var_clf_s>0],5)])

    return loss

def compute_clf(px, py, pz, luminosity, mhalo, central, upid, hid, mbins, lbins, njk_per_dim=8, boxsize=400):
    
    xidx = np.digitize(px, np.linspace(0, boxsize, njk_per_dim+1))
    yidx = np.digitize(py, np.linspace(0, boxsize, njk_per_dim+1))
    zidx = np.digitize(pz, np.linspace(0, boxsize, njk_per_dim+1))
    
    jknum = xidx * njk_per_dim**2 + yidx * njk_per_dim + zidx
    del xidx, yidx, zidx
    
    njk_tot = njk_per_dim**3
    clf_s_jk = np.zeros((njk_tot, len(lbins)-1, len(mbins)-1))
    clf_c_jk = np.zeros((njk_tot, len(lbins)-1, len(mbins)-1))
    
    mhost = np.copy(mhalo)
    cidx = (central == 1)
    hidx = hid[cidx].searchsorted(upid[~cidx])
    mhost[~cidx] = mhost[cidx][hidx]
    
    for i in range(njk_tot):
        jkidx = jknum != i

        dl = lbins[1:] - lbins[:-1]

        counts_c, e0, e1 = np.histogram2d(luminosity[cidx&jkidx], mhost[cidx&jkidx], [lbins, mbins])
        counts_s, e0, e1 = np.histogram2d(luminosity[(~cidx)&jkidx], mhost[(~cidx)&jkidx], [lbins, mbins])
        nm, e1 = np.histogram(mhalo[cidx], mbins)

        clf_c_jk[i,...] = counts_c / dl[:,np.newaxis] / nm[np.newaxis,:]
        clf_s_jk[i,...] = counts_s / dl[:,np.newaxis] / nm[np.newaxis,:]
        
    clf_c = np.mean(clf_c_jk, axis=0)
    clf_s = np.mean(clf_s_jk, axis=0)
    
    var_clf_c = (njk_tot -1) / njk_tot * np.sum((clf_c_jk - clf_c[np.newaxis,...])**2, axis=0)
    var_clf_s = (njk_tot -1) / njk_tot * np.sum((clf_s_jk - clf_s[np.newaxis,...])**2, axis=0)

    return clf_c, clf_s, var_clf_c, var_clf_s, clf_c_jk, clf_s_jk

def fitSnapshot(shamfile, debug=False):
    
    print('Fitting {}'.format(shamfile), flush=True)

    smname = shamfile.split('/')[-1].split('.')
    smname = '.'.join(smname[:-1])
    scale  = float(smname.split('_')[-1].split('.list')[0])
    z      = 1/scale - 1

    sham = fitsio.read(shamfile, columns=['LUMINOSITY', 'MVIR', 'PX', 'PY', 'PZ', 'CENTRAL', 'ID', 'UPID'])
    mag  = sham['LUMINOSITY']

    #set up bins
    nmbins = 12
    nlbins = 15
    mbins = np.logspace(11,15,nmbins+1)
    lbins = np.linspace(-24,-14,nlbins+1)
    mmean = (mbins[1:] + mbins[:-1]) / 2
    lmean = (lbins[1:] + lbins[:-1]) / 2

    #measure lcen of M
    clf_c, clf_s, var_clf_c, var_clf_s, clf_c_jk, clf_s_jk = compute_clf(sham['PX'], sham['PY'], sham['PZ'], sham['LUMINOSITY'], sham['MVIR'], 
                                                                         sham['CENTRAL'], sham['UPID'], sham['ID'], mbins, lbins, njk_per_dim=5)

    #fit models to individual density distributions
    opt = minimize(loglike, args=(clf_c, clf_s, var_clf_c, var_clf_s,
                                  DefaultCLFConfig, z, nmbins,
                                  lmean, mmean), method='BFGS')

    return opt['x'], opt['hess']


if __name__ == '__main__':
    
    shamglob = sys.argv[1]
    outdir = sys.argv[2]
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    shamfiles = glob()
    a = np.array([float(s.split('_')[-1].split('.')[0]) for s in shamfiles])
    idx = a.argsort()
    a = a[idx]
    shamfiles = shamfiles[idx]
    shamfiles_this = shamfiles[rank::size]
    
    pars = set(DefaultCLFConfig['CenOccParams']) + set(DefaultCLFConfig['SatOccParams'])
    
    x_all = np.zeros((len(shamfiles, len(pars))))
    xe_all = np.zeros((len(shamfiles, len(pars), len(pars))))
    
    for i, shamfile in shamfiles_this:
        x_all[rank+i*size], xe_all[rank+i*size] = fitSnapshot(shamfile)
        
    comm.Reduce(x_all, x_all)
    comm.Reduce(xe_all, xe_all)
    
    if rank==0:
        fitsio.write(f'{outdir}/sham_clf_params.fits', x_all, clobber=True)
        fitsio.write(f'{outdir}/sham_clf_params.fits', xe_all)
        fitsio.write(f'{outdir}/sham_clf_params.fits', 1/a - 1)

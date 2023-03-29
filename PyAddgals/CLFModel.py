from __future__ import print_function, division
from halotools.empirical_models import Cacciato09Cens, Cacciato09Sats, NFWPhaseSpace, TrivialPhaseSpace, HodModelFactory
from halotools.sim_manager import UserSuppliedHaloCatalog
from scipy.special import erfc
from time import time
import numpy as np
import sys

from .galaxyModel import GalaxyModel
from .colorModel import ColorModel
from . import luminosityFunction
from . import shape

x_gauss, w_gauss = np.polynomial.legendre.leggauss(10)

x_gauss, w_gauss = np.polynomial.legendre.leggauss(10)

class CLFCens(Cacciato09Cens):
    
    def get_published_parameters(self):
        param_dict = super().get_published_parameters()
        param_dict['M_0_sigma'] = 13
        param_dict['alpha_sigma'] = 0.1

        return param_dict
    
    def clf(self, prim_galprop=1e11, prim_haloprop=1e12):
        gamma_1 = self.param_dict["gamma_1"]
        gamma_2 = self.param_dict["gamma_2"]
        mass_c = 10 ** self.param_dict["log_M_1"]
        prim_galprop_c = 10 ** self.param_dict["log_L_0"]

        r = prim_haloprop / mass_c
        med_prim_galprop = (
            prim_galprop_c * (r / (1 + r)) ** gamma_1 * (1 + r) ** gamma_2
        )
        
        sigma = self.param_dict['sigma'] * (prim_haloprop/self.param_dict['M_0_sigma'])**(self.param_dict['alpha_sigma'])
#        print(sigma)

        return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
            -((np.log10(prim_galprop / med_prim_galprop)) ** 2)
            / (2.0 * sigma ** 2)
        )    
    
    def median_prim_galprop(self, **kwargs):
        mass = kwargs["prim_haloprop"]

        gamma_1 = self.param_dict["gamma_1"]
        gamma_2 = self.param_dict["gamma_2"]
        mass_c = 10 ** self.param_dict["log_M_1"]
        prim_galprop_c = 10 ** self.param_dict["log_L_0"]

        r = mass / mass_c    
        
        return prim_galprop_c * (r / (1 + r)) ** gamma_1 * (1 + r) ** gamma_2


    def mean_occupation(self, prim_haloprop=1e12, prim_galprop_min=1e11,
                        prim_galprop_max=1e12, **kwargs):

        a = np.log10(prim_galprop_min)
        b = np.log10(prim_galprop_max)

        if np.abs(b - a) > 0.305:
            n_1 = self.mean_occupation(
                prim_haloprop=prim_haloprop, prim_galprop_min=10**a,
                prim_galprop_max=10**((a + b) / 2), **kwargs)
            n_2 = self.mean_occupation(
                prim_haloprop=prim_haloprop, prim_galprop_max=10**b,
                prim_galprop_min=10**((a + b) / 2), **kwargs)
            return n_1 + n_2

        for i in range(len(x_gauss)):
            log_prim_galprop_gauss = (b - a) / 2 * x_gauss[i] + (a + b) / 2
            clf_gauss = self.clf(prim_haloprop=prim_haloprop,
                                 prim_galprop=10**log_prim_galprop_gauss)
            dn = (b - a) / 2 * w_gauss[i] * clf_gauss

            if i == 0:
                n = dn
            else:
                n += dn

        return n


class CLFSats(Cacciato09Sats):

    def __init__(self, threshold=10.0, prim_haloprop_key='halo_mvir',
                 prim_galprop_key='luminosity', **kwargs):
        super().__init__(
            threshold=threshold, prim_haloprop_key=prim_haloprop_key,
            prim_galprop_key=prim_galprop_key, **kwargs)
        self.central_occupation_model = CLFCens(
            threshold=threshold, prim_haloprop_key=prim_haloprop_key,
            prim_galprop_key=prim_galprop_key, **kwargs)

    def get_default_parameters(self):
        param_dict = super().get_default_parameters()
        param_dict['M_0_sigma'] = 13
        param_dict['alpha_sigma'] = 0.1
        param_dict['alpha_21'] = 0.5
        param_dict['alpha_22'] = 0.5
        param_dict['phi_2'] = 1
        param_dict["log_M_22"] = 14

        return param_dict

    def clf(self, prim_galprop=1e10, prim_haloprop=1e12, **kwargs):
        prim_galprop = np.atleast_1d(prim_galprop)
        prim_haloprop = np.atleast_1d(prim_haloprop)

        try:
            assert (
                (len(prim_haloprop) == 1)
                or (len(prim_galprop) == 1)
                or (len(prim_haloprop) == (len(prim_galprop)))
            )
        except AssertionError:
            msg = (
                "If both ``prim_galprop`` and ``prim_haloprop`` are arrays"
                " with multiple elements, they must have the same length.\n"
            )
            raise ValueError(msg)

        b_0 = self.param_dict["b_0"]
        b_1 = self.param_dict["b_1"]
        b_2 = self.param_dict["b_2"]
        log_prim_haloprop = np.log10(prim_haloprop)

        phi_sat = 10 ** (
            b_0
            + b_1 * (log_prim_haloprop - 12.0)
            + b_2 * (log_prim_haloprop - 12.0) ** 2
        )

        a_1 = self.param_dict["a_1"]
        a_2 = self.param_dict["a_2"]
        log_m_2 = self.param_dict["log_M_2"]
        
        a_12 = self.param_dict["alpha_12"]
        a_22 = self.param_dict["alpha_22"]
        log_m_22 = self.param_dict["log_M_22"]
        

        alpha_sat_1 = -2.0 + a_1 * (
            1.0 - 2.0 / np.pi * np.arctan(a_2 * (np.log10(prim_haloprop) - log_m_2))
        )
        
        alpha_sat_2 = -2.0 + a_12 * (
            1.0 - 2.0 / np.pi * np.arctan(a_2 * (np.log10(prim_haloprop) - log_m_2))
        )

        for key, value in self.param_dict.items():
            if key in self.central_occupation_model.param_dict:
                self.central_occupation_model.param_dict[key] = value

        med_prim_galprop = self.central_occupation_model.median_prim_galprop(
            prim_haloprop=prim_haloprop
        )
        prim_galprop_cut = med_prim_galprop * 0.562
        delta = 10 ** (
            self.param_dict["delta_1"]
            + self.param_dict["delta_2"] * (np.log10(prim_haloprop) - 12)
        )

        phi = (
            phi_sat
            * ((prim_galprop / prim_galprop_cut) ** (alpha_sat_1 + 1) + 
               self.param_dict['phi_2']*(prim_galprop / prim_galprop_cut) ** (alpha_sat_2 + 1))
            * np.exp(-delta * (prim_galprop / prim_galprop_cut) ** 2)
            * np.log(10)
        )

        return phi
    
    def mean_occupation(
        self, prim_haloprop=1e12, prim_galprop_min=1e11, prim_galprop_max=1e12, **kwargs
    ):
        a = np.log10(prim_galprop_min)
        b = np.log10(prim_galprop_max)
        if np.abs(b - a) > 0.305:
            n_1 = self.mean_occupation(
                prim_haloprop=prim_haloprop,
                prim_galprop_min=10**a,
                prim_galprop_max=10 ** ((a + b) / 2),
                **kwargs
            )
            n_2 = self.mean_occupation(
                prim_haloprop=prim_haloprop,
                prim_galprop_max=10**b,
                prim_galprop_min=10 ** ((a + b) / 2),
                **kwargs
            )
            return n_1 + n_2
        for i in range(len(x_gauss)):
            log_prim_galprop_gauss = (b - a) / 2 * x_gauss[i] + (a + b) / 2
            clf_gauss = self.clf(
                prim_haloprop=prim_haloprop, prim_galprop=10**log_prim_galprop_gauss
            )
            dn = (b - a) / 2 * w_gauss[i] * clf_gauss
            if i == 0:
                n = dn
            else:
                n += dn


class CLFModel(GalaxyModel):

    def __init__(self, nbody, 
                 CLFModelConfig=None,
                 colorModelConfig=None,
                 shapeModelConfig=None):


        self.nbody = nbody
        self.zeff = self.nbody.domain.getZeff()
        self.delete_after_assignment = False

        if CLFModelConfig is None:
            raise(ValueError('CLF model must define luminosityFunctionConfig'))

        if colorModelConfig is None:
            raise(ValueError('CLF model must define colorModelConfig'))
        
        self.magmin = CLFModelConfig['magmin']
        self.cen_clf_params = CLFModelConfig['CenOccParams']
        self.sat_clf_params = CLFModelConfig['SatOccParams']
        self.cen_prof_params = CLFModelConfig['CenProfParams']
        self.sat_prof_params = CLFModelConfig['SatProfParams']
        self.clf_type = CLFModelConfig['CLFModelType']
        
        colorModelConfig['derived_quantities'] = True
        
        self.luminosityFunction = getattr(luminosityFunction, 'DSGLuminosityFunction')
        self.luminosityFunction = self.luminosityFunction(
            nbody.cosmo, magmin=self.magmin)

        self.colorModel = ColorModel(self.nbody, **colorModelConfig)
        self.c = 3e5

        if shapeModelConfig is None:
            self.shapeModel = None
        else:
            shape_type = shapeModelConfig['modeltype']

            self.shapeModel = getattr(shape, shape_type)
            self.shapeModel = self.shapeModel(nbody.cosmo, **shapeModelConfig)
            
        if self.clf_type == 'Cacciato09':
            self.clf_model_instance = self.setup_cacciato09_model()
            
            
    def set_model_params(
        self,
        cens_occ_model,
        sats_occ_model,
        cens_prof_model,
        sats_prof_model,
    ):
        
        for k in self.cen_clf_params:
            cens_occ_model.param_dict[k] = self.cen_clf_params[k]
            
        for k in self.sat_clf_params:
            sats_occ_model.param_dict[k] = self.sat_clf_params[k]            
            
        for k in self.cen_prof_params:
            cens_prof_model.param_dict[k] = self.cen_prof_params[k]
            
        for k in self.sat_prof_params:
            sats_prof_model.param_dict[k] = self.sat_prof_params[k]              
            
        model_instance = HodModelFactory(
            centrals_occupation=cens_occ_model,
            centrals_profile=cens_prof_model,
            satellites_occupation=sats_occ_model,
            satellites_profile=sats_prof_model,
        )

        return model_instance            
            
        
    def setup_cacciato09_model(self):
        print('setting up cacciato', flush=True)
        threshold = (-0.4 * (self.luminosityFunction.m_min_of_z(self.zeff) - 4.76))
        print('threshold: {}'.format(threshold), flush=True)
        cens_occ_model = Cacciato09Cens(
            prim_haloprop_key="halo_mvir", redshift=self.zeff,
            threshold=threshold
        )

        sats_occ_model = Cacciato09Sats(
            prim_haloprop_key="halo_mvir", redshift=self.zeff,
            threshold=threshold
        )

        cens_prof_model = TrivialPhaseSpace(
            prim_haloprop_key="halo_mvir", redshift=self.zeff
        )

        sats_prof_model = NFWPhaseSpace(
            prim_haloprop_key="halo_mvir",
            redshift=self.zeff,
            conc_mass_model="dutton_maccio14",
        )        
        
        model_instance = self.set_model_params(cens_occ_model,
                                               sats_occ_model,
                                               cens_prof_model,
                                               sats_prof_model)
        
        return model_instance

    def paintGalaxies(self):
        """Paint galaxy positions, luminosities and SEDs into nbody.
        Saves them in self.galaxyCatalog.catalog.

        Returns
        -------
        None
        """
        self.convert_halocat_to_halotools()

        self.paintPositions()

        if not self.colorModel.no_colors:
            self.paintSEDs()
            self.paintShapes()
            
    def convert_halocat_to_halotools(self):
        hc = self.nbody.haloCatalog.catalog
        lb_ov_2 = self.nbody.domain.lbox[self.nbody.boxnum] / 2
        hcat = UserSuppliedHaloCatalog(simname='temp',
                                       redshift=self.nbody.domain.zeff, 
                                       particle_mass=self.nbody.haloCatalog.mpart, 
                                       Lbox=self.nbody.domain.lbox[self.nbody.boxnum],
                                       halo_rvir=hc['radius'][:], 
                                       halo_mvir=hc['mass'][:], 
                                       halo_id=hc['id'][:],
                                       halo_x=hc['pos'][:,0]+lb_ov_2, 
                                       halo_y=hc['pos'][:,1]+lb_ov_2, 
                                       halo_z=hc['pos'][:,2]+lb_ov_2, 
                                       halo_vx=hc['vel'][:,0], 
                                       halo_vy=hc['vel'][:,1],
                                       halo_vz=hc['vel'][:,2], 
                                       halo_hostid=hc['id'][:],
                                       halo_upid=np.zeros(len(hc['pos'][:,0]))-1, 
                                       halo_redshift=hc['redshift'])
        del self.nbody.haloCatalog.catalog
        self.nbody.haloCatalog.catalog = hcat
        return hcat

    def paintPositions(self):
        """Paint galaxy positions and luminosity in one band
        into nbody using CLF model.

        Returns
        -------
        None
        """

        domain = self.nbody.domain

        print('[{}] : Painting galaxy positions'.format(self.nbody.domain.rank))
        sys.stdout.flush()
        start = time()
        if len(self.nbody.haloCatalog.catalog) == 0:
            return

        self.clf_model_instance.populate_mock(self.nbody.haloCatalog.catalog, 
                                              halo_mass_column_key='halo_mvir',
                                              Num_ptcl_requirement=1, enforce_PBC=False)
        
        galaxies = self.clf_model_instance.mock.galaxy_table

        end = time()

        print('[{}] Finished assigning galaxies to halos. Took {}s'.format(
            self.nbody.domain.rank, end - start))
        sys.stdout.flush()

#        start = time()
#        pos, vel, z, density, mag, rhalo, halorad, haloid, halomass, bad = self.assignParticles(
#            z[~assigned], mag[~assigned], density[~assigned])
#        end = time()
#        print('[{}] Finished assigning galaxies to particles. Took {}s'.format(
#            self.nbody.domain.rank, end - start))
#        sys.stdout.flush()

        ngal = len(galaxies)
        central = galaxies['gal_type'] == 'centrals'
        central = central.astype(int)
        lb_ov_2 = self.nbody.domain.lbox[self.nbody.boxnum] / 2        
        galaxies['x'] -= lb_ov_2
        galaxies['y'] -= lb_ov_2
        galaxies['z'] -= lb_ov_2
        
        r = np.sqrt(galaxies['x']**2 + galaxies['y']**2 + galaxies['z']**2)
        z = self.nbody.cosmo.zofR(r)
        v_r = (galaxies['vx'] * galaxies['x'] + galaxies['vy'] * galaxies['y'] + galaxies['vz'] * galaxies['z']) / r
        z_rsd = z + v_r * (1 + z) / 299792.458            

        # done with halo catalog now
        if self.delete_after_assignment:
            self.nbody.haloCatalog.delete()
        else:
            pass

        self.nbody.galaxyCatalog.catalog['PX'] = galaxies['x']
        self.nbody.galaxyCatalog.catalog['PY'] = galaxies['y']
        self.nbody.galaxyCatalog.catalog['PZ'] = galaxies['z']
        self.nbody.galaxyCatalog.catalog['VX'] = galaxies['vx']
        self.nbody.galaxyCatalog.catalog['VY'] = galaxies['vy']
        self.nbody.galaxyCatalog.catalog['VZ'] = galaxies['vz']
        self.nbody.galaxyCatalog.catalog['Z_COS'] = z
        self.nbody.galaxyCatalog.catalog['Z'] = z_rsd
        self.nbody.galaxyCatalog.catalog['MAG_R'] = -2.5 * np.log10(galaxies['luminosity']) + 4.76
        self.nbody.galaxyCatalog.catalog['MVIR'] = galaxies['halo_mvir']
        self.nbody.galaxyCatalog.catalog['RVIR'] = galaxies['halo_rvir']
        self.nbody.galaxyCatalog.catalog['HALOID'] = galaxies['halo_id']
        self.nbody.galaxyCatalog.catalog['CENTRAL'] = central

    def paintSEDs(self):
        """Paint SEDs onto galaxies after positions and luminosities have
        already been assigned.

        Returns
        -------
        None

        """

        print('[{}] : Painting galaxy SEDs'.format(self.nbody.domain.rank))
        sys.stdout.flush()

        pos = np.vstack([self.nbody.galaxyCatalog.catalog['PX'],
                         self.nbody.galaxyCatalog.catalog['PY'],
                         self.nbody.galaxyCatalog.catalog['PZ']]).T
        mag = self.nbody.galaxyCatalog.catalog['MAG_R']
        z = self.nbody.galaxyCatalog.catalog['Z_COS']
        z_rsd = self.nbody.galaxyCatalog.catalog['Z']

        sigma5, ranksigma5, redfraction, \
            sed_idx, omag, amag, mag_evol, \
                sfr, met, smass = self.colorModel.assignSEDs(
                pos, mag, z, z_rsd)

        self.nbody.galaxyCatalog.catalog['SIGMA5'] = sigma5
        self.nbody.galaxyCatalog.catalog['PSIGMA5'] = ranksigma5
        self.nbody.galaxyCatalog.catalog['SEDID'] = sed_idx
        self.nbody.galaxyCatalog.catalog['MAG_R_EVOL'] = mag_evol
        self.nbody.galaxyCatalog.catalog['TMAG'] = omag
        self.nbody.galaxyCatalog.catalog['AMAG'] = amag
        self.nbody.galaxyCatalog.catalog['MSTAR'] = smass
        self.nbody.galaxyCatalog.catalog['SFR'] = sfr
        self.nbody.galaxyCatalog.catalog['METALLICITY'] = met

    def paintShapes(self):
        """Assign shapes to galaxies.

        Returns
        -------
        None
        """

        if self.shapeModel is None:
            return

        log_comoving_size, angular_size, epsilon = self.shapeModel.sampleShapes(
            self.nbody.galaxyCatalog.catalog)

        self.nbody.galaxyCatalog.catalog['TSIZE'] = angular_size
        self.nbody.galaxyCatalog.catalog['TE'] = epsilon
        self.nbody.galaxyCatalog.catalog['EPSILON_IA'] = np.zeros_like(epsilon)
        self.nbody.galaxyCatalog.catalog['COMOVING_SIZE'] = 10**log_comoving_size


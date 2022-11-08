from __future__ import print_function, division
from halotools.empirical_models import Cacciato09Cens, Cacciato09Sats, NFWPhaseSpace, TrivialPhaseSpace, HodModelFactory
from halotools.sim_manager import UserSuppliedHaloCatalog
from time import time
import numpy as np
import sys

from .galaxyModel import GalaxyModel
from .colorModel import ColorModel
from . import luminosityFunction
from . import shape


class CLFModel(GalaxyModel):

    def __init__(self, nbody, 
                 CLFModelConfig=None,
                 colorModelConfig=None,
                 shapeModelConfig=None):


        self.nbody = nbody
        self.zeff = self.nbody.domain.getZeff()

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
            
        for k in self.cen_clf_params:
            cens_prof_model.param_dict[k] = self.cen_prof_params[k]
            
        for k in self.sat_clf_params:
            sats_prof_model.param_dict[k] = self.sat_prof_params[k]              
            
        model_instance = HodModelFactory(
            centrals_occupation=cens_occ_model,
            centrals_profile=cens_prof_model,
            satellites_occupation=sats_occ_model,
            satellites_profile=sats_prof_model,
        )

        return model_instance            
            
        
    def setup_cacciato09_model(self):
        cens_occ_model = Cacciato09Cens(
            prim_haloprop_key="halo_m200m", redshift=self.zeff
        )

        sats_occ_model = Cacciato09Sats(
            prim_haloprop_key="halo_m200m", redshift=self.zeff
        )

        cens_prof_model = TrivialPhaseSpace(
            prim_haloprop_key="halo_m200m", redshift=self.zeff
        )

        sats_prof_model = NFWPhaseSpace(
            prim_haloprop_key="halo_m200m",
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
        lb_ov_2 = self.domain.lboxp[self.nbody.boxnum] / 2
        hcat = UserSuppliedHaloCatalog(simname='temp',
                                       redshift=self.domain.zeff, 
                                       particle_mass=self.nbody.haloCatalog.mpart, 
                                       Lbox=self.nbody.lbox[self.nbody.boxnum],
                                       halo_r200m=hc['radius'], 
                                       halo_m200m=hc['mass'], 
                                       halo_id=hc['id'],
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

        self.clf_model_instance.populate_mock(self.nbody.HaloCatalog.catalog, 
                                              halo_mass_column_key='halo_m200m',
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
        lb_ov_2 = self.domain.lboxp[self.nbody.boxnum] / 2        
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
        self.nbody.galaxyCatalog.catalog['M200'] = galaxies['halo_m200m']
        self.nbody.galaxyCatalog.catalog['R200'] = galaxies['halo_r200m']
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
            sed_idx, omag, amag, mag_evol = self.colorModel.assignSEDs(
                pos, mag, z, z_rsd)

        self.nbody.galaxyCatalog.catalog['SIGMA5'] = sigma5
        self.nbody.galaxyCatalog.catalog['PSIGMA5'] = ranksigma5
        self.nbody.galaxyCatalog.catalog['SEDID'] = sed_idx
        self.nbody.galaxyCatalog.catalog['MAG_R_EVOL'] = mag_evol
        self.nbody.galaxyCatalog.catalog['TMAG'] = omag
        self.nbody.galaxyCatalog.catalog['AMAG'] = amag

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


from halotools.empirical_models import (
    Cacciato09Cens,
    Cacciato09Sats,
    NFWPhaseSpace,
    TrivialPhaseSpace,
    HodModelFactory,
)
from scipy.special import erfc
import numpy as np

x_gauss, w_gauss = np.polynomial.legendre.leggauss(10)


class CLFCens(Cacciato09Cens):
    def get_published_parameters(self):
        param_dict = {
            "log_L_0": 9.557132570335106,
            "log_M_1": 9.677826309588625,
            "gamma_1": 85.26855242847066,
            "gamma_2": 0.2221830261551097,
            "sigma": 0.16923635878023224,
            "M_0_sigma": 13798870703166.668,
            "alpha_sigma": -0.10078033073027963,
        }

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

        sigma = self.param_dict["sigma"] * (
            prim_haloprop / self.param_dict["M_0_sigma"]
        ) ** (self.param_dict["alpha_sigma"])

        return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
            -((np.log10(prim_galprop / med_prim_galprop)) ** 2) / (2.0 * sigma**2)
        )

    def median_prim_galprop(self, **kwargs):
        mass = kwargs["prim_haloprop"]

        gamma_1 = self.param_dict["gamma_1"]
        gamma_2 = self.param_dict["gamma_2"]
        mass_c = 10 ** self.param_dict["log_M_1"]
        prim_galprop_c = 10 ** self.param_dict["log_L_0"]

        r = mass / mass_c

        return prim_galprop_c * (r / (1 + r)) ** gamma_1 * (1 + r) ** gamma_2

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

        return n


class CLFSats(Cacciato09Sats):
    def __init__(
        self,
        threshold=10.0,
        prim_haloprop_key="halo_mvir",
        prim_galprop_key="luminosity",
        **kwargs
    ):
        super().__init__(
            threshold=threshold,
            prim_haloprop_key=prim_haloprop_key,
            prim_galprop_key=prim_galprop_key,
            **kwargs
        )
        self.central_occupation_model = CLFCens(
            threshold=threshold,
            prim_haloprop_key=prim_haloprop_key,
            prim_galprop_key=prim_galprop_key,
            **kwargs
        )

    def get_default_parameters(self):
        param_dict = {
            "a_1": 1.204533927448177,
            "a_2": 0.3270944122861654,
            "log_M_2": 5.2617613662959775,
            "b_0": 0.03775141313660356,
            "b_1": 0.7923078487814424,
            "b_2": -0.030878419266771373,
            "delta_1": -1.0557363475245707,
            "delta_2": 0.21909775439800422,
            "log_L_0": 9.557132570335106,
            "log_M_1": 9.677826309588625,
            "gamma_1": 85.26855242847066,
            "gamma_2": 0.2221830261551097,
            "M_0_sigma": 13798870703166.668,
            "alpha_sigma": -0.10078033073027963,
            "alpha_12": 1.1637528173629659,
            "alpha_22": -4.022577531406281,
            "log_M_22": 13.0,
            "phi_2": -0.9422093297426509,
        }

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
            1.0 - 2.0 / np.pi * np.arctan(a_22 * (np.log10(prim_haloprop) - log_m_22))
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
            * (
                (prim_galprop / prim_galprop_cut) ** (alpha_sat_1 + 1)
                + self.param_dict["phi_2"]
                * (prim_galprop / prim_galprop_cut) ** (alpha_sat_2 + 1)
            )
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


class CLFCensZdep(CLFCens):
    def __init__(self, threshold=10.0, prim_haloprop_key='halo_mvir',
             prim_galprop_key='luminosity', **kwargs):
        super().__init__(
            threshold=threshold, prim_haloprop_key=prim_haloprop_key,
            prim_galprop_key=prim_galprop_key, **kwargs)

        self.list_of_haloprops_needed = ['halo_redshift']
    
    def get_published_parameters(self):
        param_dict = super().get_published_parameters()
        param_dict['M_0_sigma'] = 13
        param_dict['alpha_sigma'] = 0.1
        param_dict['gamma_z'] = 0.0
        param_dict['sigma_z'] = 0.0

        return param_dict
    
    def clf(self, prim_galprop=1e11, prim_haloprop=1e12, z=0.0):
        gamma_1 = self.param_dict["gamma_1"]
        gamma_2 = self.param_dict["gamma_2"]
        gamma_z = self.param_dict["gamma_z"]
        mass_c = 10 ** self.param_dict["log_M_1"]
        prim_galprop_c = 10 ** self.param_dict["log_L_0"]

        r = prim_haloprop / mass_c
        med_prim_galprop = (
            prim_galprop_c * (r / (1 + r)) ** gamma_1 * (1 + r) ** gamma_2 * (1 + z) ** gamma_z
        )
        
        sigma = self.param_dict['sigma'] * (prim_haloprop/self.param_dict['M_0_sigma'])**(self.param_dict['alpha_sigma']) * (1 + z) ** self.param_dict['sigma_z']

        return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
            -((np.log10(prim_galprop / med_prim_galprop)) ** 2)
            / (2.0 * sigma ** 2)
        )
    
    def median_prim_galprop(self, **kwargs):
        
        if "table" in list(kwargs.keys()):
            mass = kwargs["table"][self.prim_haloprop_key]
            z = kwargs["table"]["halo_redshift"]
        elif "prim_haloprop" in list(kwargs.keys()):
            mass = kwargs["prim_haloprop"]
            z = kwargs["z"]
        else:
            msg = (
                "\nYou must pass either a ``table`` or ``prim_haloprop`` "
                "argument to the ``median_prim_galprop`` function of the "
                "``Cacciato09Cens`` class.\n"
            )
            raise ValueError(msg)
            
#        mass = kwargs["prim_haloprop"]
#        z = kwargs["z"]

        gamma_1 = self.param_dict["gamma_1"]
        gamma_2 = self.param_dict["gamma_2"]
        gamma_z = self.param_dict["gamma_z"]
        mass_c = 10 ** self.param_dict["log_M_1"]
        prim_galprop_c = 10 ** self.param_dict["log_L_0"]

        r = mass / mass_c
        return (prim_galprop_c * (r / (1 + r)) ** gamma_1 * (1 + r) ** gamma_2 * (1 + z) ** gamma_z)


    def mean_occupation(self, prim_haloprop=1e12, prim_galprop_min=1e11,
                        prim_galprop_max=1e12, z=0.0,
                        **kwargs):
        
        if prim_galprop_min is not None:
            prim_galprop_min = prim_galprop_min
        else:
            prim_galprop_min = 10**self.threshold

        if prim_galprop_max is not None:
            if prim_galprop_max <= prim_galprop_min:
                msg = (
                    "\nFor the ``mean_occupation`` function of the "
                    "``Cacciato09Cens`` class the ``prim_galprop_max`` "
                    "keyword must be bigger than 10^threshold or "
                    "``prim_galprop_min`` if provided.\n"
                )
                raise ValueError(msg) 
                
        if "table" in list(kwargs.keys()):
            mass = kwargs["table"][self.prim_haloprop_key]
            z = kwargs["table"]["halo_redshift"]
        elif "prim_haloprop" in list(kwargs.keys()):
            mass = kwargs["prim_haloprop"]
            z = kwargs["z"]
        else:
            msg = (
                "\nYou must pass either a ``table`` or ``prim_haloprop`` "
                "argument to the ``median_prim_galprop`` function of the "
                "``Cacciato09Cens`` class.\n"
            )
            raise ValueError(msg)                

        a = np.log10(prim_galprop_min)
        b = np.log10(prim_galprop_max)

        if np.abs(b - a) > 0.305:
            n_1 = self.mean_occupation(
                prim_haloprop=mass, prim_galprop_min=10**a,
                prim_galprop_max=10**((a + b) / 2), z=z, **kwargs)
            n_2 = self.mean_occupation(
                prim_haloprop=prim_haloprop, prim_galprop_max=10**b,
                prim_galprop_min=10**((a + b) / 2), z=z, **kwargs)
            return n_1 + n_2

        for i in range(len(x_gauss)):
            log_prim_galprop_gauss = (b - a) / 2 * x_gauss[i] + (a + b) / 2
            clf_gauss = self.clf(prim_haloprop=mass,
                                 prim_galprop=10**log_prim_galprop_gauss, 
                                 z=z)
            dn = (b - a) / 2 * w_gauss[i] * clf_gauss

            if i == 0:
                n = dn
            else:
                n += dn

        return n


class CLFSatsZdep(CLFSats):

    def __init__(self, threshold=10.0, prim_haloprop_key='halo_mvir',
                 prim_galprop_key='luminosity', **kwargs):
        super().__init__(
            threshold=threshold, prim_haloprop_key=prim_haloprop_key,
            prim_galprop_key=prim_galprop_key, **kwargs)
        self.central_occupation_model = CLFCensZdep(
            threshold=threshold, prim_haloprop_key=prim_haloprop_key,
            prim_galprop_key=prim_galprop_key, **kwargs)
        self.list_of_haloprops_needed = ['halo_redshift']

    def get_default_parameters(self):
        param_dict = super().get_default_parameters()
        param_dict['M_0_sigma'] = 13
        param_dict['alpha_sigma'] = 0.1
        param_dict['alpha_21'] = 0.5
        param_dict['alpha_22'] = 0.5
        param_dict['phi_2'] = 1
        param_dict["log_M_22"] = 14
        param_dict["b_z"] = 0.0
        param_dict["a_z"] = 0.0
        param_dict["delta_z"] = 0.0

        return param_dict

    def clf(self, prim_galprop=1e10, prim_haloprop=1e12, z=0.0, **kwargs):
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
        b_z = self.param_dict["b_z"]
        
        log_prim_haloprop = np.log10(prim_haloprop)

        phi_sat = 10 ** (
            b_0
            + b_1 * (log_prim_haloprop - 12.0)
            + b_2 * (log_prim_haloprop - 12.0) ** 2
        ) * (1 + z) ** (b_z)

        a_1 = self.param_dict["a_1"]
        a_2 = self.param_dict["a_2"]
        log_m_2 = self.param_dict["log_M_2"]
        
        a_12 = self.param_dict["alpha_12"]
        a_22 = self.param_dict["alpha_22"]
        log_m_22 = self.param_dict["log_M_22"]
        
        a_z = self.param_dict["a_z"]
        
        alpha_sat_1 = -2.0 + a_1 * (
            1.0 - 2.0 / np.pi * np.arctan(a_2 * (np.log10(prim_haloprop) - log_m_2))
        ) * (1 + z) ** a_z
        
        alpha_sat_2 = -2.0 + a_12 * (
            1.0 - 2.0 / np.pi * np.arctan(a_22 * (np.log10(prim_haloprop) - log_m_22))
        ) * (1 + z) ** a_z

        for key, value in self.param_dict.items():
            if key in self.central_occupation_model.param_dict:
                self.central_occupation_model.param_dict[key] = value

        med_prim_galprop = self.central_occupation_model.median_prim_galprop(
            prim_haloprop=prim_haloprop, z=z
        )
        prim_galprop_cut = med_prim_galprop * 0.562
        delta = 10 ** (
            self.param_dict["delta_1"]
            + self.param_dict["delta_2"] * (np.log10(prim_haloprop) - 12) 
            + self.param_dict["delta_z"] * (1 + z)
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
        self, prim_haloprop=1e12, prim_galprop_min=1e11, prim_galprop_max=1e12, z=0.0, 
        **kwargs
    ):
        if prim_galprop_min is not None:
            prim_galprop_min = prim_galprop_min
        else:
            prim_galprop_min = 10**self.threshold

        if prim_galprop_max is not None:
            if prim_galprop_max <= prim_galprop_min:
                msg = (
                    "\nFor the ``mean_occupation`` function of the "
                    "``Cacciato09Cens`` class the ``prim_galprop_max`` "
                    "keyword must be bigger than 10^threshold or "
                    "``prim_galprop_min`` if provided.\n"
                )
                raise ValueError(msg) 
                
        if "table" in list(kwargs.keys()):
            mass = kwargs["table"][self.prim_haloprop_key]
            z = kwargs["table"]["halo_redshift"]
        elif "prim_haloprop" in list(kwargs.keys()):
            mass = kwargs["prim_haloprop"]
            z = kwargs["z"]
        else:
            msg = (
                "\nYou must pass either a ``table`` or ``prim_haloprop`` "
                "argument to the ``median_prim_galprop`` function of the "
                "``Cacciato09Cens`` class.\n"
            )
            raise ValueError(msg)     
            
            
        a = np.log10(prim_galprop_min)
        b = np.log10(prim_galprop_max)
        if np.abs(b - a) > 0.305:
            n_1 = self.mean_occupation(
                prim_haloprop=mass,
                prim_galprop_min=10**a,
                prim_galprop_max=10 ** ((a + b) / 2),
                z=z,
                **kwargs
            )
            n_2 = self.mean_occupation(
                prim_haloprop=mass,
                prim_galprop_max=10**b,
                prim_galprop_min=10 ** ((a + b) / 2),
                z=z,
                **kwargs
            )
            return n_1 + n_2
        
        for i in range(len(x_gauss)):
            log_prim_galprop_gauss = (b - a) / 2 * x_gauss[i] + (a + b) / 2
            clf_gauss = self.clf(
                prim_haloprop=mass, prim_galprop=10**log_prim_galprop_gauss, z=z
            )
            dn = (b - a) / 2 * w_gauss[i] * clf_gauss
            if i == 0:
                n = dn
            else:
                n += dn    
                
        return n
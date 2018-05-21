from __future__ import print_function, division
from .addgalsModel import ADDGALSModel

_available_models = ['ADDGALSModel']


class GalaxyCatalog(object):
    """
    Galaxy catalog class

    """

    def __init__(self, nbody):

        self.nbody = nbody
        self.catalog = {}

    def paintGalaxies(self, config):
        """Apply a galaxy model to the nbody sim

        Parameters
        ----------
        config : dict
            Galaxy model config file, must contain algorithm and
            relevant input information, e.g. LF, f_red(L,z), etc.

        Returns
        -------
        None
        """

        model_class = config.keys()[0]

        if not (model_class in _available_models):
            raise(ValueError("Model {} is not implemented".format(model_class)))

        if model_class == 'ADDGALSModel':
            model = ADDGALSModel(**config['ADDGALSModel'])

        model.paintGalaxies(self.lightcone)

    def writeCatalog(self):
        """Write galaxy catalog to disk.

        Returns
        -------
        None
        """

        pass

    def delete(self):
        """Delete galaxy catalog

        Returns
        -------
        None

        """

        for k in self.catalog.keys():

            del self.catalog[k]

import logging

from multiprocessing import Pool

import numpy as np
import tqdm


from binfit.fitting.minimizer import *

__all__ = [
    "BinFitter",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())

# TODO work on fixing parameters and stuff

class BinFitter:
    """This class performs the parameter estimation and calculation
    of a profile likelihood based on a constructed negative log
    likelihood function.

    Parameters
    ----------
    templates : Implemented AbstractTemplate
        An instance of a template class that provides a negative
        log likelihood function via the `create_nll` method.
    minimizer_id : str
        A string specifying the method to be used for  the
        minimization of the Likelihood function. Available are
        'scipy' and 'iminuit'.
    """

    def __init__(self, model, minimizer_id, cost='poisson'):
        self._model = model
        if(cost == 'poisson'):
         self._nll = model.create_nll()
        elif(cost == 'chi2'):
         self._nll = model.create_chi2()
        self._fit_result = None
        self._minimizer_id = minimizer_id
        self._fixed_parameters = list()
        self._bound_parameters = dict()

    def fix_nui_params(self):
        pass

    def do_fit(self, get_hesse=True, verbose=True, fix_nui_params=False):
        """Performs maximum likelihood fit by minimizing the
        provided negative log likelihoood function.

        Parameters
        ----------
        update_templates : bool, optional
            Whether to update the parameters of the given templates
            or not. Default is True.
        verbose : bool, optional
            Whether to print fit information or not. Default is True
        fix_nui_params : bool, optional
            Wheter to fix nuissance parameters in the fit or not.
            Default is False.
        get_hesse : bool, optional
            Whether to calculate the Hesse matrix in the estimated
            minimum of the negative log likelihood function or not.
            Can be computationally expensive if the number of parameters
            in the likelihood is high. It is only needed for the scipy
            minimization method. Default is True.

        Returns
        -------
        MinimizeResult : namedtuple
            A namedtuple with the most important information about the
            minimization.
        """
        minimizer = minimizer_factory(
            self._minimizer_id, self._nll, self._nll.param_names
        )

#        if fix_nui_params:
#            for i in range(self._templates.num_processes,
#                           self._templates.num_nui_params +
#                           self._templates.num_processes):
#                minimizer.set_param_fixed(i)

        for param_id in self._fixed_parameters:
            minimizer.set_param_fixed(param_id)

        for param_id, bounds in self._bound_parameters.items():
            minimizer.set_param_bounds(param_id, bounds)
        print(minimizer._param_bounds)

        # fit_result = minimizer.minimize(
        #     self._nll.x0, get_hesse=False, verbose=False
        # )
        fit_result = minimizer.minimize(
            self._nll.x0, get_hesse=get_hesse, verbose=verbose
        )

#        if update_templates:
#            self._templates.update_parameters(fit_result.params.values)

        return fit_result


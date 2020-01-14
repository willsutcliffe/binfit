import logging


from templatefitter.histograms import Hist1d
from FittingFramework.templates.abstract_template import AbstractTemplate
import numpy as np

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = ["ChannelTemplate"]


class ChannelTemplate(AbstractTemplate):
    """ combines several templates according with both yields 
    and efficiencies. There is the possibility of using shared 
    yields between channeltemplates.
    """
    def __init__(
        self,
        name,
        variable,
        templates,
        params,
        initialyields,
        initialeffs,
        sharedyields = 'False',
        yieldnames = [],
        color=None,
        pretty_variable=None,
        pretty_label=None,
    ):
        super(Template1d, self).__init__(name=name, params=params)

        self._templates = templates
        self._yield_indices = []
        self._eff_indices = []
        self._initalyields = initialyields
        self._initaleffs = initialeffs
        self._sharedyields = 'False'
        self._yieldnames = yieldnames
        self._init_params()


    def _init_params(self):
        """ Register fractions / efficiencies
        as parameters"""
        effnames = ["efficiency_{}_{}".format(self.name,x) for x in templates.keys()]
        self._yield_indices = self._params.addParameters(self._initialpars,parnames)
        if self._sharedyields:
           self._yield_indices = self._initialyields
        else:
           if( len(yieldnames) > 0):
               yieldnames = self._yieldnames
           else =:
               yieldnames = ["yield_{}_{}".format(self.name,x) for x in templates.keys()]
           self._yield_indices = self._params.addParameters(self._initialpars,parnames)

    def get_pars(self):
        return(self._params.getParameters(self._par_indices))

    def fractions(self):
        """
        Computes the multitemplate binfractions using individual templates
        constructed together with the number of 
        """
        yields = self._params.getParameters(self._yield_indices)
        efficiencies = self._params.getParameters(self._eff_indices)
        fractions_per_template = np.array(
            [template.fractions() for template
             in self.templates.values()]
        )
        return((yields*efficiencies)@fractions)
        
         



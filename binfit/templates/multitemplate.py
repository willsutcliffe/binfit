import logging


from binfit.histograms import Hist1d
from binfit.templates import AbstractTemplate
import numpy as np
from scipy.linalg import block_diag


logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = ["MultiTemplate","MultiNormTemplate"]

class MultiTemplate(AbstractTemplate):
    """ combines several templates according to fractions.
    This produces a new pdf.
    """
    def __init__(
        self,
        name,
        variable,
        templates,
        params,
        initialpars,
        color=None,
        pretty_variable=None,
        pretty_label=None,
    ):
        super(MultiTemplate, self).__init__(name=name, params=params)

        self._templates = templates
        self._par_indices = []
        self._initalpars = initialpars
        self._init_params()
        self._num_templates = len(templates)

        self.pretty_variable = pretty_variable
        self.color = color
        self.pretty_label = pretty_label


    def _init_params(self):
        """ Register fractions / efficiencies
        as parameters"""
        parnames = ["efficiency_{}".format(x) for x in templates.keys()]
        self._par_indices = self._params.addParameters(self._initialpars,parnames)

    def get_pars(self):
        return(self._params.getParametersbyIndex(self._par_indices))

    def get_bin_pars(self):
        pars = np.array(
            [template.get_bin_pars() for template
             in self._templates.values()]
        )
        return(pars)

    def fractions(self):
        """
        Computes the multitemplate binfractions using individual templates
        constructed together with the number of 
        """
        pars = self._params.getParametersbyIndex(self._par_indices)
        fractions_per_template = np.array(
            [template.fractions() for template
             in self.templates.values()]
        )
        return(pars@fractions)

class MultiNormTemplate(AbstractTemplate):
    """ combines several templates according to fractions.
    This produces a new pdf.
    """
    def __init__(
        self,
        name,
        templates,
        params,
        initialpars,
        color=None,
        pretty_variable=None,
        pretty_label=None,
    ):
        super(MultiNormTemplate, self).__init__(name=name, params=params)

        self._templates = templates
        self._bins = next(iter(self._templates.values()))._bins
        self._par_indices = []
        self._initialpars = initialpars
        self._init_params()
        self._num_templates = len(templates)

        self.pretty_variable = pretty_variable
        self.color = color
        self.pretty_label = pretty_label

    def _init_params(self):
        """ Register fractions / efficiencies
        as parameters"""
        parnames = ["fraction_{}".format(x) for x in self._initialpars.keys()]
        self._par_indices = self._params.addParameters(self._initialpars.values(),parnames)


    def get_pars(self):
        return(self._params.getParametersbyIndex(self._par_indices))

    def get_bin_pars(self):
        pars = np.concatenate([template.get_bin_pars() for template in self._templates.values()])
        return(pars)

    def fractions(self):
        """
        Computes the multitemplate binfractions using individual templates
        constructed together with the number of 
        """
        pars = self._params.getParametersbyIndex(self._par_indices)
        pars = np.append(pars,1-np.sum(pars))
        fractions_per_template = np.array(
            [template.fractions() for template
             in self._templates.values()]
        )
        return(pars@fractions_per_template)
    
    def reshapedfractions(self):
        """Calculates the expected number of events per bin using
        the current yield value and nuissance parameters. Shape
        is (`num_bins`,).
        """
        return  self.fractions().reshape(
                self.shape()
        )

    def allfractions(self):
        """
        Computes the multitemplate binfractions using individual templates
        constructed together with the number of 
        """
        pars = self._params.getParametersbyIndex(self._par_indices)
        pars = np.append(pars,1-np.sum(pars))
        fractions_per_template = np.array(
            [template.fractions() for template
             in self._templates.values()]
        )
        return(np.concatenate(pars[:,np.newaxis]*fractions_per_template))
        
    def inv_corr_mat(self):
        inv_corr_mats = [template.inv_corr_mat() for template
                         in self._templates.values()]
        return block_diag(*inv_corr_mats)
         

    def shape(self):
        """tuple of int: Template shape."""
        return(next(iter(self._templates.values())).shape())


    def bin_mids(self):
        return next(iter(self._templates.values())).bin_mids()

    def bin_edges(self):
        return next(iter(self._templates.values())).bin_edges()

    def bin_widths(self):
        return next(iter(self._templates.values())).bin_widths()


    def errors(self):
        """numpy.ndarray: Total uncertainty per bin. This value is the
        product of the relative uncertainty per bin and the current bin
        values. Shape is (`num_bins`,).
        """
        pars = self._params.getParametersbyIndex(self._par_indices)
        pars = np.append(pars,1-np.sum(pars))
        uncertainties_sq = [(par*template.fractions()*template.errors())** 2 for par,template in
                            zip(pars, self._templates.values())]
        total_uncertainty = np.sqrt(np.sum(np.array(uncertainties_sq), axis=0))
        return total_uncertainty

    def labels(self):
        return([template.labels()[0] for template in self._templates.values()]) 

    def colors(self):
        return([template.colors()[0] for template in self._templates.values()]) 

import logging


from binfit.histograms import Hist1d
from binfit.templates import SingleTemplate
import numpy as np

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = ["Template1d"]


class Template1d(SingleTemplate):
    """A 1d template class.
    """
    def __init__(
        self,
        name,
        variable,
        hist1d,
        params,
        color=None,
        pretty_variable=None,
        pretty_label=None,
    ):
        super(Template1d, self).__init__(name=name, params=params)

        self._hist = hist1d
        self._flat_bin_counts = self._hist.bin_counts.flatten()
        self._flat_bin_errors_sq = self._hist.bin_errors_sq.flatten()
        self._bins = hist1d.shape
        self._num_bins = hist1d.num_bins
        self._range = hist1d.range
        self._cov_mats = list()
        self._cov = None
        self._corr = None
        self._inv_corr = None
        self._relative_errors = None
        self._bin_par_indices = []
        self._fraction_function = self.bin_fractions
        self._sys_errors = []
        self._sys_par_indices = []
        self._nupvars = np.array([])
        self._ndownvars = np.array([])
        self._upvars = []
        self._downvars = []
        #self._init_params()

        self._init_errors()


        self._variable = variable
        self.pretty_variable = pretty_variable
        self.color = color
        self.pretty_label = pretty_label

    def colors(self):
        return([self.color])
    def labels(self):
        return([self.name])

    def add_variation(self, data, var, weights_up, weights_down, total_weight=None, nominal_weight=None):
        """Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template."""
        if (total_weight==None or nominal_weight==None):
            hup = Hist1d(
                bins=self._hist.num_bins, range=self._range, data=data[var], weights=data[weights_up]
            )
            hdown = Hist1d(
                bins=self._hist.num_bins, range=self._range, data=data[var], weights=data[weights_down]
            )
        else:
            hup = Hist1d(
                bins=self._hist.num_bins, range=self._range, data=data[var],
                weights=data[weights_up]*data[total_weight]/data[nominal_weight]
            )
            hdown = Hist1d(
                bins=self._hist.num_bins, range=self._range, data=data[var],
                weights=data[weights_down]*data[total_weight]/data[nominal_weight]
            )
        self._add_cov_mat_up_down(hup, hdown)

    def add_gaussian_variation(self, data, var, nominal_weight, new_weight, Nstart=None, Nweights=None , total_weight=None):
        """Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template."""
        if Nweights==None:
            Nweights = len([col for col in data.columns if new_weight in col])
        if total_weight == None:
            nominal = Hist1d(
            bins=self._hist.num_bins, range=self._range, data=data[var], weights=data[nominal_weight]
            ).bin_counts
            bin_counts = np.array([Hist1d(
            bins=self._hist.num_bins, range=self._range, data=data[var],
            weights=data['{}_{}'.format(new_weight,i)]).bin_counts for i in range(Nstart, Nweights+Nstart)])
        else:
            nominal = Hist1d(
            bins=self._hist.num_bins, range=self._range, data=data[var], weights=data[total_weight]
            ).bin_counts
            bin_counts = np.array([Hist1d(
            bins=self._hist.num_bins, range=self._range, data=data[var],
            weights=data['{}_{}'.format(new_weight,i)]*data[total_weight]/data[nominal_weight]).bin_counts for i in range(Nstart,Nweights+Nstart)])
        cov_mat = np.matmul((bin_counts - nominal).T, (bin_counts - nominal))/Nweights
        self._add_cov_mat(cov_mat)

    def add_singlepar_variation(self, data, weights_up, weights_down,name,register=True):
        """Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template."""
        hup = Hist1d(
            bins=self._hist.num_bins, range=self._range, data=data, weights=weights_up
        )
        hdown = Hist1d(
            bins=self._hist.num_bins, range=self._range, data=data, weights=weights_down
        )
        self._upvars.append(list(hup.bin_counts.flatten()-self._flat_bin_counts))
        self._downvars.append(list(hdown.bin_counts.flatten()-self._flat_bin_counts))
        self._nupvars = np.array(self._upvars)
        self._ndownvars = np.array(self._downvars)
        if register:
           self._params.addParameter(name,0.)
        else:
           self._sys_par_indices.append(self._params.getIndex(name))    
        self._fraction_function = self.bin_fractions_with_sys
        
    def add_cov(self, cov):
        self._add_cov_mat(cov)

    def _init_params(self):
        """ Add parameters for the template """
        binpars = np.full((self._num_bins),0.)
        binparnames = ["{}_binpar_{}".format(self.name,i) for i in range(0,self._num_bins)]
        self._bin_par_indices = self._params.addParameters(binpars,binparnames)



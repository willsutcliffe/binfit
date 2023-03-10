import logging
from functools import reduce
from itertools import product

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np

from binfit.histograms import Hist2d
from binfit.templates import SingleTemplate

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = ["Template2d"]


class Template2d(SingleTemplate):
    """A 2 dimensional template.
    """
    def __init__(
            self,
            name,
            vars,
            hist2d,
            params,
            color=None,
            x_pretty_var=None,
            y_pretty_var=None,
            pretty_label=None,
    ):
        super(Template2d, self).__init__(name=name,params=params)

        self._hist = hist2d
        self._flat_bin_counts = self._hist.bin_counts.flatten()
        self._flat_bin_errors_sq = self._hist.bin_errors_sq.flatten()
        self._bins = hist2d.shape
        self._num_bins = reduce(lambda x, y: x*y, hist2d.num_bins)
        self._range = hist2d.range

        #self._init_params()
        self._init_errors()
        self._x_var = vars[0]
        self._y_var = vars[1]
        self.x_pretty_var = x_pretty_var
        self.y_pretty_var = y_pretty_var
        self.color = color
        self.pretty_label = pretty_label
        


    def _init_params(self):
        """ Add parameters for the template """
        binpars = np.full((self._num_bins),0.)
        binparnames = ["{}_binpar_{}".format(self.name,i) for i in range(0,self._num_bins)]
        self._bin_par_indices = self._params.addParameters(binpars,binparnames)

    def colors(self):
        return([self.color])
    def labels(self):
        return([self.name])


    def add_variation(self, data, vars, weights_up, weights_down, total_weight=None, nominal_weight=None):
        """Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template."""

        if (total_weight==None or nominal_weight==None):
            
            hup = Hist2d(
                bins = self.bin_edges(), data=[data[vars[0]], data[vars[1]]], weights=data[weights_up]
            )
            hdown = Hist2d(
                bins = self.bin_edges(), data=[data[vars[0]], data[vars[1]]], weights=data[weights_down]
            )
        else:
            hup = Hist2d(
                bins = self.bin_edges(), data=[data[vars[0]], data[vars[1]]],
                weights=data[weights_up]*data[total_weight]/data[nominal_weight]
            )
            hdown = Hist2d(
                bins = self.bin_edges(), data=[data[vars[0]], data[vars[1]]],
                weights=data[weights_down]*data[total_weight]/data[nominal_weight]
            )
        self._add_cov_mat_up_down(hup, hdown)

    def add_gaussian_variation(self, data, vars, nominal_weight, new_weight, Nstart=None, Nweights=None , total_weight=None):
        """Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template."""
        if Nweights==None:
            Nweights = len([col for col in data.columns if new_weight in col])
        if total_weight == None:
            nominal = Hist2d(
            bins = self.bin_edges(), data=[data[vars[0]], data[vars[1]]], weights=data[nominal_weight]
            ).bin_counts
            bin_counts = np.array([Hist2d(
            bins = self.bin_edges(), data=[data[vars[0]], data[vars[1]]],
            weights=data['{}_{}'.format(new_weight,i)]).bin_counts for i in range(Nstart,Nstart+Nweights)])
        else:
            nominal = Hist2d(
            bins = self.bin_edges(), data=[data[vars[0]], data[vars[1]]], weights=data[total_weight]
            ).bin_counts.flatten()
            bin_counts = np.array([Hist2d(
            bins = self.bin_edges(), data=[data[vars[0]], data[vars[1]]],
            weights=data['{}_{}'.format(new_weight,i)]*data[total_weight]/data[nominal_weight]).bin_counts.flatten() for i in range(Nstart, Nstart+Nweights)])
        cov_mat = np.matmul((bin_counts - nominal).T, (bin_counts - nominal))/Nweights
        self._add_cov_mat(cov_mat)


    def add_singlepar_variation(self, data, vars, weights_up, weights_down, name):
        hup = Hist2d(
            bins = self.bin_edges(), data=[data[vars[0]], data[vars[1]]], weights=weights_up
        )
        hdown = Hist2d(
            bins = self.bin_edges(), data=[data[vars[0]], data[vars[1]]], weights=weights_down
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

    def plot_on(self, fig, ax):
        """Plots the 2d template on the given axis.
        """
        edges = self._hist.bin_mids
        xe = edges[0]
        ye = edges[1]

        xy = np.array(list(product(xe, ye)))

        viridis = cm.get_cmap("viridis", 256)
        cmap = viridis(np.linspace(0, 1, 256))
        cmap[0, :] = np.array([1, 1, 1, 1])
        newcm = ListedColormap(cmap)
        cax =ax.hist2d(
            x=xy[:, 0],
            y=xy[:, 1],
            weights=self.values.flatten(),
            bins=self._hist.bin_edges,
            cmap=newcm,
            label=self.name
        )
        ax.set_title(self.pretty_label if self.pretty_label is not None else
                     self.name)
        ax.set_xlabel(self.x_pretty_var if self.x_pretty_var is not None else
                      self._x_var)
        ax.set_ylabel(self.y_pretty_var if self.y_pretty_var is not None else
                      self._y_var)
        fig.colorbar(cax[3])

    def plot_x_projection_on(self, ax):
        """Plots the x projection of the template on the
        given axis.
        """
        values = np.sum(self.values, axis=1)
        errors = np.sqrt(np.sum(self.errors**2, axis=1))
        projection = self._hist.x_projection()
        self._plot_projection(ax, values, errors, projection)
        ax.set_xlabel(self.x_pretty_var if self.x_pretty_var is not None
                      else self._x_var)

    def plot_y_projection_on(self, ax):
        """Plots the y projection of the template on the
        given axis.
        """
        values = np.sum(self.values, axis=0)
        errors = np.sqrt(np.sum(self.errors**2, axis=0))
        projection = self._hist.y_projection()
        self._plot_projection(ax, values, errors, projection)
        ax.set_xlabel(self.y_pretty_var if self.y_pretty_var is not None
                      else self._y_var)

    def _plot_projection(self, ax, values, errors, projection):
        """Helper function to prevent code duplication.
        """
        ax.hist(
            projection.bin_mids,
            weights=values,
            bins=projection.bin_edges,
            color=self.color,
            edgecolor="black",
            histtype="stepfilled",
            label=self.pretty_label if self.pretty_label is not None else self.name,
        )
        ax.bar(
            x=projection.bin_mids,
            height=2 * errors,
            width=projection.bin_widths,
            bottom=values - errors,
            color="black",
            hatch="///////",
            fill=False,
            lw=0,
        )

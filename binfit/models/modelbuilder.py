# class which handles computation

import numpy as np
from abc import ABC, abstractmethod
from binfit.parameters import ParameterHandler
from scipy.linalg import block_diag
from binfit.utility import xlogyx
from numba import jit
from binfit.histograms import Hist1d, Hist2d
from binfit.utility import cov2corr, get_systematic_cov_mat


__all__ = ["ModelBuilder"]


class ModelBuilder:
    def __init__(self, params,data,channels='False'):
        self.params = params
        self.templates = {}
        self.plottemplates = {}
        self.packedtemplates = {}
        self.data = data
        self.xobs = data.bin_counts.flatten()
        self.xobserrors = data.bin_errors.flatten()
        self.yieldindices = []
        self.plotyieldindices = []
        self.subfraction_indices = []
        self.conindices = []
        self.connames = []
        self.convalue = np.array([])
        self.consigma = np.array([])
        self._inv_corr = np.array([])
        self.bin_par_slice = (0,0)
        self._dim = None
        self.has_data = 1
        self.shape =()
        self.convertermatrix = None
        self.convertervector = None
        self.inverseconcov = []
        self.concov = []
        self.num_fractions = 0
        self.num_templates = 0
        self.num_bins = None
        self.global_covs=[]
        self.confunc = self._nocon_term
        self.uperrors = None
        self.downerrors = None

    def AddTemplate(self,template,value,create=True, same = True):
        if(self.num_bins == None):
            self.num_bins = template.num_bins
        self.packedtemplates[template.name] = template
        if template._num_templates > 1:
           self.plottemplates[template.name] = template
           self.AddMultiTemplate(template,value,create,same)
        else:
           self.plottemplates[template.name] = template
           self.templates[template.name] = template
           if create:
              yieldindex = self.params.addParameter(
                  "{}_yield".format(template.name),value)
              self.yieldindices.append(yieldindex)
              self.plotyieldindices.append(yieldindex)
           else:
              self.yieldindices.append(value)
              self.plotyieldindices.append(value)
           

        if self._dim == None:
            self._dim = len(template.bins)
        self.num_templates = len(self.templates)

    def AddMultiTemplate(self,template,value,create=True, same = True):
        self.subfraction_indices += template._par_indices
        self.num_fractions += len(template._par_indices)
        if create:
           if same:
              yieldindex = self.params.addParameter(
                  "{}_yield".format(template.name),value)
              self.plotyieldindices.append(yieldindex)
           for subtemp in template._templates.values():
              self.templates[subtemp.name] = subtemp
              if not same:
                 yieldindex = self.params.addParameter(
                  "{}_yield".format(subtemp.name),value[subtemp.name])
              self.yieldindices.append(yieldindex)
        else:
           for subtemp in template._templates.values():
               self.templates[subtemp.name] = subtemp
               self.yieldindices.append(value[subtemp.name])

    def TemplateMatrix(self):
        """ Creates the fixed template stack """
        fractions_per_template = [template._flat_bin_counts for template
             in self.templates.values()]
         
        self.template_fractions = np.stack(fractions_per_template)
        self.shape = self.template_fractions.shape

    def RelativeErrorMatrix(self):
        """ Creates the fixed template stack """
        errors_per_template =  [template.errors() for template
             in self.templates.values()]
        
        self.template_errors = np.stack(errors_per_template)

    @property
    def up_errors(self):
        """Property that returns upwards errors"""
        return self.uperrors

    @property
    def down_errors(self):
        """Property that returns downwards errors"""
        return self.downerrors

    def InitialiseBinPars(self):
        """ Add bin parameters for the template """

        bin_par_names = []
        bin_par_indices = []
        for template in self.templates.values():
            bin_par_names = ["{}_binpar_{}".format(template.name,i) for i in range(0,self.num_bins)]
            bin_pars = [0.]*self.num_bins
            temp_bin_par_indices =  self.params.addParameters(bin_pars,bin_par_names)
            template.set_bin_par_indices(temp_bin_par_indices)
            bin_par_indices += temp_bin_par_indices
        self.bin_par_slice = (bin_par_indices[0],bin_par_indices[-1]+1)

    def StoreSysPars(self):
        """ Add bin parameters for the template """

        sys_par_indices = []
        for template in self.templates.values():
            if len(template.sys_par_indices) > 0:
                sys_par_indices.append(template.sys_par_indices)
        sys_par_indices = np.unique(np.concatenate(sys_par_indices))
        self.sys_par_indices = sys_par_indices
        self.uperrors = np.stack([template.get_up_vars() for template in self.templates.values()])
        self.downerrors = np.stack([template.get_down_vars() for template in self.templates.values()])




    @jit(forceobj=True)
    def ExpectedEventsPerBin(self, bin_pars, yields, sub_pars, sys_pars):
        singlepar_corr = np.prod(1+sys_pars*(sys_pars>0)*self.up_errors +
                                 sys_pars*(sys_pars<0)*self.down_errors,0)
        corrections = (1+self.template_errors*bin_pars) * (singlepar_corr)
        sub_fractions = np.matmul(self.convertermatrix,sub_pars) + self.convertervector
        pdfs = self.template_fractions*corrections
        norm_pdfs = pdfs/np.sum(pdfs,1)[:,np.newaxis]
        expected_per_bin = np.sum(norm_pdfs*yields*sub_fractions,axis=0)
        return(expected_per_bin)
        # compute overall correction terms
        # get sub template fractions into the correct form with the converter and additive part
        # normalised expected corrected fractions
        # determine expected amount of events in each bin
        

    def FractionConverter(self):
        """ Determines the matrices required to 
        tranform the subtemplate parameters"""
        arrays = []
        additive = []
        count = 0
        for template in self.packedtemplates.values():
            if template._num_templates == 1:
                arrays += [np.zeros((1,self.num_fractions))]
                additive += [np.full((1,1),1.)] 
            else:
                n_fractions = template._num_templates -1
                a = np.identity(n_fractions)
                a = np.vstack([a,np.full((1,n_fractions),-1.)])
                count += n_fractions
                a = np.pad(a,((0,0),(count- n_fractions,self.num_fractions-count)),mode='constant')
                arrays += [a]
                additive += [np.vstack([np.zeros((n_fractions,1)),np.full((1,1),1.)])]
        self.convertermatrix = np.vstack(arrays)
        self.convertervector = np.vstack(additive)
        
    def AddConstraint(self,name,value,sigma):
        self.conindices.append(self.params.getIndex(name))
        self.convalue = np.append(self.convalue, value)
        self.connames += [name]
        self.inverseconcov.append(np.array([1/sigma**2]))
        self.concov.append(np.array([sigma**2]))

    def AddConstraints(self,names,values,cov):
        self.connames += names
        for name, value in zip(names, values):
          self.conindices.append(self.params.getIndex(name))
          self.convalue = np.append(self.convalue, value)
        self.inverseconcov.append(np.linalg.inv(cov))
        self.concov.append(cov)


    def xexpected(self):
        yields = self.params.getParametersbyIndex(self.yieldindices)
        fractions_per_template = np.array(
            [template.fractions() for template
             in self.templates.values()]
        )
        return(yields@fractions_per_template)

    def binpars(self):
        binpars = np.concatenate(
            [template.get_bin_pars() for template
             in self.templates.values()]
        )
        return(binpars)

    def AddGlobalCov(self,cov):
        self.global_covs.append(cov)

    def add_singlepar_variation(self, var, systematic, data_dict, up_weight_name, down_weight_name):
        i = 0
        register = True
        for template_name in self.templates.keys():
            if template_name in data_dict.keys():

                self.templates[template_name].add_singlepar_variation(data_dict[template_name][var], data_dict[template_name][up_weight_name],
                                        data_dict[template_name][down_weight_name],
                                        systematic,
                                        register
                                        )
            else:

                self.templates[template_name].add_singlepar_flat(systematic,register)

            if i == 0:
                register = False
            i += 1

    def AddGlobalTrackingCov(self, inputdfs, var, weight_names, error_per_trk):
        keys=self.templates.keys()
        temp=list(self.templates.values())[0]
        dfs = [inputdfs[key] for key in keys]
        total_weight=weight_names['total_weight']
        ntrack_names=weight_names['ntrack_names']
        hnom = np.concatenate([Hist1d( bins=temp.bin_edges(), data=df[var], weights=df[total_weight]).bin_counts for df in dfs])
        hup = np.concatenate([Hist1d( bins=temp.bin_edges(), data=df[var], weights=df[total_weight]* (1 +  df[ntrack_names]*error_per_trk)).bin_counts for df in dfs])
        hdown = np.concatenate([Hist1d( bins=temp.bin_edges(), data=df[var], weights=df[total_weight]* (1 -  df[ntrack_names]*error_per_trk)).bin_counts for df in dfs])
        covMatrix=get_systematic_cov_mat(hnom, hup, hdown)
        self.AddGlobalCov(covMatrix)
        
    def AddGlobalTrackingCov2D(self, inputdfs, var, weight_names, error_per_trk):
        keys=self.templates.keys()
        temp=list(self.templates.values())[0]
        dfs = [inputdfs[key] for key in keys]
        total_weight=weight_names['total_weight']
        ntrack_names=weight_names['ntrack_names']
        hnom = np.concatenate([Hist2d( bins=temp.bin_edges(), data=[df[var[0]],df[var[1]]], weights=df[total_weight]).bin_counts for df in dfs])
        hup = np.concatenate([Hist2d( bins=temp.bin_edges(), data=[df[var[0]],df[var[1]]], weights=df[total_weight]* (1 +  df[ntrack_names]*error_per_trk)).bin_counts for df in dfs])
        hdown = np.concatenate([Hist2d( bins=temp.bin_edges(), data=[df[var[0]],df[var[1]]], weights=df[total_weight]* (1 -  df[ntrack_names]*error_per_trk)).bin_counts for df in dfs])
        covMatrix=get_systematic_cov_mat(hnom, hup, hdown)
        self.AddGlobalCov(covMatrix)
        
        
    def AddGlobalVarCov(self, inputdfs, var, weight_names, Nstart, Nvar):
        keys=self.templates.keys()
        temp=list(self.templates.values())[0]
        dfs = [inputdfs[key] for key in keys]
        nominal_weight=weight_names['nominal_weight']
        total_weight=weight_names['total_weight']
        new_weight=weight_names['new_weight']
        hnom = np.concatenate([Hist1d( bins=temp.bin_edges(), data=df[var], weights=df[total_weight]).bin_counts for df in dfs])
        varMatrix=[]
        for i in range(Nstart, Nvar+Nstart):
            row = np.concatenate([Hist1d( bins=temp.bin_edges(), data=df[var], weights=df['{}_{}'.format(new_weight,i)]*df[total_weight]/df[nominal_weight]).bin_counts for df in dfs])
            varMatrix.append(row)
        varMatrix=np.array(varMatrix)
        covMatrix=np.matmul((varMatrix-hnom).T,(varMatrix-hnom))/Nvar
        self.AddGlobalCov(covMatrix)
        
    def AddGlobalVarCov2D(self, inputdfs, var, weight_names, Nstart, Nvar):
        keys=self.templates.keys()
        temp=list(self.templates.values())[0]
        dfs = [inputdfs[key] for key in keys]
        nominal_weight=weight_names['nominal_weight']
        total_weight=weight_names['total_weight']
        new_weight=weight_names['new_weight']
        hnom = np.concatenate([Hist2d( bins=temp.bin_edges(), data=[df[var[0]],df[var[1]]], weights=df[total_weight]).bin_counts.flatten() for df in dfs])
        varMatrix=[]
        for i in range(Nstart, Nvar+Nstart):
            row = np.concatenate([Hist2d( bins=temp.bin_edges(), data=[df[var[0]],df[var[1]]], weights=df['{}_{}'.format(new_weight,i)]*df[total_weight]/df[nominal_weight]).bin_counts.flatten() for df in dfs])
            varMatrix.append(row)
        varMatrix=np.array(varMatrix)
        covMatrix=np.matmul((varMatrix-hnom).T,(varMatrix-hnom))/Nvar
        self.AddGlobalCov(covMatrix)
            
        
    def AddGaussianVariations(self, inputdfs, var, weight_names, Nstart, Nvar):
        keys=self.templates.keys()
        temp=list(self.templates.values())[0]
        dfs = [inputdfs[key] for key in keys]
        nominal_weight=weight_names['nominal_weight']
        total_weight=weight_names['total_weight']
        new_weight=weight_names['new_weight']
        hnom = np.concatenate([Hist1d( bins=temp.bin_edges(), data=df[var],
            weights=df[total_weight]).bin_counts for df in dfs])
        varMatrix=[]
        for key in keys:
            self.templates[key].add_gaussian_variation(inputdfs[key], var, 
                    nominal_weight, new_weight, Nstart, Nvar, total_weight)
        for i in range(Nstart, Nstart+Nvar):
            row = np.concatenate([Hist1d( bins=temp.bin_edges(), data=df[var],
            weights=df['{}_{}'.format(new_weight,i)]*df[total_weight]/df[nominal_weight]).bin_counts for df in dfs])
            varMatrix.append(row)
        varMatrix=np.array(varMatrix)
        covMatrix=np.matmul((varMatrix-hnom).T,(varMatrix-hnom))/Nvar
        self.AddGlobalCov(covMatrix)
        
    def AddGaussianVariations2D(self, inputdfs, var, weight_names, Nstart, Nvar):
        keys=self.templates.keys()
        temp=list(self.templates.values())[0]
        dfs = [inputdfs[key] for key in keys]
        nominal_weight=weight_names['nominal_weight']
        total_weight=weight_names['total_weight']
        new_weight=weight_names['new_weight']
        hnom = np.concatenate([Hist2d( bins=temp.bin_edges(), data=[df[var[0]],df[var[1]]],
            weights=df[total_weight]).bin_counts.flatten() for df in dfs])
        varMatrix=[]
        for key in keys:
            self.templates[key].add_gaussian_variation(inputdfs[key], var,
                    nominal_weight, new_weight, Nstart, Nvar, total_weight)
        for i in range(Nstart, Nstart+Nvar):
            row = np.concatenate([Hist2d( bins=temp.bin_edges(), data=[df[var[0]],df[var[1]]],
            weights=df['{}_{}'.format(new_weight,i)]*df[total_weight]/df[nominal_weight]).bin_counts.flatten() for df in dfs])
            varMatrix.append(row)
        varMatrix=np.array(varMatrix)
        covMatrix=np.matmul((varMatrix-hnom).T,(varMatrix-hnom))/Nvar
        self.AddGlobalCov(covMatrix)
        

    def _create_block_diag_inv_corr_mat(self):
        if len(self.global_covs) == 0:
            inv_corr_mats = [template.inv_corr_mat() for template
                         in self.templates.values()]
            self._inv_corr = block_diag(*inv_corr_mats)
        else:
            cov_mats = [template.cov_mat for template
                         in self.templates.values()]
            global_cov=np.sum(np.array(self.global_covs),axis=0)
            local_cov= block_diag(*cov_mats)
            global_cov=global_cov*(local_cov==0)
            total_corr=cov2corr(global_cov+local_cov)
            self._inv_corr = np.linalg.inv(total_corr)


    def _create_block_diag_con_corr_mat(self):
        self.confunc=self._con_term
        self.inverseconcov = block_diag(*self.inverseconcov)

    def getConstraintVector(self):
        return(self.convalue)

    def getConstraintNames(self):
        return(self.connames)

    def getConstraintCovariance(self):
        return(block_diag(*self.concov))


    @jit(forceobj=True)
    def _nocon_term(self):
        return 0.

    @jit(forceobj=True)
    def _con_term(self):
        conpars = self.params.getParametersbyIndex(self.conindices)
        v = conpars - self.convalue
        return(v @ self.inverseconcov @ v)

    @jit(forceobj=True)
    def _gauss_term(self, bin_pars):
        return (bin_pars @ self._inv_corr @ bin_pars)



    @jit(forceobj=True)
    def chi2(self, pars):
        self.params.setParameters(pars)
        yields = self.params.getParametersbyIndex(self.yieldindices).reshape(self.num_templates, 1)
        sub_pars = self.params.getParametersbyIndex(self.subfraction_indices).reshape(self.num_fractions, 1)
        bin_pars = self.params.getParametersbySlice(self.bin_par_slice)
        sys_pars = self.params.getParametersbyIndex(self.sys_par_indices)
        chi2 = self.chi2_compute(bin_pars, yields, sub_pars, sys_pars)
        return(chi2)

    @jit(forceobj=True)
    def chi2_compute(self, bin_pars, yields, sub_pars, sys_pars):
        chi2data = np.sum((self.ExpectedEventsPerBin(bin_pars.reshape(self.shape), yields, sub_pars, sys_pars) - self.xobs) ** 2 / (2 * self.xobserrors**2))
        chi2 = chi2data + self._gauss_term(bin_pars) + self.confunc()
        return(chi2)
    
    @jit(forceobj=True)
    def NLL(self, pars):
        self.params.setParameters(pars)
        sys_pars = self.params.getParametersbyIndex(self.sys_par_indices)
        yields = self.params.getParametersbyIndex(self.yieldindices).reshape(self.num_templates, 1)
        sub_pars = self.params.getParametersbyIndex(self.subfraction_indices).reshape(self.num_fractions, 1)
        bin_pars = self.params.getParametersbySlice(self.bin_par_slice)

        exp_evts_per_bin = self.ExpectedEventsPerBin(bin_pars.reshape(self.shape), yields, sub_pars, sys_pars)
        poisson_term = np.sum(exp_evts_per_bin - self.xobs - xlogyx(self.xobs, exp_evts_per_bin))
        NLL = poisson_term + (self._gauss_term(bin_pars) + self.confunc()) / 2. + np.sum(sys_pars**2)/2.
        print('NLL',NLL)
        return(NLL)


    @staticmethod
    def _get_projection(ax, bc):
        x_to_i = {
            "x": 1,
            "y": 0
        }

        return np.sum(bc, axis=x_to_i[ax])

    def plot_stacked_on(self, ax,All=False, customlabels=None, postfiterrors=np.array([]), **kwargs):

        bin_mids = [template.bin_mids() for template in self.plottemplates.values()]
        bin_edges = next(iter(self.templates.values())).bin_edges()
        bin_width = next(iter(self.templates.values())).bin_widths()
        num_bins = next(iter(self.templates.values())).num_bins
        shape = next(iter(self.templates.values())).shape()

        colors = [template.color for template in self.plottemplates.values()]
        allyields = self.params.getParametersbyIndex(self.yieldindices).reshape(self.num_templates,1)
        yields = self.params.getParametersbyIndex(self.plotyieldindices)
        sub_pars = self.params.getParametersbyIndex(self.subfraction_indices).reshape(self.num_fractions,1)
        bin_pars = self.params.getParametersbySlice(self.bin_par_slice).reshape(self.shape)
        sub_fractions = np.matmul(self.convertermatrix,sub_pars) + self.convertervector
        sys_pars = self.params.getParametersbyIndex(self.sys_par_indices)
        singlepar_corr = np.prod(1 + sys_pars * (sys_pars > 0) * self.up_errors +
                                 sys_pars * (sys_pars < 0) * self.down_errors, 0)
        corrections = (1 + self.template_errors * bin_pars) * (singlepar_corr)
        sub_fractions = np.matmul(self.convertermatrix,sub_pars) + self.convertervector
        pdfs = self.template_fractions*corrections
        norm_pdfs = pdfs/np.sum(pdfs,1)[:,np.newaxis]
        expected_bin_counts = self.ExpectedEventsPerBin(bin_pars,allyields,sub_pars, sys_pars)

        bin_counts = [tempyield*template.fractions() for tempyield,template in zip(yields,self.plottemplates.values())]
        if customlabels==None:
            labels = [template.name for template in self.plottemplates.values()]
        else:
            labels = [customlabels[key] for key in self.plottemplates.keys()]

        if(All):
          colors=[]
          for template in self.templates.values():
              colors += template.colors()
          if customlabels==None:
              labels=[]
              for template in self.templates.values():
                  labels += template.labels()
          else:
              labels = [customlabels[key] for key in self.templates.keys()]
              
          bin_counts = [tempyield*template.allfractions() for tempyield,template in zip(yields,self.plottemplates.values())]
          bin_counts = np.concatenate(bin_counts)
          N = len(bin_counts)
          bin_counts = np.split(bin_counts, N/num_bins)
          bin_mids = [bin_mids[0]]*int(N/num_bins)

        if self._dim > 1:
            bin_counts = [self._get_projection(kwargs["projection"], bc.reshape(shape)) for bc
                          in bin_counts]
            axis = kwargs["projection"]
            ax_to_index = {
                "x": 0,
                "y": 1,
            }
            bin_mids = [mids[ax_to_index[axis]] for mids in bin_mids]
            bin_edges = bin_edges[ax_to_index[axis]]
            bin_width = bin_width[ax_to_index[axis]]

        ax[0].hist(
            bin_mids,
            weights=bin_counts,
            bins=bin_edges,
            edgecolor="black",
            histtype="stepfilled",
            lw=0.5,
            color=colors,
            label=labels,
            stacked=True
        )

        #uncertainties_sq = [ (tempyield*template.fractions()*template.errors()).reshape(template.shape())** 2 for tempyield,template in
        #                    zip(yields,self.plottemplates.values())]

        if len(postfiterrors) == 0:
            uncertainties_sq = (allyields*sub_fractions*norm_pdfs*self.template_errors)**2
        else:
            binparpostfiterrs=postfiterrors[self.bin_par_slice[0]:self.bin_par_slice[1]]
            binparpostfiterrs=binparpostfiterrs.reshape(self.template_errors.shape)
            uncertainties_sq = (allyields*sub_fractions*norm_pdfs*binparpostfiterrs*self.template_errors)**2

        if self._dim > 1:
            uncertainties_sq = [
                self._get_projection(kwargs["projection"], unc_sq) for unc_sq in uncertainties_sq
            ]

        total_uncertainty = np.sqrt(np.sum(uncertainties_sq, axis=0))
        total_bin_count = np.sum(np.array(bin_counts), axis=0)

        ax[0].bar(
            x=bin_mids[0],
            height=2 * total_uncertainty,
            width=bin_width,
            bottom=total_bin_count - total_uncertainty,
            color="black",
            hatch="///////",
            fill=False,
            lw=0,
            label="MC Uncertainty"
        )

        if self.data is None:
            return ax

        data_bin_mids = self.data.bin_mids
        data_bin_counts = self.data.bin_counts
        data_bin_errors_sq = self.data.bin_errors_sq

        if self.has_data:

            if self._dim > 1:
                data_bin_counts = self._get_projection(
                    kwargs["projection"], data_bin_counts
                )
                data_bin_errors_sq = self._get_projection(
                    kwargs["projection"], data_bin_errors_sq
                )

                axis = kwargs["projection"]
                ax_to_index = {
                    "x": 0,
                    "y": 1,
                }
                data_bin_mids = data_bin_mids[ax_to_index[axis]]

            ax[0].errorbar(x=data_bin_mids, y=data_bin_counts, yerr=np.sqrt(data_bin_errors_sq),
                        ls="", marker=".", color="black", label="Data")

            total_error = np.sqrt(data_bin_errors_sq+total_uncertainty**2)
            pulls = (data_bin_counts - expected_bin_counts)/total_error
            ax[1].set_ylim((-3, 3))
            ax[1].axhline(y=0, color='dimgray', alpha=0.8)
            ax[1].errorbar(data_bin_mids, pulls, yerr=1.,
                         ls="", marker=".", color='black')



    def create_nll(self):
        """

        Returns
        -------

        """
        return NLLCostFunction(self,self.params)

    def create_chi2(self):
        """

        Returns
        -------

        """
        return Chi2CostFunction(self,self.params)

class AbstractTemplateCostFunction(ABC):
    """Abstract base class for all cost function to estimate
    yields using the template method.
    """

    def __init__(self):
        pass
    # -- abstract properties

    @property
    @abstractmethod
    def x0(self):
        """numpy.ndarray: Starting values for the minimization."""
        pass

    @property
    @abstractmethod
    def param_names(self):
        """list of str: Parameter names. Used for convenience."""
        pass

    # -- abstract methods --

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        pass


class Chi2CostFunction(AbstractTemplateCostFunction):

    def __init__(self, model: ModelBuilder, params: ParameterHandler):
        super().__init__()
        self._model = model
        self._params = params

    @property
    def x0(self):
        return(self._params.getParameters())

    @property
    def param_names(self):
        return(self._params.getParameterNames())

    def __call__(self, x):
        return(self._model.chi2(x))

class NLLCostFunction(AbstractTemplateCostFunction):

    def __init__(self, model: ModelBuilder, params: ParameterHandler):
        super().__init__()
        self._model = model
        self._params = params

    @property
    def x0(self):
        return(self._params.getParameters())

    @property
    def param_names(self):
        return(self._params.getParameterNames())

    def __call__(self, x):
        return(self._model.NLL(x))

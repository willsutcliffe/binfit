# binfit

binfit is an binned maximum likelihood fitting framework implemented using vectorized calculation in numpy. Systematic uncertainties are handled by bin nuisance parameters for each fit template.

## Installation 

The package can be installed with pip in direcotry with setup.py:
```
pip install ./

```
or
```
python3 -m pip install ./

```

## Example

Initialize histograms
```
import pandas as pd
import numpy as np
from binfit import Hist1d
from binfit import Hist2d



df_umatch = pd.read_pickle('./ulnu.pickle')
dfD = pd.read_pickle('./D.pickle')
dfDst = pd.read_pickle('./Dst.pickle')
dfDstst = pd.read_pickle('./Dstst.pickle')

var= 'gx_m'

var_binning = np.array([0.,1.6, 1.9, 2.3, 2.5, 2.8])
binrange = (var_binning[0],var_binning[-1])

hsig = Hist1d(bins=var_binning, range=binrange, data=df_umatch[var], weights=df_umatch['tot_w_0'])
hD = Hist1d(bins=var_binning, range=binrange, data=dfD[var], weights=dfD['tot_w_0'])
hDst = Hist1d(bins=var_binning, range=binrange, data=dfDst[var], weights=dfDst['tot_w_0'])
hDstst = Hist1d(bins=var_binning, range=binrange, data=dfDstst[var], weights=dfDstst['tot_w_0'])
dftot = df_umatch.append([dfD,dfDst,dfDstst])
htot = Hist1d(bins=var_binning, range=binrange, data=dftot[var], weights=dftot['tot_w_0'])
```

Define the model builder
```
import binfit
from binfit import Template1d
from binfit import Template2d
from binfit.fitting import BinFitter
from binfit.parameters.parametershandler import ParameterHandler
from binfit.models.modelbuilder  import ModelBuilder
from binfit.templates.multitemplate import MultiNormTemplate


# container for all parameters
params = ParameterHandler()

#initialise templates
tsig = Template1d('ulnu',var, hsig,params,'indianred')
tD= Template1d('D',var, hD,params,'navy')
tDst= Template1d('Dst',var, hDst,params,'orange')
tDstst= Template1d('Dstst',var, hDstst,params,'olivedrab')

# mMke a multitemplate for clnu
ctemps = {'D':tD,'Dst':tDst,'Dstst':tDstst}
pars = {'D':0.3,'Dst':0.5}
# Make a Multinorm template this requires N -1 fractions
tbkg = MultiNormTemplate('clnu', ctemps,params,pars,color = 'lightskyblue')

# define data histogram
hdata = htot

# Define model builder
nmodel = ModelBuilder(params,hdata)
nmodel.AddTemplate(tsig, 1000.)
nmodel.AddTemplate(tbkg, 4000.)

# initialisation
nmodel._create_block_diag_inv_corr_mat()
nmodel.TemplateMatrix()
nmodel.RelativeErrorMatrix()
nmodel.InitialiseBinPars()
nmodel.FractionConverter()

```

Do the a fit:
```
fitter = BinFitter(nmodel,'iminuit')
fitter.do_fit()
```



Plot the fit:
```
from matplotlib import rc
%matplotlib inline

## for Palatino and other serif fonts use:
rc('font',**{'size':17,'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import matplotlib.pyplot as plt
fig, axis = plt.subplots(2, 1, figsize=(5, 5), dpi=200, sharex=True, gridspec_kw={"height_ratios": [3.5, 1]})

nmodel.plot_stacked_on(axis,All='True')

axis[0].legend(loc='upper right',fontsize=10)
axis[0].set_ylim(top=2600)
axis[0].set_xlim(right=4.5)
axis[0].set_xlabel(r'$M_{X}$')
plt.savefig('MX_all.pdf')
```

## Fitting procedure

The number of expected events is given by 
\begin{equation}
\begin{split}
         \nu^{\rm exp}_{i}(\nu^{j},\theta^{j}_{i}) = \sum_{j} \nu^{j}  p^{j}_{i}
\end{split} 
where $p^{j}_{i}$ is the probability that an event from template, $j$, ends up in bin $i$. Meanwhile, $\nu^{j}$ are yields, which are determined by the fit.

The discrete pdfs are subject to variations from nuisance parameters, $\theta^{j}_{i}$, of the form,
\begin{equation}
\begin{split}
        p^{j}_{i} \rightarrow  \frac{ p^{j}_{i}(1 + \epsilon^{j}_{i} \theta^{j}_{i} )}{  \sum_{k}  p^{j}_{k} (1 + \epsilon^{j}_{k} \theta^{j}_{k} ) }\; ,
\end{split} 
\end{equation}
which account for both MC template statistics and additional systematic effects. The associated bin to bin correlations within and across templates, which result from systematic uncertainties, are accounted for in the correlation matrix, $\rho_{\theta}$. This correlation matrix is determined from the total covariance matrix, which is the combined sum of the covariance matrices for each systematic effect  considered. 


The model is fitted to data by minimising the following $-2 \log \mathcal{L}$,
\begin{equation}
\begin{split}
	-2 \log \mathcal{L}  =  -2 \log \prod_{i} {\rm Poisson}(\nu^{\rm obs}_{i}, \nu^{\rm exp}_{i})  + \theta^{T} \rho^{-1}_{\theta} \theta^{T}  + (k - k_{constraint})^{T} \Sigma^{-1}_{\rm constraints} (k - k_{constraint})
\end{split} 
\end{equation}
where $\nu^{\rm obs}_{i}$ is the number of events observed in a given bin $i$. 


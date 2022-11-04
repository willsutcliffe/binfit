import pandas as pd
import numpy as np
import binfit
from binfit import Hist1d
from binfit import Hist2d
from binfit import Template1d
from binfit import Template2d
from binfit.fitting import BinFitter
from binfit.parameters.parametershandler import ParameterHandler

from binfit.models.modelbuilder  import ModelBuilder
from binfit.templates.multitemplate import MultiNormTemplate



if __name__ == "__main__":
    df_umatch = pd.read_pickle('./ulnu.pickle')
    dfD = pd.read_pickle('./D.pickle')
    dfDst = pd.read_pickle('./Dst.pickle')
    dfDstst = pd.read_pickle('./Dstst.pickle')

    var= 'gx_m'
    #var_binning = np.array([0., 1.6, 1.9, 2.3, 2.5, 2.8, 3.1, 3.4, 3.7, 4.2, 5.0])
    var_binning = np.array([0.,1.6, 1.9, 2.3, 2.5, 2.8])
    #var_binning = np.linspace(0.,2.8,20)
    binrange = (var_binning[0],var_binning[-1])

    hsig = Hist1d(bins=var_binning, range=binrange, data=df_umatch[var], weights=df_umatch['tot_w_0'])
    hD = Hist1d(bins=var_binning, range=binrange, data=dfD[var], weights=dfD['tot_w_0'])
    hDst = Hist1d(bins=var_binning, range=binrange, data=dfDst[var], weights=dfDst['tot_w_0'])
    hDstst = Hist1d(bins=var_binning, range=binrange, data=dfDstst[var], weights=dfDstst['tot_w_0'])
    dftot = df_umatch.append([dfD,dfDst,dfDstst])
    htot = Hist1d(bins=var_binning, range=binrange, data=dftot[var], weights=dftot['tot_w_0'])

    # Run for 2D fitting

    var2='event_q2'
    #var2_binning = np.array([0., 2, 4, 6, 8, 10, 12, 14, 26])
    var2_binning = np.array([0., 2, 4, 6, 8])
    hsig2d = Hist2d(bins=[var_binning,var2_binning],data=[df_umatch[var],df_umatch[var2]], weights=df_umatch['tot_w_0'])
    hD2d = Hist2d(bins=[var_binning,var2_binning], range=binrange, data=[dfD[var],dfD[var2]], weights=dfD['tot_w_0'])
    hDst2d = Hist2d(bins=[var_binning,var2_binning], range=binrange, data=[dfDst[var],dfDst[var2]], weights=dfDst['tot_w_0'])
    hDstst2d = Hist2d(bins=[var_binning,var2_binning], range=binrange, data=[dfDstst[var],dfDstst[var2]], weights=dfDstst['tot_w_0'])
    htot2D = Hist2d(bins=[var_binning,var2_binning], range=binrange, data=[dftot[var],dftot[var2]], weights=dftot['tot_w_0'])

    from scipy.optimize import minimize
    # container for all parameters
    params = ParameterHandler()

    tsig = Template1d('ulnu',var, hsig,params,'indianred')
    tD= Template1d('D',var, hD,params,'navy')
    tDst= Template1d('Dst',var, hDst,params,'orange')
    tDstst= Template1d('Dstst',var, hDstst,params,'olivedrab')

    # 2D case
    #tsig = Template2d('ulnu',var, hsig2d,params,'indianred')
    #tD= Template2d('D',var, hD2d,params,'navy')
    #tDst= Template2d('Dst',var, hDst2d,params,'orange')
    #tDstst= Template2d('Dstst',var, hDstst2d,params,'olivedrab')

    ctemps = {'D':tD,'Dst':tDst,'Dstst':tDstst}
    pars = {'D':0.3,'Dst':0.5}
    # Make a Multinorm template this requires N -1 fractions
    tbkg = MultiNormTemplate('clnu', ctemps,params,pars,color = 'lightskyblue')

    #hdata = h.from_binned_data(data,var_binning)
    hdata = htot
    # 2D case
    #hdata = htot2D


    # here the classic variation scheme of max is used where a new covariance matrix is computed from variations
    # and added with the original diagonal covariance according to max' implementation. This functions
    #
    #tD.add_variation(dfD[var],dfD['Dlnu_FF_downweight0']*dfD['tot_w_0']/dfD['Dlnu_FF_weight'],dfD['Dlnu_FF_upweight0']*dfD['tot_w_0']/dfD['Dlnu_FF_weight'])
    #tD.add_variation([dfD[var],dfD[var2]],dfD['Dlnu_FF_downweight0'],dfD['Dlnu_FF_upweight0'])
    #[dftot[var],dftot[var2]]

    # add a single parameter variation not working 100% yet
    # here the up and down variations are stored in the template  and used
    #tDst.add_singlepar_variation(dfDst[var],dfDst['Dlnu_FF_downweight1'],dfDst['Dlnu_FF_upweight1'],'Dlnu_FF_weight')
    #tD.add_singlepar_variation(dfD[var],dfD['Dlnu_FF_downweight1'],dfD['Dlnu_FF_upweight1'],'Dlnu_FF_weight')
    #tDstst.add_singlepar_variation(dfDstst[var],dfDstst['Dlnu_FF_downweight1'],dfDstst['Dlnu_FF_upweight1'],'Dlnu_FF_weight')
    #tsig.add_singlepar_variation(df_umatch[var],df_umatch['Dlnu_FF_downweight1'],df_umatch['Dlnu_FF_upweight1'],'Dlnu_FF_weight')
    #tD.add_singlepar_variation([dfD[var],dfD[var2]],dfD['Dlnu_FF_downweight1'],dfD['Dlnu_FF_upweight1'],'Dlnu_FF_1')


    nmodel = ModelBuilder(params,hdata)
    # add templates, each of which have a yield
    # the capability of sharing yields will be possible
    # note that it can be a base or multitemplate
    # later I plan to make an abstract model and have TemplateModel and ChannelModel as channels have multiple datasets

    nmodel.AddTemplate(tsig, 1000.)
    nmodel.AddTemplate(tbkg, 4000.)

    dfs = {'D' : dfD}
    nmodel.add_singlepar_variation(var,'sys_Dlnu_FF', dfs,'Dlnu_FF_downweight1','Dlnu_FF_upweight1')


    # add a gaussian constraint on a given parameter
    # e.g here we constrain
    #nmodel.AddConstraint('fraction_D',0.03,0.0001)

    # initialise the block diagonal covariance for all bin paramaters
    # this is again in line with what max does
    nmodel._create_block_diag_inv_corr_mat()
    nmodel.TemplateMatrix()
    nmodel.RelativeErrorMatrix()
    nmodel.InitialiseBinPars()
    nmodel.FractionConverter()
    nmodel.StoreSysPars()




    fitter = BinFitter(nmodel,'iminuit')

    result=fitter.do_fit()

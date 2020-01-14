# class which contains parameters
import numpy as np


__all__ = ["ParameterHandler"]

class ParameterHandler:
    """ The class provides an interface for registering and storing parameters """
    def __init__(self):
        """ Initalise parameter array and dictionary"""
        # pars used for registration
        self._pars = []
        # access provided by npars
        self._npars = np.array([])
        self._parsdict = {}
        
    def addParameters(self, pars,names):
        indices = []
        for key, value in zip(names,pars):
            index = self.addParameter(key,value,update='False')
            indices.append(index)
        self._npars = np.array(self._pars)    
        return(indices)

    def addParameter(self,name,parameter,update='True'):
        self._pars += [parameter]
        yieldindex = len(self._pars) - 1
        self._parsdict[name] = yieldindex
        if update:
            self._npars = np.array(self._pars)    
        return(yieldindex)

    def getParametersbySlice(self,slicing):
        return(self._npars[slicing[0]:slicing[1]])

    def getParametersbyIndex(self,indices):
        return(self._npars[indices])

    def getParameterNames(self):
        return(list(self._parsdict.keys()))

    def getParameters(self):
        return(self._npars)

    def setParameters(self,pars):
        self._npars[:] = pars

    def getIndex(self,name):
        return(self._parsdict[name])




'''
 Dirichlet Distribution

'''

from scipy.special import gammaln
from numpy import log, exp, longdouble



class DirichletDistribution (object):

    def __init__ (self):
        self.alpha = None
    

    ''' eval distribution '''
    
    # log-likelihood p(data|alpha)
    def logll (self, d):
        self.__check_data (d)
        sump = log(d).sum(axis=0)
        n = d.shape[0]
        nfact = (gammaln(self.alpha.sum()) - gammaln(self.alpha).sum())
        return n * nfact + ((self.alpha - 1) * sump).sum()


    # pdf (i.e., Dir(d)) as n-array of the data [n * k], i.e., one value per input vector
    def pdf (self, d):
        self.__check_data (d)
        nfact = gammaln(self.alpha.sum()) - gammaln(self.alpha).sum()
        a = self.alpha - 1
        return exp((nfact + (a * d).sum(axis=1)).astype(longdouble))


    ''' private function '''

    # check if data are corret before estimating pdf
    def __check_data (self, d):
        if d is None:
            raise Exception ('no data to evaluate')
        if self.alpha is None:
            raise Exception ('alpha parameters not estimated yet')
        if d.shape[1] != self.alpha.shape[0]:
            raise Exception ('input data and alpha coefficients with different dimension')
        return
        

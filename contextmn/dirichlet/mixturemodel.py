'''
 Dirichlet Mixture Model

 @author: Riccardo Miotto
'''

from scipy.special import psi, polygamma, gammaln, gamma
from numpy import log, array, median, zeros, exp, transpose, longdouble
from numpy.random import random
from distribution import DirichletDistribution
import sys


class DMM (DirichletDistribution):

    def __init__ (self, nmix = 4):
        DirichletDistribution.__init__(self)
        self.nmix = nmix
        self.beta = None


    # parameter estimation using generalized EM
    def training (self, d, preproc = True, maxiter = None):
        if maxiter is None:
            maxiter = sys.maxint
        if preproc:
            d = self.__preprocessing (d)
        try:
            self.__gem (d, maxiter)
        except Exception as e:
            print ' --- %s' % e
            return False
        return True


    # pdf as n-array of the data [n * k], i.e., one value per input vector
    def estimate (self, data):
        return self.pdf (data)


    # eval pdf of the mixture model
    def pdf (self, d, b = None, a = None):
        if b is None:
            b = self.beta
        if a is None:
            a = self.alpha
        nfact = gammaln(a.sum(axis=1)) - gammaln(a).sum(axis=1)
        pdfval = array ([log(b[k]) + nfact[k] + ((a[k] - 1) * log(d)).sum(axis=1) for k in xrange(self.nmix)], dtype=longdouble)
        return exp(pdfval).sum(axis=0)


    
    ''' private functions '''

    # generalized EM algorithm for parameter estimation
    def __gem (self, d, maxiter):
        (b, a) = self.__init_param (d)
        n = d.shape[0]
        if self.nmix == 1:
            nconv = 1
        else:
            nconv = self.nmix - 1
        isconv = array([0 for i in xrange(self.nmix)])
        for i in xrange(maxiter):
            # expectation
            e = self.__eval_expectation (d, b, a)
            # maximization
            b = e.mean(axis=0)
            b /= b.sum()
            for k in xrange(self.nmix):
                if isconv[k] == 1:
                    continue
                aknew = self.__newton (d, a[k], e[:,k])
                # likelihood test
                aknew[(aknew < 10e-10)] = 10e-10
                if max(abs(aknew - a[k])) < 1e-7:
                    isconv[k] = 1
                    print ' --- mixture %d has converged' % (k+1)
                a[k] = aknew
            if (isconv.sum() >= nconv):
                break
        # store data
        self.alpha = a
        self.beta = b
        if (isconv.sum() < nconv):
            raise Exception ('failed to converge after %d iterations - stored current values' % maxiter)
        return


    # compute mixture contributes (expectation)
    def __eval_expectation (self, d, b, a):
        e = array([b[i] * self.__eval_pdf(d,a[i]) for i in xrange(self.nmix)]).transpose()
        dnorm = e.sum(axis = 1)
        e /= dnorm[:,None]
        return e.astype(float)

    
    # pdf of one mixture
    def __eval_pdf (self, d, a):
        nfact = gammaln(a.sum()) - gammaln(a).sum()
        pdfll = nfact + ((a - 1) * log(d)).sum(axis=1)
        return exp(pdfll.astype(longdouble))


    # newton iteration to estimate new alpha parameters for one mixture
    def __newton (self, d, ak, ek):
        if ak.sum() == 0:
            return ak
        sump = (log(d) * ek[:,None]).sum(axis=0)
        sumek = ek.sum()
        n = d.shape[0]
        a = ak
        # init expectation
        e = self.__sump2logll (sump, a, sumek, n)
        lmbd = float(0.1)        
        g = sumek * (psi(a.sum()) - psi(a)) + sump
        # loop until the hessian matrix is nonsingular
        while 1:
            hg = self.__compute_hg (a, g, sumek, lmbd);
            if all(hg < a):
                enew = self.__sump2logll (sump, a-hg, sumek, n)
                if enew > e:
                    return a - hg
            lmbd *= 10
            if lmbd > 10e+7:
                break
        return a


    # compute the hessian time gradient vector
    def __compute_hg (self, a, g, sume, lmbd):
        q = 1 / (-polygamma(1,a) * sume - lmbd)
        z = -polygamma(1,a.sum()) * sume
        b = (g * q).sum() / (1/z + q.sum())
        hg = (g - b) * q
        return hg


    # ensure no 0s are in the data
    def __preprocessing (self, data):
        izero = (data == 0)
        data[izero] = 10e-7
        data /= data.sum(axis=1)[:,None]
        return data


    # initialize parameters
    def __init_param (self, d):
        # init beta
        b = random (self.nmix)
        b = b / b.sum()
        # init alpha
        e = d.mean(axis=0)
        e2 = (d**2).mean(axis=0)
        i = (e > 0)
        a = zeros ((self.nmix, d.shape[1]))
        a[:] = e * median((e[i] - e2[i]) / (e2[i] - e2[i]**2))
        return (b, a)
        

    # single loglikelihood from "sump"
    def __sump2logll (self, sump, a, sume, n):
        return n * sume * (gammaln(a.sum()) - gammaln(a).sum()) + ((a - 1) * sump).sum()


'''
Single Dirichlet Distribution

'''

from scipy.special import psi, polygamma, gammaln, gamma
from numpy import log, array, median, zeros, exp
from distribution import DirichletDistribution
import sys


class DirichletModel (DirichletDistribution):

    def __init__ (self):
        super(DirichletDistribution, self).__init__()


    # parameter estimation
    def training (self, d, method='newton', maxiter=None):
        d = self.__preprocessing (d)
        if method == 'fixedpoint':
            self.__fixedpoint (d, maxiter)
        else:
            self.__newton (d, maxiter)
        return


    # pdf as n-array of the data [n * k], i.e., one value per input vector
    def estimate (self, d):
        return self.pdf(d)


    
    ''' private functions '''

    # newton iteration to learn the distribution parameters
    def __newton (self, d, maxiter):
        sump = log(d).sum(axis=0)
        n = d.shape[0]

        # init alpha coefficients
        a = self.__init_alpha (d)

        # init expectation
        e = self.__sump2logll (sump, a, n)
        lmbd = float(0.1)
        if maxiter is None:
            maxiter = sys.maxint

        # iterate
        for i in xrange(maxiter):
            if a.sum() == 0:
                break
            g = n * (psi(a.sum()) - psi(a)) + sump
            # loop until the hessian matrix is nonsingular
            while 1:
                hg = self.__compute_hg (a, g, lmbd);
                if all(hg < a):
                    enew = self.__sump2logll (sump, a-hg, n)
                    if enew > e:
                        e = enew
                        anew = a - hg
                        lmbd /= 10
                        break
                lmbd *= 10
                if lmbd > 10e+7:
                    break
            # likelihood test
            anew[(anew < 2e-52)] = 2e-52
            if max(abs(anew - a)) < 1e-7:
                self.alpha = anew
                return
            a = anew
        self.alpha = a
        raise Exception ('failed to converge after %d iterations' % maxiter)

            
    # fixed point iteration to learn the dmm parameters
    def __fixedpoint (self, d, maxiter):
        sump = log(d).sum(axis=0)
        n = d.shape[0]

        # init alpha coefficients
        a = self.__init_alpha (d)
        if maxiter is None:
            maxiter = sys.maxint

        # iterate
        e = self.__sump2logll (sump, a, n)
        for i in xrange(maxiter):
            anew = self.__ipsi (psi(a.sum()) + sump/n)
            enew = self.__sump2logll (sump, anew, n)
            if abs(enew - e) < 10e-7:
                self.alpha = anew
                return True
            a = anew
            e = enew
        self.alpha = a
        raise Exception ('failed to converge after %d iterations' % maxiter)
            

    # invert the digamma function (psi): Newton iteration to solve psi(x) - y = 0
    def __ipsi (self, y):
        x = zeros(y.shape[0], dtype=float)

        # init
        for i in xrange(y.shape[0]):
            if y[i] >= -2.22:
                x[i] = exp(y[i]) + 0.5
            else:
                x[i] = -1 / (y[i] - psi(1))

        # iterate - no more than 5 iterations with enough training data
        for i in xrange(sys.maxint):
            xnew = x - ((psi(x) - y) / polygamma(1,x))
            if max(abs(xnew - x)) < 10e-7:
                return xnew
            x = xnew
        return x

    
    # compute the hessian time gradient vector
    def __compute_hg (self, a, g, lmbd):
        q = 1 / (-polygamma(1,a) - lmbd)
        z = polygamma(1,a.sum())
        b = (g * q).sum() / (1/z + q.sum())
        hg = (g - b) * q
        return hg


    # ensure no 0s are in the data
    def __preprocessing (self, data):
        izero = (data == 0)
        data[izero] = 10e-7
        for i in xrange(data.shape[0]):
            data[i] /= data[i].sum()
        return data


    # initialize alpha coefficients
    def __init_alpha (self, d):
        e = d.mean(axis=0)
        e2 = (d**2).mean(axis=0)
        i = (e > 0)
        return e * median((e[i] - e2[i]) / (e2[i] - e2[i]**2))


    # loglikelihood from "sump"
    def __sump2logll (self, sump, a, n):
        return n * (gammaln(a.sum()) - gammaln(a).sum()) + ((a - 1) * sump).sum()

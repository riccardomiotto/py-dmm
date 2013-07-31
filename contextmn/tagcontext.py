'''
 Define, train, and use a Tag Context Model

 @input SMN: semantic multinomials as N vectors of size V = (no. of tags in the vocabulary)
 
 @author: Riccardo Miotto
'''

from dirichlet.mixturemodel import DMM
from numpy import log, exp, argsort, ceil
from scipy.stats import kurtosis
import cPickle


class TagContextModel (object):

    def __init__ (self, tag, nmix = 4):
        self.tag = tag
        self.mcontext = DMM(nmix)


    '''
    train the context model

    @input tmsn: SMNs of the songs (multiple SMNs per song) positively associated with the tag in the ground truth
    @input kpreproc: to pre-process the data using the kurtosis analysis
    '''
    def train (self, tsmn, kpreproc = True):
        print 'learning context model for tag: %s' % self.tag
        print ' --- using %d mixtures' % self.mcontext.nmix
        print ' --- using %d training SMNs' % tsmn.shape[0]
        if kpreproc:
            tsmn = self.__kpreproc (tsmn)
            print ' --- kurtosis-based pre-processed data' 
        if self.mcontext.training (tsmn, False):
            print ' --- model learned correctly'
        return


    '''
    predict tag weight for a set of semantic multinomials

    @input smn: SMNs of the input song (multiple SMNs per song by windowing the song)
    @output: return geometric mean of the predictions
    '''
    def predict (self, smn):
        return exp(self.loglikelihood(smn))


    '''
    prediction in log base

    @output: prediciton in log-version
    '''
    def loglikelihood (self, smn):
        p = self.mcontext.estimate (smn)
        return log(p).mean()


    # save the model
    def save (self, filename):
        try:
            cPickle.dump(self, open(filename, 'wb'))
	    return True
	except Exception as e:
            print e
	    return False

        
    ''' private functions '''

    # pre-process training data using kurtosis
    # NOTE: can "remove max(3, ..)" when the vocabulary is larger (add it just for small vocabularies, to retain some peaks)
    def __kpreproc (self, smn):
        k = kurtosis (smn, axis=1, fisher=False)
        meank = k.mean()
        sind = argsort (smn, axis = 1)[:,::-1]
        for i in xrange(smn.shape[0]):
            if k[i] > meank:
                npeak = max (3, ceil(0.05 * smn.shape[1]))
            else:
                npeak = max (3, ceil(0.1 * smn.shape[1]))
            smn[i][sind[i][npeak:]] = min(smn[i])
            smn[i] /= smn[i].sum()
        return smn


'''
 Experimetal framework using the TagContextModels to predict CMNs (context multinomials)

 @Note-1: adapt the code to load real data (random SMNs now)

 @Note-2: no evaluation infrastructure

'''
 

from contextmn.tagcontext import TagContextModel
from contextmn.dirichlet.model import DirichletModel
import numpy as np
import os


# parameters
vocab = sorted(['rock', 'pop', 'jazz', 'classic', 'punk'])
ntrain_smn = 500
ntest_smn = 50
ntags = len(vocab)
nmix = 4
dout = './data'

# create output directory
try:
    os.makedirs(dout)
except OSError:
    pass
except Exception as e:
    print e


# train tag context models
tagmodels = []
print ''
for t in sorted(vocab):
    m = TagContextModel (t, nmix)

    # generate training fake data (substitute with real SMNs <ntrain_smn * ntags>)
    # NOTE: 'col1' refers to 'classic', 'col2' refers to 'jazz', etc.
    smn = np.random.random ([ntrain_smn, ntags])
    smn /= smn.sum(axis=1)[:,None]

    # train context model
    m.train (smn, True)
    print 'alpha = %s' % m.mcontext.alpha
    print 'beta = %s' % m.mcontext.beta

    # store
    tagmodels.append (m)
    m.save ('%s/%s-tag-context-model.pkl' % (dout, m.tag))
    print ''

print ''
print 'trained %d tag context models and saved in %s' % (ntags, dout)


# predict on new data
test_cmn = np.zeros([ntest_smn, ntags])
for s in xrange(ntest_smn):
    
    # generate testing fake data (substitute with real SMNs <ntest_smn * ntags>)
    # NOTE1: 'col1' refers to 'classic', 'col2' refers to 'jazz', etc.
    # NOTE2: using 100 SMNs per song (i.e., each one referring to overlapping windows along the song)
    data = np.random.random ([100, ntags])
    smn = np.array([d/d.sum() for d in data])

    # prediction
    cmn = np.zeros ([ntags])
    for i in xrange(ntags):
        cmn[i] = tagmodels[i].predict(smn)

    # normalize to sum one
    cmn = cmn / cmn.sum()

    # save
    test_cmn[s,:] = cmn

print ''
print 'estimated CMNs for %d songs' % ntest_smn
for cmn in test_cmn:
    print cmn

    
        
    
    
    
    

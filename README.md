Python Implementation of a context-based auto-tagger using the Dirichlet Mixture 
Model (DMM).

The DMM refines the predictions of audio-based auto-taggers (which tag the songs 
according to the audio signal characteristics) by modeling the contextual patterns 
among the tags in the semantic multinomials (SMNs).

Parameter estimation is based on Newton-Raphson and EM algorithms.

More details of this work are available in:

Miotto, R. and Lanckriet, G.R.G. (2012)
A Generative Context Model for Semantic Music Annotation and Retrieval 
IEEE Transactions on Audio, Speech and Language Processing
Vol. 20(4), pp. 1096-1108.

Please cite this paper if you use the code.

The package also provides functions to work with a single Dirichlet distribution; 
parameter estimation can be performed using the fixed iteration (slower) or the 
Newton–Raphson algorithm (faster).

Python Requirements: numpy, scipy

Author: Riccardo Miotto

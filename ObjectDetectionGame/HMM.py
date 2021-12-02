import warnings
import numpy as np
from hmmlearn.hmm import MultinomialHMM
import hmmlearn.hmm as hmm
from distanceFinder import *

# warnings.filterwarnings('ignore')
startprob = np.array([.5, .5])  # 0 is slow, 1 is fast - initial prob is 50-50
transmatrix = np.array([[0.7, 0.3],
                        [0.3, 0.7]])
# given slow (0), probability of slow is 0.7 (same with fast)
# given fast (1), probability of slow is 0.3
# > probability of changing from slow to fast or fast to slow is 0.3 - generalized lol
emitmatrix = np.array([[0.9, 0.1],
                       [0.2, 0.8]])
# arbitrarily assigned probabilities
# 0.9 - given < 2ft/s, 90% chance they're slow
# 0.1 - given < 2ft/s, 10% chance they're fast
# 0.2 - given > 2ft/s, 20% chance they're slow
# 0.8 - given > 2ft/s, 80% chance they're fast
# -------------------------------------------------
# states: slow or fast
# observations: difference in distance traveled/seconds (time interval)
# obs 1: < 2 ft/s
# obs 2: > 2 ft/s

h = hmm.MultinomialHMM(n_components=2, startprob_prior=startprob, transmat_prior=transmatrix)
h.emissionprob = emitmatrix

'''
Here our states are:
        slow = 0,0,0,0,0
        fast = 1,1,1,1,1
'''

X = [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
h.fit(X)
print(h, '\n')

print("Transmission Probability")
print(h.transmat_)
print("Emission Probability")
print(h.emissionprob_)

print(h.decode(np.array([0, 0, 1, 0, 0]).reshape(5, 1)))  # using viterbi algorithm
print(h.predict(np.array([0, 0, 1, 0, 0]).reshape(5, 1)))  # find most likely state sequence corresponding to X
print(h.score(X))  # evaulate probability of sequence (X)

'''
TEST #1: 

MultinomialHMM(n_components=2, random_state=RandomState(MT19937) at 0x1108E3B40,
               startprob_prior=array([0.5, 0.5]),
               transmat_prior=array([[0.7, 0.3],
       [0.3, 0.7]])) 

Transmission Probability
[[1.        0.       ]
 [0.0750042 0.9249958]]
Emission Probability
[[9.99947218e-01 5.27816559e-05]
 [7.95614637e-06 9.99992044e-01]]
(-24.181282130155786, array([1, 0, 0, 0, 0], dtype=int32))
[1 0 0 0 0]
-2.902314742277875

'''
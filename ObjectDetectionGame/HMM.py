import warnings
import numpy as np
from hmmlearn.hmm import MultinomialHMM
import hmmlearn.hmm as hmm

#warnings.filterwarnings('ignore')
startprob = np.array([.5, .5]) #0 is slow, 1 is fast - initial prob is 50-50
transmatrix = np.array([[0.7, 0.3],
                        [0.3, 0.7]])
#given slow (0), probability of slow is 0.7 (same with fast)
#given fast (1), probability of slow is 0.3
#> probability of changing from slow to fast or fast to slow is 0.3 - generalized lol
emitmatrix = np.array([[0.9, 0.1],
                       [0.2, 0.8]])
#arbitrarily assigned probabilities
#0.9 - given < 2ft/s, 90% chance they're slow
#0.1 - given < 2ft/s, 10% chance they're fast
#0.2 - given > 2ft/s, 20% chance they're slow
#0.8 - given > 2ft/s, 80% chance they're fast
#-------------------------------------------------
#states: slow or fast
#observations: difference in distance traveled/seconds (time interval)
    #obs 1: < 2 ft/s
    #obs 2: > 2 ft/s

h = hmm.MultinomialHMM(n_components=2, startprob_prior=startprob, transmatprior=transmatrix)
h.emissionprob = emitmatrix

X = [[1, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]
h.fit(X)
print(h, '\n')

print("Transmission Probability")
print(h.transmat)
print("Emission Probability")
print(h.emissionprob)

print(h.decode(np.array([0,0,1,0,0]).reshape(5,1))) #using viterbi algorithm
print(h.predict(np.array([0,0,1,0,0]).reshape(5,1))) #find most likely state sequence corresponding to X
print(h.score(X)) #evaulate probability of sequence (X)
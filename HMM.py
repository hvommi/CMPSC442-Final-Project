import numpy as np
import hmmlearn.hmm as hmm
from distanceFinder import *

class HMM:

        def __init__(self):
                self.observation_dict = {0: "slow", 1:"fast"}
                self.startprob = np.array([.5, .5])  # 0 is slow, 1 is fast - initial prob is 50-50
                self.transmatrix = np.array([[0.7, 0.3],
                                        [0.3, 0.7]])
                # given slow (0), probability of slow is 0.7 (same with fast)
                # given fast (1), probability of slow is 0.3
                # > probability of changing from slow to fast or fast to slow is 0.3
                self.emitmatrix = np.array([[0.9, 0.1], [0.2, 0.8]])
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

                self.h = hmm.MultinomialHMM(n_components=2, startprob_prior=self.startprob, transmat_prior=self.transmatrix)
                self.h.emissionprob = self.emitmatrix

                '''
                Here our states are:
                        slow = 0,0,0,0,0
                        fast = 1,1,1,1,1
                '''

        # Accepts current speed observation and predicts the next state
        def feedHMM(self, X):
                self.h.fit([[1, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]])
                if X > 2:
                        speedCategory = 1
                else:
                        speedCategory = 0
                
                # print(self.h, '\n')

                # print("Transmission Probability")
                # print(self.h.transmat_)
                # print("Emission Probability")
                # print(self.h.emissionprob_)

                # print(self.h.decode(np.array([speedCategory]).reshape(len([speedCategory]), 1)))  # using viterbi algorithm
                # print(self.h.predict(np.array([speedCategory]).reshape(len([speedCategory]), 1)))  # find most likely state sequence corresponding to X
                # print(self.h.score(np.array([speedCategory]).reshape(len([speedCategory]), 1)))  # evaulate probability of sequence (X)

                return self.h.predict([[speedCategory]])
import math

import numpy as np
from scipy.special import expit
from scipy.stats import kendalltau, rankdata


class BaseEnv:
    '''
    Generic environment class
    '''

    def __init__(self, X, y, true_model, noise_factor=1, seed=None):

        self.X = X

        self.Y = y
        self.noise_factor = noise_factor

        self.theta = true_model

        self.true_scores = self.X.dot(self.theta)
        self.true_order = rankdata(self.true_scores)

        self.random_state = np.random.RandomState(seed)

    def step(self, action):
        '''
        Take action in environment

        return: reward, aux
        '''

        # The probability of a correct comparison is simulated using
        # the "best" model which has seen all data
        probability = expit(
            self.theta.dot(self.X[action[0],
                                  :] - self.X[action[1],
                                              :]) * self.noise_factor)

        # Note, a reward of 0 is not worse than a reward of 1, not trying to maximize reward...
        if self.random_state.uniform() < probability:
            return 1, probability > 0.5
        else:
            return 0, probability < 0.5

    def get_ranking_loss(self, estimated_order):
        distance = self._kendalldist(estimated_order, self.true_order)

        if math.isnan(distance):
            # everyone item has the same rank, order is arbitrary...
            return 0.5
        else:
            return distance

    def _kendalldist(self, *args, **kwargs):
        '''Return the normalized Kendall tau distance (0 <= tau <= 1)
        This function wraps around scipy.stats.kendalltau()
        and forwards all arguments to that function.
        '''
        tau_coef, p_value = kendalltau(*args, **kwargs)
        return self._kendall_convert_coef_to_dist(tau_coef)

    def _kendall_convert_coef_to_dist(self, tau_coef):
        '''Convert kendall correlation coefficient to normalized distance'''
        return (1 - tau_coef) / 2

class RealDataEnv:
    '''
    Environment for when data is in the form of real collected comparisons between objects 
    '''

    def __init__(self, X, y, true_order, seed=None):

        self.X = X
        self.y = y
        self.true_order = true_order

    def step(self, action):
        '''
        Take action in environment

        return: reward, aux
        '''

        result = None
        for i in range(len(self.X)):
            x1, x2 = self.X[i]
            if x1 == action[0] and x2 == action[1]:
                result = self.y[i]

                self.y = np.delete(self.y, i, 0)
                self.X = np.delete(self.X, i, 0)

                break
            elif x2 == action[0] and x1 == action[1]:
                
                result = 1 - self.y[i]

                self.y = np.delete(self.y, i, 0)
                self.X = np.delete(self.X, i, 0)

                break
        
        return result, None

    def get_ranking_loss(self, estimated_order):
        distance = self._kendalldist(estimated_order, self.true_order)

        if math.isnan(distance):
            # everyone item has the same rank, order is arbitrary...
            return 0.5
        else:
            return distance

    def _kendalldist(self, *args, **kwargs):
        '''Return the normalized Kendall tau distance (0 <= tau <= 1)
        This function wraps around scipy.stats.kendalltau()
        and forwards all arguments to that function.
        '''
        tau_coef, p_value = kendalltau(*args, **kwargs)
        return self._kendall_convert_coef_to_dist(tau_coef)

    def _kendall_convert_coef_to_dist(self, tau_coef):
        '''Convert kendall correlation coefficient to normalized distance'''
        return (1 - tau_coef) / 2
    

    def add_data(self, new_comparisons , new_labels, new_order):
        self.X = np.concatenate((self.X, new_comparisons))
        self.y = np.concatenate((self.y, new_labels))
        self.true_order = new_order
        


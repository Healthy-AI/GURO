import itertools
from typing import Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from trueskill import Rating, rate_1vs1
from scipy.linalg import block_diag
import copy

PRECISION = 1e-12


class BaseAlgorithm:
    '''
    Generic algorithm for pair-wise comparisons bandits
    '''

    def __init__(self, X, update_every=10, seed=None):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.t = 0
        self.update_every = update_every
        self.obs_data = []
        self.obs_labels = []
        self.obs_indices = []
        self.random_state = np.random.RandomState(seed)
        self.theta_hat = np.ones((self.d,))

    def act(self):
        '''
        Decide which pair to compare next
        '''
        raise NotImplementedError

    def update(self, i, j, observation):
        '''
        Update the algorithm with the observation of the pair (i, j)
        '''

        self.t += 1
        x = self.X[i] - self.X[j]
        self.obs_data.append(x)
        self.obs_labels.append(observation)
        self.obs_indices.append((i, j))
        if self.t % self.update_every == 0 and self.t >= 10 and len(np.unique(self.obs_labels)) >= 2:
            self.update_model()

    def update_model(self):
        '''
        Update the model with the current observations
        '''
        raise NotImplementedError

    def ordering(self):
        '''
        Return ordering of data
        '''
        scores = self.X.dot(self.theta_hat)
        return rankdata(scores)

    def get_model_info(self):
        '''
        Return the current state of the model
        '''
        raise NotImplementedError

    def get_hess(self):

        hess = np.zeros((self.d, self.d))
        for x, _ in zip(self.obs_data, self.obs_labels):
            y_hat = expit(x.dot(self.theta_hat))
            hess += y_hat * (1 - y_hat) * np.outer(x, x)
        return hess


class UniformSampling(BaseAlgorithm):
    '''
    Uniform sampler
    '''

    def __init__(self, X, update_every=10, seed=None):
        super().__init__(X, update_every, seed)

        self.model = LogisticRegression()

    def act(self):
        return self.random_state.choice(self.n, size=2, replace=False), None

    def update_model(self):
        self.model.fit(self.obs_data, self.obs_labels)
        self.theta_hat = self.model.coef_[0]

    def get_model_info(self):
        return {
            "theta_hat": copy.copy(self.theta_hat),
            "inv_hessian": np.linalg.pinv(self.get_hess())
        }


class BayesianAlgorithm(BaseAlgorithm):

    def __init__(
            self, X, update_every=10, seed=None, personal_coefs=False, personal_var=1):
        super().__init__(X, update_every, seed)

        # These parameters define our prior distribution
        # theta ~ N(theta_map, hess^(-1))
        # inv_hess could be multiplied with coeficient to increase prior variance etc.
        lambda_ = 1
        self.inv_hess = lambda_ * np.eye(self.d)
        self.hess = np.linalg.pinv(self.inv_hess)
        self.theta_hat = np.zeros((self.d))

        self.M = np.identity(self.d)
        self.M_inv = np.linalg.pinv(self.M)

        self.personal_coefs = personal_coefs

        if self.personal_coefs:
            self.individual_coefs = []

            for _ in X:
                self.individual_coefs.append((0, personal_var))

            self.X_one_hot = np.concatenate((X, np.identity(self.n)), axis=1)
            self.full_inv_hess = block_diag(self.inv_hess, np.identity(self.n))
            self.full_theta_hat = np.zeros((self.d + self.n))

    def act(self):
        raise NotImplementedError

    def update_model(self):

        X_train = np.array(self.obs_data[-self.update_every:])
        y_train = np.array(self.obs_labels[-self.update_every:])

        temp_cov = self.inv_hess
        temp_hess = self.hess
        temp_theta = self.theta_hat

        if self.personal_coefs:
            # X_train with one_hot
            X_train, global_indices = self.add_independent_features(
                X_train, self.obs_indices[-self.update_every:])

            personal_covs = np.identity(len(global_indices))
            personal_mus = np.zeros((len(global_indices)))

            for i in range(len(global_indices)):
                coef = self.individual_coefs[global_indices[i]]
                personal_mus[i] = coef[0]
                personal_covs[i, i] = coef[1]

            temp_cov = block_diag(self.inv_hess, personal_covs)

            temp_hess = np.linalg.pinv(temp_cov)

            temp_theta = np.concatenate((self.theta_hat, personal_mus))

        solution = minimize(self.negative_log_posterior, temp_theta,
                            args=(X_train, y_train, temp_theta, temp_hess), method="BFGS", jac=self.g_negative_log_posterior)

        temp_theta_map = solution.x
        temp_hess = self.h_negative_log_posterior(
            temp_theta_map, X_train, temp_hess)
        temp_cov = np.linalg.pinv(temp_hess)

        if self.personal_coefs:
            self.theta_hat = temp_theta_map[:-len(global_indices)]

            self.hess = temp_hess[:-len(global_indices), :-len(global_indices)]
            self.inv_hess = temp_cov[:-
                                     len(global_indices), :-len(global_indices)]

            for i in range(len(global_indices)):
                new_mu = temp_theta_map[self.d + i]
                new_cov = temp_cov[self.d + i, self.d + i]
                self.individual_coefs[global_indices[i]] = (new_mu, new_cov)

            personal_mus = np.array([c0 for c0, _ in self.individual_coefs])
            personal_covs = np.array([c1 for _, c1 in self.individual_coefs])

            self.full_theta_hat = np.concatenate(
                (self.theta_hat, personal_mus))
            self.full_inv_hess = block_diag(
                self.inv_hess, np.diag(personal_covs))

        else:
            self.theta_hat = temp_theta_map
            self.hess = temp_hess
            self.inv_hess = temp_cov
            self.M = self.M + X_train.T.dot(X_train)
            self.M_inv = np.linalg.pinv(self.M)

    def add_independent_features(self, X_train, indices):
        batch_unique_idxs = list(set(np.array(indices).flatten()))

        batch_one_hot_dict = {}

        for i in range(len(batch_unique_idxs)):
            batch_one_hot_dict[batch_unique_idxs[i]] = i

        one_hot_features = np.zeros((X_train.shape[0], len(batch_unique_idxs)))

        for index in range(len(indices)):
            i, j = indices[index]
            plus_index = batch_one_hot_dict[i]
            minus_index = batch_one_hot_dict[j]

            one_hot_features[index][plus_index] = 1
            one_hot_features[index][minus_index] = -1

        return np.concatenate((X_train, one_hot_features), axis=1), batch_unique_idxs

    def negative_log_posterior(self, w, X, y, theta_map, Hess):
        y_hats = expit(X.dot(w))

        prior = 0.5 * (w - theta_map).dot(Hess).dot((w - theta_map))

        likelihood = np.sum(
            y * np.log(y_hats + PRECISION) + (1-y)*np.log(1-y_hats + PRECISION))

        return prior - likelihood

    def g_negative_log_posterior(self, w, X, y, theta_map, prior_hess):
        '''
        Calculates the normalized gradient of the negative log posterior at point w
        (this is used by the optimizer for more stable convergence)
        '''

        y_hats = expit(X.dot(w))

        gradient = np.dot(
            X.transpose(), y_hats - y) + prior_hess.dot(w - theta_map)

        return gradient

    def h_negative_log_posterior(self, w, X, prior_hess):
        '''
        Calculates the hessian of the negative log posterior at point w
        '''

        y_hats = expit(X.dot(w))
        coef = np.multiply(y_hats, 1-y_hats)

        return prior_hess + np.dot(X.transpose(), X * coef[:, np.newaxis])

    def ordering(self):
        '''
        Return ordering of data
        '''
        scores = self.X.dot(self.theta_hat)
        if self.personal_coefs:
            scores = scores + np.array([c[0] for c in self.individual_coefs])
        return rankdata(scores)

    def get_model_info(self):
        model_info = {
            "theta_hat": copy.copy(self.theta_hat),
            "inv_hessian": copy.copy(self.inv_hess),
        }

        if self.personal_coefs:
            model_info["personal_coefs"] = copy.copy(self.individual_coefs)

        return model_info


class BayesGURO(BayesianAlgorithm):

    def __init__(
            self, X, update_every=10, seed=None, personal_coefs=False, post_sample_size=50,
            sample_combinations=False, random_comparison=False, personal_var=1):
        super().__init__(X, update_every, seed,
                         personal_coefs=personal_coefs, personal_var=personal_var)

        self.post_sample_size = post_sample_size
        self.sample_combinations = sample_combinations
        self.random_comparison = random_comparison

    def act(self):

        if self.random_comparison:
            return self.random_state.choice(self.n, size=2, replace=False), None

        if self.personal_coefs:
            data = self.X_one_hot

            sample_thetas = self.random_state.multivariate_normal(
                self.theta_hat, self.inv_hess, size=self.post_sample_size)

            individual_theta_hats = self.full_theta_hat[self.d:]
            individual_variances = np.array(
                [c1 for _, c1 in self.individual_coefs])

            sample_individual_thetas = np.random.normal(individual_theta_hats, np.sqrt(
                individual_variances), size=(self.post_sample_size, len(individual_theta_hats)))

            sample_thetas = np.concatenate(
                (sample_thetas, sample_individual_thetas), axis=1)
        else:
            data = self.X
            sample_thetas = self.random_state.multivariate_normal(
                self.theta_hat, self.inv_hess, size=self.post_sample_size)

        return self.find_largest_average_disagreement(
            sample_thetas, data, sample=self.sample_combinations), sample_thetas

    def find_largest_average_disagreement(
            self, sampled_thetas, data, sample=False, combinations=None):

        if combinations is None:
            combinations = np.array(
                list(
                    itertools.combinations(
                        range(len(data)),
                        2)))

        if sample:
            sample_idxs = self.random_state.choice(
                len(combinations), (5000,), replace=False)
            combinations = combinations[sample_idxs]

        # x_i - x_j for all combinations of i and j
        Z = np.array([data[c[0]] - data[c[1]] for c in combinations])

        prediction_probs = []

        for s_theta in sampled_thetas:
            prediction_probs.append(expit(Z.dot(s_theta)))

        prediction_probs = np.array(prediction_probs)
        variances = np.var(prediction_probs, axis=0)
        idx = np.argmax(variances)

        return combinations[idx]


class CoLSTIM(BaseAlgorithm):

    def __init__(
            self, X, update_every=10, seed=None, c=0.1,
            random_first_element=False, gumbel_scale=1, adapted=False):
        super().__init__(X, update_every=update_every, seed=seed)

        self.M = np.identity(self.d)
        self.M_inv = np.linalg.pinv(self.M)
        self.b = np.zeros((self.d,))
        self.theta_hat = np.matmul(np.linalg.inv(self.M), self.b)
        self.c = c

        self.random_first_element = random_first_element
        self.gumbel_scale = gumbel_scale

        self.adapted = adapted

        self.model = LogisticRegression()

    def act(self):

        if len(self.obs_data) < 10:
            return self.random_state.choice(
                self.n, size=2, replace=False), None

        if self.adapted:
            return self.find_best_pair(self.X), None

        estimated_scores = np.argmax(self.X.dot(self.theta_hat))

        # generate noise
        if self.random_first_element or len(self.obs_data) < self.update_every:
            i = self.random_state.randint(0, self.n)
        else:
            noise = self.random_state.gumbel(
                scale=self.gumbel_scale, size=self.n)

            i_exploration_term = np.sqrt(np.diagonal(self.X.dot(
                self.M_inv).dot(self.X.T)))
            if not self.adapted:
                i = np.argmax(estimated_scores + noise * i_exploration_term)
            else:
                i = np.argmax(noise * i_exploration_term)

        # Z
        diff_matrix = np.array([d - self.X[i] for d in self.X])
        estimated_diff_scores = diff_matrix.dot(self.theta_hat)

        j_exploration_term = np.sqrt(
            np.sum(diff_matrix.dot(self.M_inv) * diff_matrix, axis=1))

        if not self.adapted:
            upper_bounds = estimated_diff_scores + self.c * j_exploration_term
            j = np.argmax(upper_bounds)
        else:
            j = np.argmax(j_exploration_term)

        return [i, j], None

    def find_best_pair(self, data, sample=False):

        combinations = np.array(
            list(
                itertools.combinations(
                    range(len(data)),
                    2)))

        if sample:
            sample_idxs = self.random_state.choice(
                len(combinations), (5000,), replace=False)
            combinations = combinations[sample_idxs]

        # x_i - x_j for all combinations of i and j
        diff_matrix = np.array([data[c[0]] - data[c[1]] for c in combinations])

        exploration_term = np.sqrt(
            np.sum(diff_matrix.dot(self.M_inv) * diff_matrix, axis=1))

        idx = np.argmax(exploration_term)

        return combinations[idx]

    def update_model(self):

        X_train = np.array(self.obs_data[-self.update_every:])
        y_train = np.array(self.obs_labels[-self.update_every:])

        # Sequential updates of M and b
        self.M = self.M + X_train.T.dot(X_train)
        self.M_inv = np.linalg.pinv(self.M)
        self.b = self.b + X_train.T.dot(y_train)

        self.model.fit(self.obs_data, self.obs_labels)
        self.theta_hat = self.model.coef_[0]

    def get_model_info(self):
        return {
            "theta_hat": copy.copy(self.theta_hat),
            "inv_hessian": np.linalg.pinv(self.get_hess())
        }


class GURO(BaseAlgorithm):
    '''
    Greedy shrinkage of confidence ellipsoid
    '''

    def __init__(self, X, update_every=10, lazy=False, seed=None, sample_combinations=False, personal_coefs=False):

        super().__init__(X, update_every=update_every, seed=seed)

        self.personal_coefs = personal_coefs
        self.lazy = lazy
        self.sample_combinations = sample_combinations

        if personal_coefs:
            self.feature_d = self.d
            self.X = np.concatenate(
                (self.X, np.identity(self.X.shape[0])), axis=1)
            self.d = self.X.shape[1]
            self.theta_hat = np.ones((self.d,))

        self.M = np.identity(self.d)
        self.M_inv = np.linalg.inv(self.M)

        self.model = LogisticRegression()
        self.det = np.linalg.det(self.M)
        self.c = 1

    def act(self):
        return self.find_best_pair(sample=self.sample_combinations)

    def find_best_pair(self, sample=False, combinations=None):

        if self.t < 10:
            return self.random_state.choice(
                self.n, size=2, replace=False), None

        if combinations is None:
            combinations = np.array(
                list(
                    itertools.combinations(
                        range(len(self.X)),
                        2)))

        if sample:
            sample_idxs = self.random_state.choice(
                len(combinations), (5000,), replace=False)
            combinations = combinations[sample_idxs]

        if sample:
            sample_idxs = self.random_state.choice(
                len(combinations), (5000,), replace=False)
            combinations = combinations[sample_idxs]

        # x_i - x_j for all combinations of i and j
        diff_matrix = np.array([self.X[c[0]] - self.X[c[1]]
                               for c in combinations])

        # Update
        # estimate weighted norm of each data point in diff_matrix
        M_inv = self.M_inv

        if self.personal_coefs:
            covs = np.diag(M_inv)[self.feature_d:]
            covs_appended = covs * diff_matrix[:, self.feature_d:]

            M_inv_cropped = M_inv[:self.feature_d, :self.feature_d]
            real_feature_part = diff_matrix[:,
                                            :self.feature_d].dot(M_inv_cropped)

            dot_product = np.concatenate(
                (real_feature_part, covs_appended), axis=1)
        else:
            dot_product = diff_matrix.dot(M_inv)

        weighted_norm = np.sum(dot_product * diff_matrix, axis=1)

        # scale weighted norm with sigmoid derivative
        weighted_norm = weighted_norm * \
            expit(diff_matrix.dot(self.theta_hat)) * \
            (1 - expit(diff_matrix.dot(self.theta_hat)))

        # find the pair with the largest weighted norm
        max_index = np.argmax(weighted_norm)
        return combinations[max_index], None

    def update(self, i, j, observation):
        '''
        Append to data and update Hessian and model
        '''
        self.t += 1
        x = self.X[i] - self.X[j]
        self.obs_data.append(x)
        self.obs_labels.append(observation)
        self.obs_indices.append((i, j))
        # Update Hessian via Sherman-Morrison formula

        outer = np.outer(x, x)
        y = expit(x.dot(self.theta_hat))
        self.M = self.M + outer * y * (1 - y)

        if self.personal_coefs:
            # By assuming independence between individual params we can use the sparsity of the covariance matrix
            outer_cropped = outer[:self.feature_d, :self.feature_d]
            inv_hess_cropped = self.M_inv[:self.feature_d, :self.feature_d]

            inv_hess_diag = self.M_inv[self.feature_d:,
                                       self.feature_d:].diagonal()
            outer_diag = outer[self.feature_d:, self.feature_d:].diagonal()

            dense_mul = inv_hess_cropped @ outer_cropped @ inv_hess_cropped
            sparse_mul = inv_hess_diag * outer_diag * inv_hess_diag

            matrix_prod = block_diag(dense_mul, np.diag(sparse_mul))
            self.M_inv = self.M_inv - y * \
                (1-y) * matrix_prod / (1 + y * (1-y) * x.dot(self.M_inv).dot(x))
        else:
            self.M_inv = self.M_inv - y * \
                (1-y) * (self.M_inv @ outer @ self.M_inv) / \
                (1 + y * (1-y) * x.dot(self.M_inv).dot(x))

        # Update model
        if self.t % self.update_every == 0 and self.t >= 10 and not self.lazy:
            self.update_model()

    def update_model(self):

        if len(np.unique(self.obs_labels)) < 2:
            return

        self.model.fit(self.obs_data, self.obs_labels)
        self.theta_hat = self.model.coef_[0]

    def get_model_info(self):
        return {
            "theta_hat": copy.copy(self.theta_hat),
            "inv_hessian": copy.copy(self.M_inv),
        }


class NormMin(BaseAlgorithm):
    """
    Greedy shrinkage of confidence ellipsoid
    """

    def __init__(self, X, update_every=10, lazy=False, seed=None):
        super().__init__(X, update_every=update_every, seed=seed)
        self.lazy = lazy
        self.M = np.identity(self.d)
        self.M_inv = np.linalg.inv(self.M)
        # compute pair-wise difference for all data points in X (n x n x d)
        self.diff_matrix = np.array([x - self.X for x in self.X])
        self.model = LogisticRegression()
        self.det = np.linalg.det(self.M)
        self.c = 1

    def act(self):
        if self.t < 10:
            return self.random_state.choice(self.n, size=2, replace=False), None
        # Update
        # estimate weighted norm of each data point in diff_matrix
        M_inv = self.M_inv
        # use einstein summation to compute weighted norm for each vector in diff_matrix
        # That is diff_matrix[i, j, :].T @ M_inv @ diff_matrix[i, j, :]
        weighted_norm = np.einsum(
            "ijk,kl,ijl->ij", self.diff_matrix, M_inv, self.diff_matrix
        )
        assert weighted_norm.shape == (self.n, self.n)
        # find the pair with the largest weighted norm
        i, j = np.unravel_index(np.argmax(weighted_norm), weighted_norm.shape)
        assert weighted_norm[i, j] == weighted_norm.max()
        return [i, j], None

    def update(self, i, j, observation):
        """
        Append to data and update Hessian and model
        """
        self.t += 1
        x = self.X[i] - self.X[j]
        self.obs_data.append(x)
        self.obs_labels.append(observation)
        # Update Hessian via Sherman-Morrison formula
        outer = np.outer(x, x)
        y = expit(x.dot(self.theta_hat))
        self.M = self.M + outer * y * (1 - y)
        self.M_inv = self.M_inv - y * (1 - y) * (self.M_inv @ outer @ self.M_inv) / (
            1 + y * (1 - y) * x.dot(self.M_inv).dot(x)
        )
        # Update model
        if self.t % self.update_every == 0 and self.t >= 10 and not self.lazy:
            self.update_model()

    def update_model(self):
        if len(np.unique(self.obs_labels)) < 2:
            return
        self.model.fit(self.obs_data, self.obs_labels)
        self.theta_hat = self.model.coef_[0]
        # Compute Hessian of the negative log-likelihood
        hess = np.zeros((self.d, self.d))
        for x, _ in zip(self.obs_data, self.obs_labels):
            y_hat = expit(x.dot(self.theta_hat))
            hess += y_hat * (1 - y_hat) * np.outer(x, x)

    def get_model_info(self):
        return {
            "theta_hat": copy.copy(self.theta_hat),
            "inv_hessian": copy.copy(self.M_inv),
        }


class TrueSkill:

    def __init__(self, X, seed=None, update_overlaps=True, available_combinations=None):

        self.n = X.shape[0]
        self.d = X.shape[1]

        self.obs_data = []
        self.obs_labels = []
        self.random_state = np.random.RandomState(seed)
        self.update_overlaps = update_overlaps

        self.ratings = [Rating() for k in range(self.n)]
        self.comparisons = []

        self.available_combinations = available_combinations
        self.comparison_mask = np.ones((self.n, self.n))

        self.update_comparison_mask()

        self.overlap_matrix = np.full(
            (self.n, self.n),
            self.intervals_overlap(0, 1))
        np.fill_diagonal(self.overlap_matrix, 0)

    def intervals_overlap(
            self, key1: Union[int, float, str],
            key2: Union[int, float, str]) -> float:
        """
        Calculate the overlap between the intervals of two keys.

        Args:
            key1: The first key.
            key2: The second key.

        Returns:
            The overlap value between the intervals.
        """

        n_sd = 3

        r1_low = self.ratings[key1].mu - n_sd * self.ratings[key1].sigma
        r1_high = self.ratings[key1].mu + n_sd * self.ratings[key1].sigma

        r2_low = self.ratings[key2].mu - n_sd * self.ratings[key2].sigma
        r2_high = self.ratings[key2].mu + n_sd * self.ratings[key2].sigma

        common_gap = min(r1_high, r2_high) - max(r1_low, r2_low)
        overall_gap = (max(r1_high, r2_high) - min(r1_low, r2_low))
        largest_span = max(r1_high-r1_low, r2_high-r2_low)

        return common_gap / overall_gap * largest_span

    def update_overlap_matrix(self, key: int):
        """
        Update the overlap matrix with the overlap values for a specific key.

        Args:
            key: The key to update the overlap matrix.
        """

        for i in range(self.n):
            if i != key:
                overlap = self.intervals_overlap(key, i)
                self.overlap_matrix[i][key] = overlap
                self.overlap_matrix[key][i] = overlap

    def act(self):

        if not self.update_overlaps:
            index = self.random_state.choice(
                len(self.available_combinations), size=1, replace=False)
            return self.available_combinations[index[0]], None

        max_sum = - np.inf
        masked = self.overlap_matrix * self.comparison_mask
        for i in range(self.n):
            max_index = np.argmax(masked[i])
            if masked[i][max_index] > max_sum:
                max_sum = masked[i][max_index]
                comparison = [i, max_index]

        return comparison, None

    def update(self, i, j, observation):

        if observation == 1:
            self.ratings[i], self.ratings[j] = rate_1vs1(
                self.ratings[i], self.ratings[j])
        else:
            self.ratings[j], self.ratings[i] = rate_1vs1(
                self.ratings[j], self.ratings[i])

        if self.update_overlaps:
            self.update_overlap_matrix(i)
            self.update_overlap_matrix(j)

        if self.available_combinations is not None:
            self.remove_from_combinations_and_mask(i, j)

    def remove_from_combinations_and_mask(self, i, j):

        for index in range(len(self.available_combinations)):
            x1, x2 = self.available_combinations[index]

            if (x1 == i and x2 == j) or (x2 == i and x1 == j):
                self.available_combinations = np.delete(
                    self.available_combinations, index, 0)
                if self.update_overlaps:
                    self.update_comparison_mask()
                break

    def update_comparison_mask(self):
        if self.available_combinations is not None:
            self.comparison_mask = np.zeros((self.n, self.n))
            for x1, x2 in self.available_combinations:
                self.comparison_mask[x1, x2] = 1
                self.comparison_mask[x2, x1] = 1

    def ordering(self):
        '''
        Return ordering of data
        '''

        scores = [r.mu for r in self.ratings]
        return rankdata(scores)

    def add_data(self, new_data, new_comparisons=np.array([])):

        self.n = self.n + new_data.shape[0]
        self.available_combinations = np.concatenate(
            (self.available_combinations, new_comparisons))

        self.ratings = self.ratings + [Rating()
                                       for _ in range(new_data.shape[0])]

        self.update_comparison_mask()

        self.overlap_matrix = np.zeros(
            (self.n, self.n))
        for i in range(self.n):
            self.update_overlap_matrix(i)

    def get_model_info(self):
        return copy.deepcopy(self.ratings)


class BALD(BayesianAlgorithm):

    def __init__(
            self, X, update_every=10, seed=None, marginal_only=False,
            sample_combinations=False, marginal_priority=False, personal_coefs=False):
        super().__init__(X, update_every, seed, personal_coefs=personal_coefs)

        self.marginal_only = marginal_only
        self.sample_combinations = sample_combinations
        self.marginal_priority = marginal_priority

    def act(self):

        if self.t < 10:
            return self.random_state.choice(
                self.n, size=2, replace=False), None

        if self.personal_coefs:
            data = self.X_one_hot
        else:
            data = self.X

        return self.max_expected_posterior_entropy_decrease(
            data, sample=self.sample_combinations)

    def max_expected_posterior_entropy_decrease(self, data, sample=False, combinations=None):

        if combinations is None:
            combinations = np.array(
                list(
                    itertools.combinations(
                        range(len(data)),
                        2)))

        if sample:
            sample_idxs = self.random_state.choice(
                len(combinations), (5000,), replace=False)
            combinations = combinations[sample_idxs]

        if self.personal_coefs:
            theta_hat = self.full_theta_hat
            inv_hess = self.full_inv_hess
        else:
            theta_hat = self.theta_hat
            inv_hess = self.inv_hess

        # x_i - x_j for all combinations of i and j
        diff_matrix = np.array([data[c[0]] - data[c[1]] for c in combinations])

        mu_z = diff_matrix.dot(theta_hat)

        if self.personal_coefs:
            covs = np.diag(inv_hess)[self.d:]

            covs_appended = covs * diff_matrix[:, self.d:]

            real_feature_part = diff_matrix[:, :self.d].dot(self.inv_hess)

            dot_product = np.concatenate(
                (real_feature_part, covs_appended), axis=1)
        else:
            dot_product = diff_matrix.dot(inv_hess)

        sigma_z = np.sum(dot_product * diff_matrix, axis=1)

        predictive_prob = expit(
            np.divide(mu_z, np.sqrt(1 + (np.pi/8)*sigma_z)))

        marginal_entropy = np.array(list(map(self.entropy, predictive_prob)))

        C = np.sqrt(4 * np.log(2))
        expected_entropy = C / np.sqrt(sigma_z + (C ** 2)) * np.exp(- np.divide(
            np.square(mu_z), 2 * (sigma_z + C ** 2)))

        if self.marginal_only:
            result = marginal_entropy
        else:
            result = marginal_entropy - expected_entropy

        idx = np.argmax(result)

        if self.marginal_priority and marginal_entropy[idx] < 1/2 and not self.marginal_only:
            self.marginal_only = True
            print("switching to marginal only")

        return combinations[idx], [
            marginal_entropy[idx],
            expected_entropy[idx]]

    def entropy(self, x):
        return - x * np.log2(x) - (1-x) * np.log2(1-x)

# The subclasses below handle the scenario where we only have access to a subset of comparisons
# while also allowing for the addition of more data later on.


class BayesGURORealData(BayesGURO):

    def __init__(self, *args, available_combinations, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_combinations = available_combinations

    def act(self):

        if self.random_comparison:
            index = self.random_state.choice(
                len(self.available_combinations), size=1, replace=False)
            return self.available_combinations[index[0]], None

        if self.personal_coefs:
            data = self.X_one_hot

            sample_thetas = self.random_state.multivariate_normal(
                self.theta_hat, self.inv_hess, size=self.post_sample_size)

            individual_theta_hats = self.full_theta_hat[self.d:]
            individual_variances = np.array(
                [c1 for _, c1 in self.individual_coefs])

            sample_individual_thetas = np.random.normal(individual_theta_hats, np.sqrt(
                individual_variances), size=(self.post_sample_size, len(individual_theta_hats)))

            sample_thetas = np.concatenate(
                (sample_thetas, sample_individual_thetas), axis=1)
        else:
            data = self.X
            sample_thetas = self.random_state.multivariate_normal(
                self.theta_hat, self.inv_hess, size=self.post_sample_size)

        return self.find_largest_average_disagreement(
            sample_thetas, data, sample=self.sample_combinations, combinations=self.available_combinations), None

    def update(self, i, j, observation):
        '''
        Update the algorithm with the observation of the pair (i, j)
        '''

        for index in range(len(self.available_combinations)):
            x1, x2 = self.available_combinations[index]
            if (x1 == i and x2 == j) or (x2 == i and x1 == j):
                self.available_combinations = np.delete(
                    self.available_combinations, index, 0)
                break

        super().update(i, j, observation)

    def add_data(self, new_data, new_comparisons=np.array([])):
        new_n = new_data.shape[0]
        self.X = np.concatenate((self.X, new_data))
        self.n = self.X.shape[0]
        self.available_combinations = np.concatenate(
            (self.available_combinations, new_comparisons))

        if self.personal_coefs:

            for _ in new_data:
                self.individual_coefs.append((0, 1))

            self.X_one_hot = np.concatenate(
                (self.X, np.identity(self.n)), axis=1)

            self.full_inv_hess = block_diag(
                self.full_inv_hess, np.identity(new_n))
            self.full_theta_hat = np.concatenate(
                (self.full_theta_hat, np.zeros(new_n)))


class BALDRealData(BALD):

    def __init__(self, *args, available_combinations, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_combinations = available_combinations

    def act(self):

        if self.t < 10:
            index = self.random_state.choice(
                len(self.available_combinations), size=1, replace=False)
            return self.available_combinations[index[0]], None

        if self.personal_coefs:
            data = self.X_one_hot
        else:
            data = self.X

        return self.max_expected_posterior_entropy_decrease(
            data, sample=self.sample_combinations, combinations=self.available_combinations)

    def update(self, i, j, observation):
        '''
        Update the algorithm with the observation of the pair (i, j)
        '''

        for index in range(len(self.available_combinations)):
            x1, x2 = self.available_combinations[index]
            if (x1 == i and x2 == j) or (x2 == i and x1 == j):
                self.available_combinations = np.delete(
                    self.available_combinations, index, 0)
                break

        super().update(i, j, observation)

    def add_data(self, new_data, new_comparisons=np.array([])):
        new_n = new_data.shape[0]
        self.X = np.concatenate((self.X, new_data))
        self.n = self.X.shape[0]
        self.available_combinations = np.concatenate(
            (self.available_combinations, new_comparisons))

        if self.personal_coefs:

            for _ in new_data:
                self.individual_coefs.append((0, 1))

            self.X_one_hot = np.concatenate(
                (self.X, np.identity(self.n)), axis=1)

            self.full_inv_hess = block_diag(
                self.full_inv_hess, np.identity(new_n))
            self.full_theta_hat = np.concatenate(
                (self.full_theta_hat, np.zeros(new_n)))


class GURORealData(GURO):

    def __init__(self, *args, available_combinations, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_combinations = available_combinations

    def act(self):
        if self.t < 10:
            index = self.random_state.choice(
                len(self.available_combinations), size=1, replace=False)
            return self.available_combinations[index[0]], None

        return self.find_best_pair(sample=self.sample_combinations, combinations=self.available_combinations)

    def update(self, i, j, observation):
        '''
        Update the algorithm with the observation of the pair (i, j)
        '''

        for index in range(len(self.available_combinations)):
            x1, x2 = self.available_combinations[index]
            if (x1 == i and x2 == j) or (x2 == i and x1 == j):
                self.available_combinations = np.delete(
                    self.available_combinations, index, 0)
                break

        super().update(i, j, observation)

    def add_data(self, new_data, new_comparisons=np.array([])):
        new_n = new_data.shape[0]
        self.n = self.X.shape[0] + new_n
        self.available_combinations = np.concatenate(
            (self.available_combinations, new_comparisons))

        if self.personal_coefs:
            self.X = np.concatenate(
                (self.X[:, :self.feature_d], new_data), axis=0)
            self.X = np.concatenate((self.X, np.identity(self.n)), axis=1)

            self.M_inv = block_diag(self.M_inv, np.identity(new_n))
            self.M = block_diag(self.M, np.identity(new_n))

            for i in range(len(self.obs_data)):
                self.obs_data[i] = np.concatenate(
                    (self.obs_data[i], np.zeros(new_n)))

            self.update_model()
        else:
            self.X = np.concatenate((self.X, new_data))


class UniformSamplingRealData(UniformSampling):

    def __init__(self, *args, available_combinations, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_combinations = available_combinations

    def act(self):
        index = self.random_state.choice(
            len(self.available_combinations), size=1, replace=False)
        return self.available_combinations[index[0]], None

    def update(self, i, j, observation):
        '''
        Update the algorithm with the observation of the pair (i, j)
        '''

        for index in range(len(self.available_combinations)):
            x1, x2 = self.available_combinations[index]
            if (x1 == i and x2 == j) or (x2 == i and x1 == j):
                self.available_combinations = np.delete(
                    self.available_combinations, index, 0)
                break
        super().update(i, j, observation)

    def add_data(self, new_data, new_comparisons=np.array([])):
        new_n = new_data.shape[0]
        self.n = self.X.shape[0] + new_n
        self.available_combinations = np.concatenate(
            (self.available_combinations, new_comparisons))

        self.X = np.concatenate((self.X, new_data))


class CoLSTIMRealData(CoLSTIM):

    def __init__(self, *args, available_combinations, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_combinations = available_combinations

    def update(self, i, j, observation):
        '''
        Update the algorithm with the observation of the pair (i, j)
        '''

        for index in range(len(self.available_combinations)):
            x1, x2 = self.available_combinations[index]
            if (x1 == i and x2 == j) or (x2 == i and x1 == j):
                self.available_combinations = np.delete(
                    self.available_combinations, index, 0)
                break
        super().update(i, j, observation)

    def act(self):

        if len(self.obs_data) < 10:
            index = self.random_state.choice(
                len(self.available_combinations), size=1, replace=False)
            return self.available_combinations[index[0]], None

        estimated_scores = np.argmax(self.X.dot(self.theta_hat))

        noise = self.random_state.gumbel(
            scale=self.gumbel_scale, size=self.n)

        i_exploration_term = np.sqrt(np.diagonal(self.X.dot(
            self.M_inv).dot(self.X.T)))

        candidates = np.argsort(estimated_scores + noise * i_exploration_term)

        found = False
        matching_comps = []
        for c in candidates:
            for i1, i2 in self.available_combinations:

                if i1 == c or i2 == c:
                    found = True
                    matching_comps.append([i1, i2])

            if found:
                break

        # Z
        diff_matrix = np.array([self.X[c[0]] - self.X[c[1]]
                               for c in matching_comps])

        estimated_diff_scores = diff_matrix.dot(self.theta_hat)

        j_exploration_term = np.sqrt(
            np.sum(diff_matrix.dot(self.M_inv) * diff_matrix, axis=1))

        upper_bounds = estimated_diff_scores + self.c * j_exploration_term
        pair = matching_comps[np.argmax(upper_bounds)]

        return pair, None

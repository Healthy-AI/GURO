import numpy as np


class DataGenerator:
    """
    Generates toy data of given dimensionality
    """

    def __init__(
        self,
        d,
        seed=None,
        true_features_fraction=1.0,
        low=-3,
        high=3,
        unbalanced_theta=False,
    ) -> None:
        self.d = d
        self.random_state = np.random.RandomState(seed)

        if unbalanced_theta:
            small_feature_amount = self.d // 2
            large_feature_amount = self.d - small_feature_amount

            theta_low = self.random_state.uniform(
                low=low, high=high, size=small_feature_amount
            )
            theta_high = self.random_state.uniform(
                low=low * 100, high=high * 100, size=large_feature_amount
            )

            self.theta = np.concatenate((theta_low, theta_high))
        else:
            self.theta = self.random_state.uniform(low=low, high=high, size=self.d)

        bad_feature_amount = int(d * (1 - true_features_fraction))

        bad_features = self.random_state.choice(
            self.d, bad_feature_amount, replace=False
        )

        self.theta[bad_features] = 0

    def generate_noisy_data(
        self, amount, standard_deviation, unbalanced_features=False
    ):
        """
        Generate data and scores with noise
        args:
        amount: number of data points
        standard_deviation: standard deviation of noise

        return:
        data: data points
        noisy_scores: scores with noise
        """

        if unbalanced_features:
            small_feature_amount = self.d // 2
            large_feature_amount = self.d - small_feature_amount

            small_data_amount = amount // 2
            large_data_amount = amount - small_data_amount

            small_feature_data = self.random_state.normal(
                0, 0.1, size=(small_data_amount, small_feature_amount)
            )

            small_feature_data = np.concatenate(
                (
                    small_feature_data,
                    np.zeros((small_data_amount, large_feature_amount)),
                ),
                axis=1,
            )

            large_feature_data = self.random_state.normal(
                0, 100, size=(large_data_amount, large_feature_amount)
            )

            large_feature_data = np.concatenate(
                (
                    np.zeros((large_data_amount, small_feature_amount)),
                    large_feature_data,
                ),
                axis=1,
            )

            data = np.concatenate((small_feature_data, large_feature_data), axis=0)

            self.random_state.shuffle(data)
        else:
            data = self.random_state.normal(0, 1, size=(amount, self.d))

        scores = data.dot(self.theta)

        noise = self.random_state.normal(scale=standard_deviation, size=amount)

        noisy_scores = scores + noise

        return data, noisy_scores


class AdvGenerator:
    """
    Generate instances that are hard for regret minimization algorithms
    """

    def __init__(
        self, d=2, seed=None, extra_hard=True, fraction_good=0.1, noise=0.1
    ) -> None:
        """
        d_1 and d_2 are the directions that matters for the problem. If extra_hard is True, then d_1 and d_2 are orthogonal.
        This implies tha playing the best-arm won't give any information about the other direction (i.e. sorting the arms will be hard)

        """

        self.d = d
        self.random_state = np.random.RandomState(seed)
        self.theta = self.random_state.uniform(low=-0.1, high=0.1, size=self.d)
        self.theta[0] = 1
        self.theta[1] = 0.5
        self.extra_hard = extra_hard
        self.fraction_good = fraction_good
        self.noise = noise

    def generate_data(self, amount):
        data = []
        scores = []
        d1_samples = int(amount * self.fraction_good)
        # draw samples from the first direction (high reward)
        if self.extra_hard:
            d1 = np.zeros((d1_samples, self.d))
        else:
            d1 = np.random.normal(0, self.noise, size=(d1_samples, self.d))
        d1[:, 0] = np.random.uniform(1, 2, size=d1_samples)
        data.append(d1)
        scores.append(d1.dot(self.theta))
        # draw samples from the second direction (low reward)
        d2_samples = amount - d1_samples
        print(d2_samples)
        if self.extra_hard:
            d2 = np.zeros((d2_samples, self.d))
        else:
            d2 = np.random.normal(0, self.noise, size=(d2_samples, self.d))
        d2[:, 1] = np.random.uniform(1, 2, size=d2_samples)
        data.append(d2)
        scores.append(d2.dot(self.theta))
        data = np.concatenate(data)
        scores = np.concatenate(scores)
        return data, scores


class BimodalGenerator:
    """
    Generate data where arms are drawn from two clusters in R^2
    """
    def __init__(self, n_samples=100, seed=0) -> None:
        """ """
        from sklearn.datasets import make_blobs
        self.n_samples = n_samples
        self.random_state = np.random.RandomState(seed)
        self.c1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.c2 = -np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.c3 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        self.c4 = -np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        self.theta = np.zeros(10)
        self.theta[0] = 0.3
        self.theta[1] = -0.3
        self.theta[2] = -0.2
        self.d = len(self.theta)
        self.X = []
        self.scores = []
        scale = 1
        for i in range(n_samples):
            if i < n_samples * 0.3:
                x = self.random_state.normal(loc=self.c1, scale=scale)
                x[2:] = 0
            elif i < n_samples * 0.6:
                x = self.random_state.normal(loc=self.c2, scale=scale)
                x[2:] = 0
            elif i < n_samples * 0.9:
                x = self.random_state.normal(loc=self.c3, scale=scale)
                x[0:2] = 0
            else:
                x = self.random_state.normal(loc=self.c4, scale=scale)
                x[0:2] = 0
            self.X.append(x)
            self.scores.append(self.theta.dot(x))
        self.X = np.array(self.X)
    def generate_data(self):
        return self.X, self.scores


class SkewedTestTrain:
    """
    Generate train and test data with different distributions
    """

    def __init__(self, n_samples, seed=0) -> None:
        self.d = 10
        self.n_samples = n_samples
        self.random_state = np.random.RandomState(seed)

        self.c1 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.c2 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        self.theta = self.random_state.uniform(low=-1, high=1, size=self.d)

        self.X_train = []
        self.scores_train = []
        self.train_dist = [0.9, 0.1]

        # construct train data
        for _ in range(n_samples):
            if self.random_state.uniform() < self.train_dist[0]:
                x = self.random_state.normal(loc=self.c1, scale=1)
            else:
                x = self.random_state.normal(loc=self.c2, scale=1)
            self.X_train.append(x)
            self.scores_train.append(self.theta.dot(x))

        self.X_train = np.array(self.X_train)

        self.X_test = []
        self.scores_test = []
        self.test_dist = [0.1, 0.9]

        # construct test data
        for _ in range(n_samples):
            if self.random_state.uniform() < self.test_dist[0]:
                x = self.random_state.normal(loc=self.c1, scale=1)
            else:
                x = self.random_state.normal(loc=self.c2, scale=1)
            self.X_test.append(x)
            self.scores_test.append(self.theta.dot(x))

        self.X_test = np.array(self.X_test)

    def generate_data(self):
        return self.X_train, self.scores_train, self.X_test, self.scores_test

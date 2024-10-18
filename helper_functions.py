import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import kendalltau, rankdata

def plot_thetas(auxiliary, true_theta):

    k = (true_theta / true_theta[0])[1]

    true_theta_1 = np.linspace(-4,4, 100)
    true_theta_2 = k * true_theta_1

    plt.figure(figsize=(8, 6), dpi=100)

    theta_1_distr_initial = np.array(auxiliary['aux'][0].transpose()[0])
    theta_2_distr_initial = np.array(auxiliary['aux'][0].transpose()[1])

    theta_1_distr_mid = np.array(auxiliary['aux'][10].transpose()[0])
    theta_2_distr_mid = np.array(auxiliary['aux'][10].transpose()[1])

    theta_1_distr_last = np.array(auxiliary['aux'][-1].transpose()[0])
    theta_2_distr_last = np.array(auxiliary['aux'][-1].transpose()[1])

    plt.scatter(theta_1_distr_initial, theta_2_distr_initial)
    plt.scatter(theta_1_distr_mid, theta_2_distr_mid)
    plt.scatter(theta_1_distr_last, theta_2_distr_last)

    plt.grid()
    plt.title("Posterior distribution sampled at different time points")

    plt.scatter(true_theta[0], true_theta[1], zorder=10, marker='x', s=80)
    plt.legend(["0 comparisons", "10 comparisons", "100 comparisons", 'True $\\theta$'])
    plt.plot(true_theta_1, true_theta_2, color="black")
    plt.plot(-true_theta_2, true_theta_1, color="black")

    plt.xlabel('$\\theta_1$', fontsize=16)
    plt.ylabel('$\\theta_2$', fontsize=16)
    plt.axis('equal')
    plt.show()

def plot_samples(auxiliary, true_theta, X_sample):

    k = (true_theta / true_theta[0])[1]

    true_theta_1 = np.linspace(-3,3, 100)
    true_theta_2 = k * true_theta_1

    plt.figure(figsize=(8, 6), dpi=100)

    features = np.array([X_sample[x[0]] - X_sample[x[1]] for x in auxiliary['comparisons']])

    plt.grid()
    plt.title("$x_i - x_j$ of selected pairs", fontsize=16)

    plt.quiver(0, 0, true_theta[0], true_theta[1], width = 0.005)
    
    plt.plot(-true_theta_2, true_theta_1, color="blue")
    plt.legend(['$\\theta^T (x_i - x_j) = 0$', 'True $\\theta$'])

    c = Counter([tuple(x) for x in auxiliary['comparisons']])
    sizes = [(c[tuple(point)])*10 for point in auxiliary['comparisons']]

    plt.scatter(features.T[0], features.T[1], c=range(len(features)), s=sizes, cmap='Wistia', zorder=12)
    plt.xlabel('$x_{i,1} - x_{j,1}$', fontsize=16)
    plt.ylabel('$x_{i,1} - x_{j,1}$', fontsize=16)
    
    plt.axis('equal')
    plt.show()


def get_ranking_loss(X_test, theta_hat, theta_star):

    estimated_scores = X_test.dot(theta_hat)
    estimated_order = rankdata(estimated_scores)

    true_scores = X_test.dot(theta_star)
    true_order = rankdata(true_scores)

    distance = kendalldist(estimated_order, true_order)

    if math.isnan(distance):
        # everyone item has the same rank, order is arbitrary...
        return 0.5
    else:
        return distance

def kendalldist(*args, **kwargs):
    '''Return the normalized Kendall tau distance (0 <= tau <= 1)
    This function wraps around scipy.stats.kendalltau()
    and forwards all arguments to that function.
    '''
    tau_coef, p_value = kendalltau(*args, **kwargs)
    return kendall_convert_coef_to_dist(tau_coef)

def kendall_convert_coef_to_dist(tau_coef):
    '''Convert kendall correlation coefficient to normalized distance'''
    return (1 - tau_coef) / 2
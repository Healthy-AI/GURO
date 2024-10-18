import numpy as np
import pickle
from scipy.special import expit
import matplotlib.pyplot as plt
import itertools

from data_generator import DataGenerator
from envs import BaseEnv
from algorithms import (
    UniformSampling,
    GURO,
    BayesGURO,
    BALD,
    CoLSTIM,
    NormMin
)
from experiments import Simulation


def get_bounds(model_info, Z, theta):
    bounds = []
    epsilon = 0.2
    d = len(theta)
    C1 = 3 * ((1 + np.linalg.norm(theta)) ** 2)
    for i in range(len(model_info)):
        state = model_info[i]
        T = 10 * (i + 1)
        theta_hat = state["theta_hat"]
        H_inv = state["inv_hessian"] / T
        sum = 0
        for z in Z:
            Delta = np.abs(expit(z.dot(theta)) - 1 / 2)
            matrix_norm = np.sqrt(z.dot(H_inv).dot(z))
            w_n = (
                matrix_norm * expit(z.dot(theta_hat)) *
                (1 - expit(z.dot(theta_hat)))
                + 1e-12
            )
            delta_1 = np.exp(-(Delta**2) * T / (8 * d * C1 * (w_n**2)))
            delta_2 = np.exp(-Delta * T / (d * C1 * (matrix_norm**2)))

            sum += min(d * T * 2 * (delta_1 + delta_2), 1)
        norm_sum = sum / len(Z)

        current_bound = min(norm_sum / epsilon, 1)

        bounds.append(current_bound)
    return bounds


if __name__ == "__main__":
    # Setup exp
    n_seeds = 1
    update_every = 10
    iterations = 2000

    noise_factor = 0.5
    # for each seed run uniform sampling, guro, bayesguro, bald, colstim
    # generate data
    generator = DataGenerator(d=10, seed=0)
    X, y = generator.generate_noisy_data(100, 1)

    combinations = np.array(
        list(
            itertools.combinations(
                range(X.shape[0]),
                2)))

    # x_i - x_j for all combinations of i and j
    Z = np.array([X[c[0]] - X[c[1]] for c in combinations])

    true_model = generator.theta

    experiments = ["Uniform", "GURO", "NormMin",
                   "BayesGURO", "BALD", "CoLSTIM"]

    # create environment
    results = {
        "Uniform": {"loss": [], "upper_bound": [], "aux": []},
        "GURO": {"loss": [], "upper_bound": [], "aux": []},
        "NormMin": {"loss": [], "upper_bound": [], "aux": []},
        "BayesGURO": {"loss": [], "upper_bound": [], "aux": []},
        "BALD": {"loss": [], "upper_bound": [], "aux": []},
        "CoLSTIM": {"loss": [], "upper_bound": [], "aux": []},
    }

    for seed in range(n_seeds):
        print("Seed:", seed)

        # run algorithms
        for exp in experiments:

            if exp == "Uniform":
                alg = UniformSampling(X, update_every=update_every, seed=seed)
            elif exp == "GURO":
                alg = GURO(X, update_every=update_every, seed=seed)
            elif exp == "NormMin":
                alg = NormMin(X, update_every=update_every, seed=seed)
            elif exp == "BayesGURO":
                alg = BayesGURO(X, update_every=update_every,
                                post_sample_size=5, seed=seed)
            elif exp == "BALD":
                alg = BALD(X, update_every=update_every, seed=seed)
            elif exp == "CoLSTIM":
                c = np.sqrt(X.shape[1] * np.log(iterations))
                alg = CoLSTIM(X, update_every=update_every, c=c, seed=seed)

            env = BaseEnv(X, y, true_model,
                          noise_factor=noise_factor, seed=seed)
            sim = Simulation(
                alg, env, test_set=None, collect_model_info=True)
            loss, aux = sim.run(iterations, eval_steps=update_every)
            results[exp]["loss"].append(loss)
            results[exp]["aux"].append(aux)
            results[exp]["upper_bound"].append(
                get_bounds(aux["model_info"], Z, true_model)
            )

    # save results
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Generate plots

    # plot average upper bound vs iterations
    epsilon = 0.2
    plt.figure()

    algs_to_plot = ["Uniform", "GURO",
                    "BayesGURO", "BALD", "CoLSTIM", "NormMin"]
    markers = ["o", "v", "^", "s", "p", "*"]
    m_every = 30
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    plt.rc('font', size=17, family='serif')
    for i in range(len(algs_to_plot)):
        alg = algs_to_plot[i]

        avg = np.mean(results[alg]["upper_bound"], axis=0)
        std = np.std(results[alg]["upper_bound"], axis=0)
        x = range(0, iterations, update_every)

        plt.plot(
            x,
            avg,
            zorder=2,
            linewidth=1.8,
            marker=markers[i],
            markersize=8,
            markeredgewidth=1.8,
            markevery=m_every + i * m_every//5,
            label=alg,
            c=colors[i]
        )

        plt.fill_between(x, avg - std, avg + std, alpha=0.15,
                         zorder=1, color=colors[i])

    plt.xlim(0, iterations)
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel("Comparisons")
    plt.ylabel(rf"Upper bound on $P(R(\theta_t) \geq {epsilon})$")
    plt.grid()
    plt.savefig("upper_bounds_new.pdf")

    # plot average loss vs iterations
    plt.figure()

    for i in range(len(algs_to_plot)):
        alg = algs_to_plot[i]

        avg = np.mean(results[alg]["loss"], axis=0)[1:]
        std = np.std(results[alg]["loss"], axis=0)[1:]

        x = range(1, iterations, update_every)

        plt.plot(
            x,
            avg,
            zorder=2,
            linewidth=1.8,
            marker=markers[i],
            markersize=8,
            markeredgewidth=1.8,
            markevery=m_every + i * m_every//5,
            label=alg,
            c=colors[i]
        )

        plt.fill_between(x, avg - std, avg + std, alpha=0.15,
                         zorder=1, color=colors[i])

    plt.rc('font', size=17, family='serif')
    plt.xlim(0, iterations)
    plt.ylim(0, 0.3)
    plt.legend()
    plt.xlabel("Comparisons")
    plt.ylabel(r"$R(\theta_t)$")
    plt.grid()
    plt.savefig("loss_new.pdf")

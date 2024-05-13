import numpy as np
import scipy
import math
import matplotlib.pyplot as plt


def p1_part_A(N=64):
    # distribution of r for a single draw of x
    g = scipy.signal.windows.gaussian(N, std=5) * np.cos(2 * 0.1 * np.pi * np.arange(N))

    X = 2 * np.random.rand(N)

    rate = math.exp(np.dot(g, X))
    r = np.random.poisson(rate, 1000)

    plt.figure()
    plt.hist(r, bins=20)
    plt.title("Distribution of r for a single draw of x")
    plt.xlabel("r")
    plt.ylabel("Frequency")

    # distribution of r for 1000 draws of x
    X = 2 * np.random.rand(1000, N)
    rate = np.exp(X @ g)
    r = np.random.poisson(rate, 1000)

    plt.figure()
    plt.hist(r, bins=20)
    plt.title("Distribution of r for different draws of x")
    plt.xlabel("r")
    plt.ylabel("Frequency")

    return


def p1_part_B(N=64, M=100):
    g = scipy.signal.windows.gaussian(N, std=5) * np.cos(2 * 0.1 * np.pi * np.arange(N))

    X = 2 * np.random.rand(M, N)

    rate = np.exp(X @ g)
    r = np.random.poisson(rate, M)

    g_hat = np.linalg.inv(X.T @ X) @ X.T @ r  # Gaussian MLE = least squares solution

    return g, g_hat


def p1_part_C(N=64, M=100):
    g = scipy.signal.windows.gaussian(N, std=5) * np.cos(2 * 0.1 * np.pi * np.arange(N))

    X = 2 * np.random.rand(M, N)

    rate = np.exp(X @ g)
    r = np.random.poisson(rate, M)

    def poisson_mle_cost(g):
        return -(r * (X @ g) - np.exp(X @ g)).sum()

    g_hat = scipy.optimize.minimize(poisson_mle_cost, np.zeros(N)).x

    return g, g_hat


def p1_part_D_gaussian(N=64, M=100, sig_post=0.2, sig_prior=1):
    g = scipy.signal.windows.gaussian(N, std=5) * np.cos(2 * 0.1 * np.pi * np.arange(N))

    X = 2 * np.random.rand(M, N)

    rate = np.exp(X @ g)
    r = np.random.poisson(rate, M)

    def gaussian_map_cost(g):
        return -(
            -0.5
            * ((r - X @ g).T @ np.linalg.inv(sig_post**2 * np.eye(M)) @ (r - X @ g))
        ).sum() + 0.5 * (g.T @ np.linalg.inv(sig_prior**2 * np.eye(N)) @ g)

    g_hat = scipy.optimize.minimize(gaussian_map_cost, np.zeros(N)).x

    return g_hat


def p1_part_D_poisson(N=64, M=100, sig_prior=1):
    g = scipy.signal.windows.gaussian(N, std=5) * np.cos(2 * 0.1 * np.pi * np.arange(N))

    X = 2 * np.random.rand(M, N)

    rate = np.exp(X @ g)
    r = np.random.poisson(rate, M)

    def poisson_map_cost(g):
        return -(r * (X @ g) - np.exp(X @ g)).sum() + 0.5 * (
            np.diff(g).T @ np.linalg.inv(sig_prior**2 * np.eye(N - 1)) @ (np.diff(g))
        )

    g_hat = scipy.optimize.minimize(poisson_map_cost, np.zeros(N)).x

    return g_hat

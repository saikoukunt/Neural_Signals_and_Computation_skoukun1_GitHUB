import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import jPCA
from jPCA.util import load_churchland_data, plot_projections
from sklearn.decomposition import PCA


def p3_part_A(data):
    plt.imshow(data[26])
    plt.xlabel("Timepoint")
    plt.ylabel("Neuron")
    plt.title("Raster plot of neuron activity for condition 27")


# commented code below is for likelihood optimization, but it takes a very long time
# to run, so I return the analytical least squares solution averaged across trials instead
def p3_part_B_opt(data):
    dt = 0.01

    # def gaussian_mle_cost(A, x, sigma, dt, N):
    #     A = A.reshape((N, N))

    #     cost = (
    #         (x[:, :-1] - A @ x[:, :-1] * dt - x[:, :-1]).T
    #         @ np.linalg.inv(np.eye(N) * sigma**2)
    #         @ (x[:, :-1] - A @ x[:, :-1] * dt - x[:, :-1])
    #     ).sum()

    #     return cost

    # A_hat = scipy.optimize.minimize(
    #     gaussian_mle_cost,
    #     np.zeros((218 * 218)),
    #     args=(data[26], 0.1, 0.01, 218),
    #     options={"maxiter": 100, "disp": True},
    #     method="L-BFGS-B",
    # ).x

    A_hat = np.zeros((218, 218))

    for i in range(data.shape[0]):
        try:
            A_hat += (
                1
                / data.shape[0]
                * np.linalg.inv(data[i][:, :-1] @ data[i][:, :-1].T)
                @ data[i][:, :-1]
                @ ((data[i][:, 1:].T - data[i][:, :-1].T) / dt)
            )
        except np.linalg.LinAlgError:
            pass

    plt.imshow(A_hat.reshape((218, 218)))
    plt.title("Estimated dynamics matrix A_hat")
    plt.colorbar()

    return A_hat


def p3_part_B_error(A_hat, data):
    dt = 0.01
    pred_all = np.zeros((data.shape[0], 218, 60))
    errors = np.zeros((data.shape[0], 218, 60))

    for i in range(pred_all.shape[0]):
        x = data[i][:, :-1]
        pred_all[i] = (A_hat @ x) * dt + x

        errors[i] = (data[i][:, 1:] - pred_all[i]) ** 2

    plt.hist(errors.flatten(), bins=100)
    plt.title("Histogram of prediction errors")
    plt.xlabel("Prediction error")
    plt.ylabel("Frequency")

    return pred_all, errors


def p3_part_C_opt(data):
    all_trials = np.hstack((data)).T

    pca = PCA(n_components=6)
    all_trials_pca = pca.fit_transform(all_trials)
    pca_trials = all_trials_pca.reshape((108, 61, -1))

    def gaussian_mle_cost(A, x, sigma, dt, N):
        A = A.reshape((N, N))
        cost = 0

        for i in range(x.shape[0]):
            cost += (
                (x[i, 1:, :].T - A @ x[i, :-1, :].T * dt - x[i, :-1, :].T).T
                @ np.linalg.inv(np.eye(N) * sigma**2)
                @ (x[i, 1:, :].T - A @ x[i, :-1, :].T * dt - x[i, :-1, :].T)
            ).sum()

        return cost

    A_hat = scipy.optimize.minimize(
        gaussian_mle_cost,
        np.zeros((6 * 6)),
        args=(pca_trials, 0.1, 0.01, 6),
        options={"disp": True},
        method="L-BFGS-B",
    ).x

    plt.imshow(A_hat.reshape(6, 6))
    plt.title("Estimated PCA dynamics matrix A_hat")
    plt.colorbar()

    return A_hat, pca_trials, pca


def p3_part_C_error(A_hat, pca_trials, data, pca):
    dt = 0.01
    pred_all = np.zeros((108, 6, 60))
    errors = np.zeros((108, 218, 60))

    for i in range(pred_all.shape[0]):
        x = pca_trials[i, :-1, :]
        pred_all[i] = (A_hat.reshape(6, 6) @ x.T) * dt + x.T

        pca_proj = pca.inverse_transform(
            pred_all[i].T
        ).T  # inverse transform automatically adds the mean back

        errors[i] = (data[i, :, 1:] - pca_proj) ** 2

    plt.hist(errors.flatten(), bins=100)
    plt.title("Histogram of prediction errors")
    plt.xlabel("Prediction error")
    plt.ylabel("Frequency")

    return pred_all, errors


def p3_part_D(pred_all_pca):
    for i in range(108):
        plt.plot(pred_all_pca[i, 0, :], pred_all_pca[i, 1, :], linewidth=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("First two dynamical PCs for all trials/conditions")


def p3_part_E(data, A_hat_pca):
    datas, times = load_churchland_data("./exampleData.mat")

    jpca = jPCA.JPCA(num_jpcs=6)

    (projected, full_data_var, pca_var_capt, jpca_var_capt) = jpca.fit(
        datas, times=times, tstart=-50, tend=150
    )

    plot_projections(projected)
    plt.title("JPCA projections of neural data")
    plt.xlabel("jPC1")
    plt.ylabel("jPC2")

    return jpca, np.linalg.eig(A_hat_pca)[0], np.linalg.eig(jpca.M_skew)[0]

import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF

def run_pca(pixel_vs_time, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(pixel_vs_time)
    return pca, pca.transform(pixel_vs_time)

def run_nmf(pixel_vs_time, n_components):
    nmf = NMF(n_components=n_components)
    nmf.fit(pixel_vs_time)
    return nmf, nmf.transform(pixel_vs_time)

def run_ica(pixel_vs_time, n_components):
    ica = FastICA(n_components=n_components)
    ica.fit(pixel_vs_time)

    # mean squared error calculation
    err = np.sum((pixel_vs_time - ica.inverse_transform(ica.transform(pixel_vs_time)))**2)

    return ica, ica.transform(pixel_vs_time), err
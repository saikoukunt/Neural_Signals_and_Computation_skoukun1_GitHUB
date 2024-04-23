import scipy
import matplotlib.pyplot as plt
import numpy as np

def plot_mean(images):
    plt.figure()
    plt.imshow(images.mean(axis=0)); 
    plt.title("Mean Frame")

def plot_median(images):
    plt.figure()
    plt.imshow(np.median(images, axis=0))
    plt.title("Median Frame")

def plot_variance(images):
    plt.figure()
    plt.imshow(images.var(axis=0))
    plt.title("Frame Variance")

def plot_max(images):
    plt.figure()
    plt.imshow(images.max(axis=0))
    plt.title("Max Frame")

def plot_80th(images):
    plt.figure()
    plt.imshow(np.percentile(images, 80, axis=0))
    plt.title("80th Percentile Frame")
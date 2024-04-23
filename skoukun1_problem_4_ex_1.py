import numpy as np


def calc_time_trace(images, roi):
    return images[:, roi.astype(bool)].mean(axis=1)

import cv2
import numpy as np


def find_rois(summary_img, n_roi):
    # normalize pixel values from 0 to 255
    summary_img = cv2.normalize(summary_img, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # find connected components with gaussian adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        summary_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=2,
    )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, 8, cv2.CV_32S
    )

    # sort by area
    areas = stats[:, cv2.CC_STAT_AREA]
    sorted_areas = np.argsort(areas)[::-1]

    # return the n_roi largest components
    rois = np.zeros((n_roi, summary_img.shape[0], summary_img.shape[1]))
    for i in range(n_roi):
        rois[i] = (labels == sorted_areas[i + 2]).astype(
            np.uint8
        )  # i+2 to ignore the background

    return rois

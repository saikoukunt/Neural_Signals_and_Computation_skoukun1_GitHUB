import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2


def play_video(video_path):
    images = io.imread(video_path)
    fig = plt.figure() 
    image = plt.imshow(images[0])

    # function to update figure
    def update(j):
        image.set_array(images[j])
        return [image]
    # start animation
    ani = animation.FuncAnimation(fig, update, frames=range(images.shape[0]), 
                                interval=1, blit=True)

def spatial_corr(img1, img2):
    # 2d convolution gives an unnormalized spatial correlation
    corr = cv2.filter2D(img1, ddepth=-1, kernel=img2, borderType = 0)

    return corr
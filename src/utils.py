"""
Utils for LSCI video processing
"""
import os
from pathlib import Path
from typing import Tuple
from natsort import natsorted
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter1d

def update_sums(sum_s: np.ndarray,
                sum_s2:np.ndarray,
                new_frame: np.ndarray,
                old_frame:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    """
    Updates the sum and squared sum with the given new and old frames

    Parameters
    ------------
    sum_s: np.ndarray
        Current sum over the window
    sum_s2: np.ndarray
        Current sum of squares over the window
    new_frame: np.ndarray
        Newly added frame
    old_frame: np.ndarray
        Newly removed frame
    """
    sum_s += new_frame - old_frame 
    sum_s2 += new_frame**2 - old_frame**2

    return sum_s, sum_s2

def calculate_contrast_from_sums(sum_s:np.ndarray, sum_s2:np.ndarray, window:int)->np.ndarray:
    """
    Calculate the speckle constrast from the sum over the temporal window and the sum of squares

    Parameters
    --------------
    sum_s: np.ndarray (shape (H, W))
        Sum along time axis for the current window 
    sum_s2: np.ndarray (shape (H, W))
        Sum of squared pixel values along time axis for the current window
    window: int
        Window size

    Return
    -----------
    contrast: np.ndarray (shape (H, W))
        Speckle contrast image for the current window
    """
    mean = sum_s/window
    var = (sum_s2/window) - (mean**2)
    var = np.clip(var, 0, None)
    # print('variance: ', var)
    return np.sqrt(var)/np.clip(mean, 1e-15, None)


def temporal_contrast(raw_video:np.ndarray, window_size:int, baseline:np.ndarray=None)->Tuple[np.ndarray, int]:
    """
    Calculate the temporal contrast for a given video

    window size must be odd
    """
    raw_video = raw_video.astype(np.float32)
    # get dimensions
    stack_size, width, height = raw_video.shape

    n_processed_frames = stack_size-window_size+1

    # create array for lsci images that will be calculated
    processed_frames = np.zeros([n_processed_frames, width, height], dtype=np.float32)

    for i in range(n_processed_frames):
        frames_window = raw_video[i:i+window_size, :, :]
        print('frames window max: ', np.max(frames_window))

        if i==0:
        # for first frame
            sum_window = np.sum(frames_window, axis=0)
            sum_squares_window = np.sum(frames_window ** 2, axis=0)

            # print('sum window max: ', np.max(sum_window))
            # print('sum squares max: ', np.max(sum_squares_window))

            # current_mean = np.mean(frames_window, axis=0)
            # current_variance = np.var(frames_window, axis=0)

        else:
        # for all other frames, just update the mean and variance
            new_frame = frames_window[-1]
            sum_window += new_frame - old_frame
            sum_squares_window += new_frame**2 - old_frame**2
            # print('sum window max: ', np.max(sum_window))
            # print('sum squares max: ', np.max(sum_squares_window))
            # current_mean = update_mean(current_mean, new_frame, old_frame, window_size)
            # current_variance = update_variance(current_variance, current_mean, new_frame, old_frame, window_size)

        # calculate contrast and add to array
        contrast = calculate_contrast_from_sums(sum_window, sum_squares_window, window_size)
        print('contrast max: ', np.max(contrast))

        if baseline is None:
            processed_frames[i, :, :] = contrast
        else:
            processed_frames[i, :, :] = contrast - baseline

        print('processed frames max: ', processed_frames.max())
        

        old_frame = frames_window[0]

    return processed_frames, n_processed_frames


def read_folder_of_frames(folder_path:Path)->np.ndarray:
    filenames = os.listdir(folder_path)
    filenames = natsorted(filenames)
    images = [np.array(Image.open(folder_path/file), dtype=np.float32) for file in filenames if os.path.isfile(folder_path/file)]

    
    stacked_images = np.stack(images, axis=0)
    return stacked_images


def plot_and_save_profile(vid1:np.ndarray,
                          label1:str,
                          vid2:np.ndarray,
                          label2:str,
                          fps:int,
                          out_path:Path):
    # Setup
    writer = animation.writers['ffmpeg']
    writer = writer(fps=fps)

    print('vid1 shape: ', vid1.shape)
    print('vid2 shape: ', vid2.shape)

    middle_line_values1 = vid1[:,:, vid1.shape[2]//2]
    middle_line_values2 = vid2[:,:, vid2.shape[2]//2]

    nb_frames = max(vid1.shape[0], vid2.shape[0])

    # Create a plot for the middle vertical line pixel profile
    fig = plt.figure()
    with writer.saving(fig, out_path.as_posix(), nb_frames):
        for i in range(nb_frames):
            fig.clear()
            plt.ylim(0, 255)
            
            if i < len(middle_line_values1):
                plt.plot(gaussian_filter1d(middle_line_values1[i], sigma=6), color='blue', label=label1)
                
            if i < len(middle_line_values2):
                plt.plot(gaussian_filter1d(middle_line_values2[i], sigma=6), color='red', label=label2)

            plt.legend()
            writer.grab_frame()
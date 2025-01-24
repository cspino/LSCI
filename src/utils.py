"""
Utils for LSCI video processing
"""

import numpy as np

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
    return np.sqrt(var)/np.clip(mean, 1e-25, None)


def temporal_contrast(raw_video:np.ndarray, window_size:int, baseline:np.ndarray=None)->np.ndarray:
    """
    Calculate the temporal contrast for a given video

    window size must be odd
    """
    raw_video = raw_video.astype(np.float32)
    # get dimensions
    stack_size, width, height = raw_video.shape

    n_processed_frames = stack_size-window_size+1

    # create array for lsci images that will be calculated
    processed_frames = np.zeros([n_processed_frames, width, height])

    for i in range(n_processed_frames):
        frames_window = raw_video[i:i+window_size, :, :]

        if i==0:
        # for first frame
            sum_window = np.sum(frames_window, axis=0)
            sum_squares_window = np.sum(frames_window ** 2, axis=0)

            # current_mean = np.mean(frames_window, axis=0)
            # current_variance = np.var(frames_window, axis=0)

        else:
        # for all other frames, just update the mean and variance
            new_frame = frames_window[-1]
            sum_window += new_frame - old_frame
            sum_squares_window += new_frame**2 - old_frame**2
            # current_mean = update_mean(current_mean, new_frame, old_frame, window_size)
            # current_variance = update_variance(current_variance, current_mean, new_frame, old_frame, window_size)

        # calculate contrast and add to array
        contrast = calculate_contrast_from_sums(sum_window, sum_squares_window, window_size)

        if baseline is None:
            processed_frames[i, :, :] = contrast
        else:
            processed_frames[i, :, :] = contrast - baseline
        

        old_frame = frames_window[0]

    return processed_frames, n_processed_frames
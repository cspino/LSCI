"""
Utils for LSCI video processing
"""
import os
import cv2
from pathlib import Path
from typing import Tuple, Optional
from natsort import natsorted
import numpy as np
import skvideo.io
from tqdm import tqdm
from scipy.ndimage import uniform_filter


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
    return np.sqrt(var)/np.clip(mean, 1e-15, None)


def calculate_bfi_from_sums(sum_s:np.ndarray, sum_s2:np.ndarray, window:int)->np.ndarray:
    """
    Calculate the blood flow index from the sum over the temporal window and the sum of squares

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
        BFI image for the current window
    """
    mean = sum_s/window
    mean_squared = mean**2
    var = (sum_s2/window) - mean_squared

    return mean_squared/np.clip(var, 1e-15, None)


def temporal_contrast(raw_video:np.ndarray, window_size:int)->np.ndarray:
    """
    Compute the temporal contrast

    Parameters
    -------------
    raw_video: np.ndarray
        Stack of frames to process
    window_size: int
        Temporal window size

    Return
    --------------
    processed_frames: np.ndarray
    """
    raw_video = raw_video.astype(np.float32)
    stack_size, width, height = raw_video.shape
    n_processed_frames = stack_size-window_size+1

    # create array for lsci images that will be calculated
    processed_frames = np.zeros([n_processed_frames, width, height], dtype=np.float32)

    for i in tqdm (range(n_processed_frames), desc="Processing"):
        frames_window = raw_video[i:i+window_size, :, :]

        if i==0:
        # for first frame
            sum_window = np.sum(frames_window, axis=0)
            sum_squares_window = np.sum(frames_window ** 2, axis=0)

        else:
        # for all other frames, just update the mean and variance
            new_frame = frames_window[-1]
            sum_window += new_frame - old_frame
            sum_squares_window += new_frame**2 - old_frame**2


        # calculate contrast and add to array
        contrast = calculate_contrast_from_sums(sum_window, sum_squares_window, window_size)
        processed_frames[i, :, :] = contrast
        old_frame = frames_window[0]

    return processed_frames


def temporal_bfi(raw_video:np.ndarray, window_size:int, sum_window:Optional[np.ndarray]=None, 
                 sum_squares_window:Optional[np.ndarray]=None, old_frame:Optional[np.ndarray]=None)->Tuple[np.ndarray, int]:
    """
    Compute the temporal blood flow index (1/K**2)

    Parameters
    -------------
    raw_video: np.ndarray
        Stack of frames to process
    window_size: int
        Temporal window size
    sum_window, sum_squares_window, old_frame: np.ndarray or None
        If provided, the processing continues from these values
        Otherwise, processing is started anew

    Returns
    --------------
    processed_frames: np.ndarray
    frames window: np.ndarray
        Last window of frames
    (sum_window, sum_squares_window, old_frame): np.ndarray
        Values used for processing. These are useful if a video is being processed in batches
    """
    raw_video = raw_video.astype(np.float32)
    stack_size, width, height = raw_video.shape
    n_processed_frames = stack_size - window_size + 1

    # create array for lsci images that will be calculated
    processed_frames = np.zeros([n_processed_frames, width, height], dtype=np.float32)

    for i in tqdm (range(n_processed_frames), desc="Processing"):
        frames_window = raw_video[i:i+window_size, :, :]

        if sum_window is None:
        # for first frame
            sum_window = np.sum(frames_window, axis=0)
            sum_squares_window = np.sum(frames_window ** 2, axis=0)

        else:
        # for all other frames, just update the mean and variance
            new_frame = frames_window[-1]
            sum_window += new_frame - old_frame
            sum_squares_window += new_frame**2 - old_frame**2

        # calculate contrast and add to array
        processed_frames[i, :, :] =  calculate_bfi_from_sums(sum_window, sum_squares_window, window_size)

        old_frame = frames_window[0]

    return processed_frames, frames_window, (sum_window, sum_squares_window, old_frame)


def spatial_contrast(raw_video:np.ndarray, kernel_size:Tuple[int, int])->np.ndarray:
    """
    Compute the spatial contrast for an entire video

    Parameters
    -------------
    raw_video: np.ndarray
        Stack of frames to process
    kernel_size: (int, int)
        Spatial window size

    Return
    --------------
    processed_frames: np.ndarray
    """
    raw_video = raw_video.astype(np.float32)
    processed_frames = np.zeros(raw_video.shape, dtype=np.float32)
    
    for i in tqdm (range(raw_video.shape[0]), desc="Processing"):
        processed_frames[i] = spatial_one_frame(raw_video[i], kernel_size)

    return processed_frames


def spatial_one_frame(frame:np.ndarray, kernel_size:Tuple[int, int])->np.ndarray:
    """
    Compute the spatial contrast for a single frame

    Parameters
    -------------
    frame: np.ndarray
        Frame to process
    kernel_size: (int, int)
        Spatial window size

    Return
    --------------
    processed frame: np.ndarray
    """
    mean = uniform_filter(frame, size=kernel_size, mode='nearest')
    std = np.sqrt(uniform_filter(frame**2, size=kernel_size, mode='nearest') - mean**2)

    return np.divide(std, mean, out=np.zeros_like(std), where=mean != 0)


def spatial_bfi(raw_video:np.ndarray, kernel_size:Tuple[int, int])->np.ndarray:
    """
    Compute the spatial blood flow index for an entire video

    Parameters
    -------------
    raw_video: np.ndarray
        Stack of frames to process
    kernel_size: (int, int)
        Spatial window size

    Return
    --------------
    processed_frames: np.ndarray
    """
    raw_video = raw_video.astype(np.float32)
    processed_frames = np.zeros(raw_video.shape, dtype=np.float32)
    
    for i in tqdm (range(raw_video.shape[0]), desc="Processing"):
        processed_frames[i] = spatial_bfi_one_frame(raw_video[i], kernel_size)

    return processed_frames


def spatial_bfi_one_frame(frame:np.ndarray, kernel_size:Tuple[int, int])->np.ndarray:
    """
    Compute the spatial blood flow index for a single frame

    Parameters
    -------------
    frame: np.ndarray
        Frame to process
    kernel_size: (int, int)
        Spatial window size

    Return
    --------------
    processed frame: np.ndarray
    """
    mean = uniform_filter(frame, size=kernel_size, mode='nearest')
    mean_squared = mean**2
    var = uniform_filter(frame**2, size=kernel_size, mode='nearest') - mean_squared

    return np.divide(mean_squared, var, out=np.zeros_like(mean_squared), where=var != 0)


def spatial_bfi_frame(frame:np.ndarray, kernel_size:Tuple[int, int])->np.ndarray:
    """
    Used for real-time spatial processing. 
    Each frame is clipped independently since we don't have access to every frame.

    Parameters
    -------------
    frame: np.ndarray
        Frame to process
    kernel_size: (int, int)
        Spatial window size

    Return
    --------------
    processed frame: np.ndarray
    """
    flowmap = spatial_bfi_one_frame(frame, kernel_size)

    clip_to = np.percentile(flowmap, 95)
    clip_to = np.max(flowmap[flowmap<=clip_to])
    return np.clip(flowmap, 0, clip_to)


def read_folder_of_frames(folder_path:Path)->np.ndarray:
    """
    Util for reading a folder of individual frames

    Parameter
    -------------
    folder_path: Path
        Path to the folder of images

    Return
    --------------
    stacked images: np.ndarray
        3d numpy array containing all frames
    """
    filenames = os.listdir(folder_path)
    filenames = natsorted(filenames)
    images = [cv2.imread(str(folder_path / file), cv2.IMREAD_UNCHANGED).astype(np.float32)
              for file in filenames if os.path.isfile(folder_path / file)]
    
    return np.stack(images, axis=0)


def read_video(vid_path:Path, 
               default_fps:int=30, 
               default_shape:Tuple[int, int]=(1080, 1920)
               )->Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Reads a video from given path and creates a 3d numpy array
    Compatible formats include:
    - A folder of jpgs or pngs
    - .raw
    - .avi

    Parameters
    ------------
    vid_path: Path
    default_fps: int
        Only used if the format of the video doesn't contain the info
        (folders of frames and .raw format)
    default_shape: (int, int)
        Only used if the format of the video doesn't contain the info
        (.raw format)

    Returns
    -----------
    video: np.ndarray or None (in the case of an error)
    fps: int or None (in the case of an error)
    """
    if vid_path.is_dir():
        # video is a directory
        frame_rate = default_fps
        video = read_folder_of_frames(vid_path)[:,:,:]
        width = video.shape[2]
        height = video.shape[1]

    elif vid_path.suffix == ".raw":
        # video is a raw file
        height, width = default_shape # TODO these could be read from a params file?
        frame_rate = default_fps
        data = np.fromfile(vid_path, dtype=np.float32)
        num_frames = data.size // (width*height)
        video = data.reshape((num_frames, height, width))

    else:
        try:
            # Get info on video
            cap = cv2.VideoCapture(vid_path.as_posix())
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            video = skvideo.io.vread(vid_path.as_posix())[:,:,:,1]
        
        except Exception as e:
            print("Couldn't read video: ", vid_path, 'error: ', e)
            return None, None
    
    return video, frame_rate

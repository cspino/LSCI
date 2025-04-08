"""
Utils for LSCI video saving
"""
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
import cv2
from tqdm import tqdm
from typing import List

def plot_and_save_profiles(videos: List[np.ndarray], labels: List[str], fps: int, out_path: Path):
    """
    Plot and save pixel profiles of multiple videos. Works for any number of videos.
    
    Parameters
    ----------------
    videos: list of numpy arrays
    labels: List of labels corresponding to the videos
    fps: frame rate (all videos are assumed to have the same frame rate)
    out_path: Output path to save the animation
    """
    # Check that the number of videos matches the number of labels
    assert len(videos) == len(labels), "The number of videos must match the number of labels."

    # Setup
    writer = animation.writers['ffmpeg']
    writer = writer(fps=fps)

    # Calculate average along x axis for each frame
    x_averages = [np.mean(video, axis=2) for video in videos]  # Calculate for all videos

    nb_frames = max(video.shape[0] for video in videos)  # Max number of frames across videos
    range_max = max(video.max() for video in videos) # Max value across videos

    # Create a plot for the middle vertical line pixel profile
    fig = plt.figure(dpi=600, figsize=(16,10))

    with writer.saving(fig, out_path.as_posix(), nb_frames):
        for i in range(nb_frames):
            fig.clear()
            plt.ylim(0, range_max)
            
            # Plot the pixel profiles for each video
            for idx, (line_values, label) in enumerate(zip(x_averages, labels)):
                if i < len(line_values):
                    plt.plot(line_values[i], label=label, color=f"C{idx}")

            plt.legend(loc='upper right')
            writer.grab_frame()
 

def save_flowmap(stack:np.ndarray, output_path:Path, fps:int):
    """
    Save a stack of frames as a .avi video

    Parameters
    ---------------
    stack: np.ndarray
        Stack of frames to save
    output_path: Path
        Path of .avi video
    fps: int
        Frame rate
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, (stack.shape[2], stack.shape[1]))

    stack = 255*(stack-stack.min())/(stack.max()-stack.min()) # Normalize 
    stack = stack.astype(np.uint8)

    for i in tqdm(range(stack.shape[0]), desc="Saving video", disable=False):
        vidout = cv2.resize(stack[i], (stack.shape[2], stack.shape[1]))
        vidout = cv2.cvtColor(vidout, cv2.COLOR_GRAY2BGR)
        video_writer.write(vidout)
    
    # Release the VideoWriter
    video_writer.release()

def save_speckle(speckle:np.ndarray, file_path:Path, frame_rate:int):
    """
    Save a speckle video in .avi format

    Parameters
    ---------------
    speckle: np.ndarray
        Stack of frames to save
    file_path: Path
        Output path of .avi video
    frame_rate: int
    """
    clip_value = 0.5*speckle.max() # For visualization
    speckle = np.clip(speckle, 0, clip_value)
    
    save_flowmap(speckle, file_path, frame_rate)


def save_as_bytes(stack:np.ndarray, output_file:Path):
    """
    Save the given numpy array in raw format

    Parameters
    ---------------
    stack: np.ndarray
        Array to save
    output_file: Path
        Path to .raw file where array will be saved
    """
    with open(output_file, 'wb') as f:
        f.write(stack.tobytes())

def save_as_avi(vid_path:Path, norm_max:int, bytes_per_frame:int, frame_rate:int=30, batch_size:int=200):
    """
    Read the video saved at vid_path and re-save it in .avi format
    The video is read in batches (of size batch_size) to avoid an Out Of Memory error

    Parameters
    --------------
    vid_path: Path
        Path to the .raw file containing the video to re-save
    norm_max: int
        Max value in the video. Used to normalize every frame
    bytes_per_frame: int
        Number of bytes needed for a frame
    frame_rate: int
    batch_size: int
        Size of the batches that are read
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    out = cv2.VideoWriter((vid_path.with_suffix('.avi')).as_posix(), fourcc, frame_rate, (1920,1080), False)

    with open(vid_path, 'rb') as f:
        while True:
            buffer = f.read(bytes_per_frame*batch_size)
            if not buffer:
                break

            video = np.frombuffer(buffer, dtype=np.float32)
            video = video.reshape((-1, 1080, 1920))
            video = (video/norm_max)*255

            for i in range(video.shape[0]):
                frame = video[i]
                frame = frame.astype(np.uint8)
                out.write(frame)
        out.release()


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np
from utils import plot_and_save_profile
import cv2
from typing import Tuple

FRAME = 50

def read_video(video_path:Path)->Tuple[np.ndarray, int]:
    frames =[]
    cap = cv2.VideoCapture(video_path.as_posix())
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[:,:,1])
    cap.release()

    return np.array(frames), fps

if __name__ == "__main__":
    parser = ArgumentParser(
    prog = 'process_image',
    description = 'Get temporal contrast image from thorcam video feed',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--video_one',
                        required= True,
                        type = Path,
                        help = 'Path to video')
    parser.add_argument('-u', '--video_two',
                        required= True,
                        type = Path,
                        help = 'Path to video')
    parser.add_argument('-o', '--out',
                        required= True,
                        type = Path,
                        help = 'Output path')
    args = parser.parse_args()
    

    # Read frame
    frames1, fps1 = read_video(args.video_one)
    frames2, fps2 = read_video(args.video_two)

    # Get labels
    label1 = args.video_one.parent.name
    label2 = args.video_two.parent.name

    if fps1 != fps2:
        print("WARNING: videos don't have the same fps. Using the min.")
    
    # Plot middle profile
    plot_and_save_profile(frames1, label1, frames2, label2, min(fps1, fps2), args.out)




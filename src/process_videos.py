import os
import sys
sys.path.append('..') # this is to be able to import files in project folder

import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import List, Tuple, Optional
import cv2

from utils.utils import temporal_bfi, spatial_bfi, read_video
from utils.save_utils import save_as_avi, plot_and_save_profiles

BYTES_PER_FRAME = 1920*1080*np.dtype(np.float32).itemsize
BATCH_SIZE = 200


def plot_profile_from_path(filepath:Path, label:str, output_path:Path):
    try:
        video, fps = read_video(Path(filepath))
    except :
        print(f"Error while trying to read {filepath}. It might be too long")
        return
    
    plot_and_save_profiles([video], [label], fps, output_path)


def spatial_main(output_dir:Optional[Path],
                videos:List[Path],
                spatial_window:int,
                profile:bool):
    
    for vid_path in videos:
        print('Processing video: ', vid_path.as_posix())
        vid_name = vid_path.stem

        norm_max = None # reset normalization
        clip_to = None

        if output_dir is None:
            output_dir = vid_path.parent
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir/(vid_name+'_spatial_map.raw')

        with open(vid_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            while True:
                buffer = f_in.read(BYTES_PER_FRAME*BATCH_SIZE)
                if not buffer:
                    break

                video = np.frombuffer(buffer, dtype=np.float32)
                video = video.reshape((-1, 1080, 1920))

                flowmap = spatial_bfi(video, spatial_window)

                if not clip_to:
                    clip_to = np.percentile(flowmap, 99.9)
                    clip_to = np.max(flowmap[flowmap<=clip_to])
                flowmap = np.clip(flowmap, 0, clip_to)

                f_out.write(flowmap.tobytes())

                if norm_max:
                    norm_max = max(flowmap.max(), norm_max)
                else:
                    norm_max = flowmap.max()
        
        save_as_avi(output_path, norm_max, BYTES_PER_FRAME, batch_size=BATCH_SIZE)
        if profile:
            plot_profile_from_path(output_path, "Spatial", output_dir/(vid_name+'_spatial_profile.mp4'))


def temporal_main(output_dir:Optional[Path],
                  videos:List[Path],
                  temporal_window:int,
                  profile:bool):
    
    for vid_path in videos:
        print('Processing video: ', vid_path.as_posix())
        vid_name = vid_path.stem

        norm_max = None # reset normalization
        sum_window, sum_squares_window, old_frame = None, None, None
        window = None
        clip_to = None

        if output_dir is None:
            output_dir = vid_path.parent
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir/(vid_name+'_temporal_map.raw')

        with open(vid_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            while True:
                buffer = f_in.read(BYTES_PER_FRAME*BATCH_SIZE)
                if not buffer:
                    break

                video = np.frombuffer(buffer, dtype=np.float32)
                video = video.reshape((-1, 1080, 1920))

                if not window is None:
                    video = np.concatenate((window[1:], video), axis=0)

                flowmap, window, (sum_window, sum_squares_window, old_frame) = temporal_bfi(video, 
                                                                                    temporal_window,
                                                                                    sum_window, 
                                                                                    sum_squares_window, 
                                                                                    old_frame)

                # Use the same window level for every batch to avoid jumps
                if not clip_to:
                    clip_to = np.percentile(flowmap, 99.9)
                    clip_to = np.max(flowmap[flowmap<=clip_to])
                flowmap = np.clip(flowmap, 0, clip_to)

                f_out.write(flowmap.tobytes()) # Save to .raw file

                if norm_max:
                    norm_max = max(flowmap.max(), norm_max)
                else:
                    norm_max = flowmap.max()
        
        save_as_avi(output_path, norm_max, BYTES_PER_FRAME, batch_size=BATCH_SIZE)
        if profile:
            plot_profile_from_path(output_path, "Temporal", output_dir/(vid_name+'_temporal_profile.mp4'))

def main(output_dir:Optional[Path],
         videos:List[Path],
         temporal:bool,
         temporal_window:int,
         spatial:bool,
         spatial_window:Tuple[int,int],
         profile:bool):
    if temporal:
        temporal_main(output_dir, videos, temporal_window, profile)

    if spatial:
        spatial_main(output_dir, videos, spatial_window, profile)


if __name__=="__main__":
    parser = ArgumentParser(
    prog = 'process_videos',
    description = 'Get temporal or spatial speckle contrast of video, or the relative flow map',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--videos',
                        required = True,
                        type = Path,
                        nargs = '+',
                        help = 'Paths to videos to process')
    
    parser.add_argument('-sw', '--spatial_window',
                        default = (5,5),
                        type = tuple,
                        help = 'Spatial window size. 5 by default')
    parser.add_argument('-tw', '--temporal_window',
                        default = 10,
                        type = int,
                        help = 'Temporal window size. 10 by default')
    parser.add_argument('-o', '--output_dir',
                        default = None,
                        type = Path,
                        help = 'Directory where processed videos will be saved. By default, they are '
                        'saved in the same directory as the original video')
    
    parser.add_argument('--spatial',
                        dest = "spatial",
                        action = "store_true",
                        help = 'Use spatial contrast.')
    parser.add_argument('--temporal',
                        dest = "temporal",
                        action = "store_true",
                        help = 'Use temporal contrast.')
    
    parser.add_argument('--profile',
                        dest = "profile",
                        action = "store_true",
                        help = 'Save profile plots')
    
    args = parser.parse_args()

    assert args.spatial or args.temporal, "No contrast type chosen. Use at least one of [spatial, temporal]."

    main(args.output_dir,
         args.videos,
         args.temporal,
         args.temporal_window,
         args.spatial,
         args.spatial_window,
         args.profile)

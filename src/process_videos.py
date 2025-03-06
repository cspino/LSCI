
import sys
sys.path.append('..') # this is to be able to import files in project folder
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import os

from utils import temporal_contrast, spatial_contrast, read_video, get_relative_flowmap
from save_utils import save_relative_flow_map, save_speckle

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
    parser.add_argument('--show',
                        action = 'store_true',
                        help = 'Show videos as they are saved')
    
    parser.add_argument('--spatial',
                        dest = "spatial",
                        action = "store_true",
                        help = 'Use spatial contrast.')
    parser.add_argument('--temporal',
                        dest = "temporal",
                        action = "store_true",
                        help = 'Use temporal contrast.')
    
    parser.add_argument('--relative',
                        dest = "relative",
                        action = "store_true",
                        help = 'Compute relative flow maps.')
    
    args = parser.parse_args()

    if not args.spatial and not args.temporal:
        print('No contrast type chosen. Use at least one of [spatial, temporal].')
        sys.exit()

    output_dir = args.output_dir

    for vid_path in args.videos:
        print('Processing video: ', vid_path.as_posix())

        video, frame_rate = read_video(vid_path)
        vid_name = vid_path.stem

        if video is None:
            # Error reading video, we skip it
            continue

        if args.output_dir is None:
            output_dir = vid_path.parent
        
        os.makedirs(output_dir, exist_ok=True)

        if args.temporal:
            speckle, _ = temporal_contrast(video, args.temporal_window)
            filename = vid_name+f"_temporal.avi"
            save_speckle(speckle, output_dir/filename, frame_rate, args.show)

            if args.relative:
                flowmap = get_relative_flowmap(speckle)
                save_relative_flow_map(flowmap, output_dir/(vid_name+'_temporal_map.avi'), frame_rate, show=args.show)

        if args.spatial:
            speckle = spatial_contrast(video, args.spatial_window)
            filename = vid_name+f"_spatial.avi"
            save_speckle(speckle, output_dir/filename, frame_rate, args.show)

            if args.relative:
                flowmap = get_relative_flowmap(speckle)
                save_relative_flow_map(flowmap, output_dir/(vid_name+'_spatial_map.avi'), frame_rate, show=args.show)




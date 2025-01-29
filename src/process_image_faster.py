
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
from pathlib import Path

import time
import cv2
import skvideo.io
import numpy as np

from utils import temporal_contrast


if __name__=="__main__":
    parser = ArgumentParser(
    prog = 'process_image',
    description = 'Get temporal contrast image from raw video',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--video',
                        required = True,
                        type = Path,
                        help = 'Path to video to process')
    parser.add_argument('-o', '--output_dir',
                        required = True,
                        type = Path,
                        help = 'Path to output folder')
    parser.add_argument('-w', '--window_size',
                        default = 10,
                        type = int,
                        help = 'Temporal window size. 10 by default')
    parser.add_argument('-s', '--suffix',
                        default = None,
                        type = str,
                        help = 'Suffix to add to filename')
    parser.add_argument('--show',
                        action = 'store_true',
                        help = 'Show video as it is saved')
    parser.add_argument('-b', '--baseline',
                        default=False,
                        type = Path,
                        help = 'If provided, contrast will be relative to baseline')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Get baseline
    baseline_contrast = None
    if args.baseline:
        baseline_img = skvideo.io.vread(args.baseline.as_posix())[:,:,:,1]
        baseline_contrast, _ = temporal_contrast(baseline_img, args.window_size)
        baseline_contrast = np.mean(baseline_contrast, axis=0)

    # Get info on original video
    og_name = args.video.stem
    print(args.video.as_posix())
    cap = cv2.VideoCapture(args.video.as_posix())
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    print("FRAME RATE: ", frame_rate)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('dims: ', width, height)
    cap.release()

    # Process video
    print("Starting processing...")
    raw_speckle_img_seq = skvideo.io.vread(args.video.as_posix())[:,:,:,1]
    print(raw_speckle_img_seq.shape)
    t_lsci, num_frames = temporal_contrast(raw_speckle_img_seq, args.window_size, baseline_contrast)
    t_lsci = (t_lsci*255).astype(np.uint8)

    # save processed video
    print("Processing done. Now saving...")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    if args.suffix:
        filename = og_name+f"_processed_{args.suffix}.avi"
    else:
        filename = og_name+"_processed.avi"
    video_writer = cv2.VideoWriter((args.output_dir/filename).as_posix(), fourcc, frame_rate, (width, height))

    for i in range(num_frames):
        frame = t_lsci[i]

        # Convert grayscale to BGR by repeating channels
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # Grayscale
        # frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE) # Colormap
        video_writer.write(frame)

        if args.show:
            cv2.imshow('Frame', frame)

            # Wait for 1 ms and check if the user pressed the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release the VideoWriter
    video_writer.release()


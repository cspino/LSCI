
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
from pathlib import Path

import time
import cv2
import skvideo.io
# from skimage.exposure import equalize_adapthist
import numpy as np

from skimage.exposure import rescale_intensity
from tqdm import tqdm
from utils import spatial_contrast, read_folder_of_frames, get_relative_flowmap
from save_utils import save_relative_flow_map

if __name__=="__main__":
    parser = ArgumentParser(
    prog = 'spatial_contrast',
    description = 'Get spatial contrast from raw video',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--video',
                        required = True,
                        type = Path,
                        help = 'Path to video to process')
    parser.add_argument('-o', '--output_dir',
                        required = True,
                        type = Path,
                        help = 'Path to output folder')
    parser.add_argument('-w', '--kernel_size',
                        default = (5,5),
                        type = tuple,
                        help = 'window size. 10 by default')
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
        baseline_contrast, _ = spatial_contrast(baseline_img, args.window_size)
        baseline_contrast = np.mean(baseline_contrast, axis=0)

    # Get info on original video
    og_name = args.video.stem

    # Process video
    print("Starting processing...")
    # If video is a folder of frames
    if args.video.is_dir():
        print("video is a directory")
        frame_rate = 30 # TODO change this
        raw_speckle_img_seq = read_folder_of_frames(args.video)[:,:,:]
        width = raw_speckle_img_seq.shape[2]
        height = raw_speckle_img_seq.shape[1]

    elif args.video.suffix == ".raw":
        height = 1080 # TODO these could be read from a params file?
        width = 1920
        frame_rate = 30
        data = np.fromfile(args.video, dtype=np.float32)
        num_frames = data.size // (width*height)
        raw_speckle_img_seq = data.reshape((num_frames, height, width))

    else:
        # Get info on video
        cap = cv2.VideoCapture(args.video.as_posix())
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        # print("FRAME RATE: ", frame_rate)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        raw_speckle_img_seq = skvideo.io.vread(args.video.as_posix())[:,:,:,1]

    # print("stack shape: ", raw_speckle_img_seq.shape)
    # print("stack dtype: ", raw_speckle_img_seq.dtype)
    # print("stack range: ")
    # print(" min: ", raw_speckle_img_seq.min())
    # print(" max: ", raw_speckle_img_seq.max())
    t_lsci = spatial_contrast(raw_speckle_img_seq, args.kernel_size)
    # print(" K min: ", t_lsci.min())
    # print(" K max: ", t_lsci.max())
    # print(" K shape: ", t_lsci.shape)
    # print(" K dtype: ", t_lsci.dtype)
    # clip_value = 0.4*t_lsci.max()
    # t_lsci = np.clip(t_lsci, 0, clip_value)
    # t_lsci = (t_lsci*255/t_lsci.max()).astype(np.uint8)

    t_lsci = get_relative_flowmap(t_lsci)
    save_relative_flow_map(t_lsci, args.output_dir/'processed.avi', frame_rate, show=args.show)

    # # save processed video
    # print("Processing done. Now saving...")
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # if args.suffix:
    #     filename = og_name+f"_processed_{args.suffix}.avi"
    # else:
    #     filename = og_name+"_processed.avi"
    # video_writer = cv2.VideoWriter((args.output_dir/filename).as_posix(), fourcc, frame_rate, (width, height))

    # # print("t_lsci range: ")
    # # print(" min: ", t_lsci.min())
    # # print(" max: ", t_lsci.max())

    # for i in tqdm (range(t_lsci.shape[0]), desc="Saving video"):
    #     frame = t_lsci[i]

    #     # Convert grayscale to BGR by repeating channels
    #     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # Grayscale
    #     # frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE) # Colormap
    #     vidout=cv2.resize(frame,(width,height))
    #     # print(f"Writing frame {i + 1}: shape={frame.shape}, dtype={frame.dtype}")
    #     video_writer.write(vidout)

    #     if args.show:
    #         cv2.imshow('Frame', frame)

    #         # Wait for 1 ms and check if the user pressed the 'q' key to exit
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    
    # # Release the VideoWriter
    # video_writer.release()


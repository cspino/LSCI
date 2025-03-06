
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
from pathlib import Path

import time
import cv2
from matplotlib import pyplot as plt
import skvideo.io
# from skimage.exposure import equalize_adapthist
import numpy as np

from skimage.exposure import rescale_intensity
from tqdm import tqdm
from utils import temporal_contrast, read_folder_of_frames
from save_utils import save_relative_flow_map


if __name__=="__main__":
    parser = ArgumentParser(
    prog = 'temporal_contrast',
    description = 'Get temporal contrast from raw video',
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
    t_lsci, num_frames = temporal_contrast(raw_speckle_img_seq, args.window_size, baseline_contrast)
    # print(" K min: ", t_lsci.min())
    # print(" K max: ", t_lsci.max())
    # print(" K shape: ", t_lsci.shape)
    # print(" K dtype: ", t_lsci.dtype)

    # clip_value = 0.5*t_lsci.max()
    # t_lsci = np.clip(t_lsci, 0, clip_value)
    # t_lsci = (t_lsci*255/t_lsci.max()).astype(np.uint8)

    fig = plt.figure()
    # plt.hist(t_lsci.flatten(), bins=50)
    # plt.show()
    # fig.clear()

    # t_lsci = rescale_intensity(t_lsci)
    # plt.hist(t_lsci.flatten(), bins=50)
    # plt.show()
    # fig.clear()
    # t_lsci = (t_lsci*255).astype(np.uint8)
    print('k: ', t_lsci.min(), ' -- ', t_lsci.max())
    t_lsci = t_lsci**2
    print('k2: ', t_lsci.min(), ' -- ', t_lsci.max())
    # plt.hist(t_lsci.flatten(), bins=100)
    # plt.show()
    # fig.clear()
    
    t_lsci = 1/np.clip(t_lsci, 1e-5, None)
    print('1/k2: ', t_lsci.min(), ' -- ', t_lsci.max())
    # plt.hist(t_lsci.flatten(), bins=100)
    # plt.show()
    # fig.clear()

    # Here, we clip any value higher than the 95th percentile
    # to eliminate the extremely high values
    clip_to = np.percentile(t_lsci, 95)
    clip_to = np.max(t_lsci[t_lsci<=clip_to])
    t_lsci = np.clip(t_lsci, 0, clip_to)
    print('clipped: ', t_lsci.min(), ' -- ', t_lsci.max())
    # plt.hist(t_lsci.flatten(), bins=50)
    # plt.show()
    # fig.clear()
    
    # # this might be needed for visualization, but the scale needs to be based on the
    # # actual RFI values!!!
    # t_lsci = ((t_lsci/t_lsci.max())*255).astype(np.uint8)
    # print('normalized: ', t_lsci.min(), ' -- ', t_lsci.max())

    # # plt.hist(t_lsci.flatten(), bins=50)
    # # plt.show()
    # # fig.clear()

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

    # for i in tqdm (range(num_frames), desc="Saving video"):
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


"""
Real-time temporal contrast calculation and display with options to save raw and/or processed frames

"""
import sys
sys.path.append('..') # this is to be able to import files in project folder
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import numpy as np
import os
import cv2
import time
import tempfile
import h5py
import threading
import queue
from collections import deque
import skvideo

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE
from thorlabs_tsi_sdk.tl_camera_enums import DATA_RATE
from utils import calculate_contrast_from_sums, temporal_contrast, update_sums

raw_frame_queue = queue.Queue(maxsize=100)
k_frame_queue = queue.Queue(maxsize=100)
stop_flag = False

def writer_thread(output_file:Path, dimensions:tuple, frame_queue:queue.Queue):
    """
    Writes frames from frame_queue to given output_file path

    Parameters
    -----------
    output_file: Path
        Path to hdf5 file where frames will be saved
    dimensions: tuple
        Dimensions of each frame: (height, width)
    """
    with h5py.File(output_file.as_posix(), 'w') as h5file:
        dataset = h5file.create_dataset(
            "frames",
            shape=(0, dimensions[0], dimensions[1]),
            maxshape=(None, dimensions[0], dimensions[1]),
            dtype=np.float32
        )
        frame_count = 0
        while not stop_flag or not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=1)
                dataset.resize((frame_count+1, dimensions[0], dimensions[1]))
                dataset[frame_count] = frame
                frame_count +=1
                frame_queue.task_done()
            except queue.Empty:
                pass


if __name__=="__main__":
    parser = ArgumentParser(
    prog = 'process_image',
    description = 'Get temporal contrast image from thorcam video feed',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', '--window_size',
                        default = 10,
                        type = int,
                        help = 'Temporal window size. 10 by default')

    parser.add_argument('-b', '--baseline',
                        default=False,
                        type = Path,
                        help = 'If provided, contrast will be relative to baseline')
    parser.add_argument('-o', '--output_path',
                        default = False,
                        type = Path,
                        help = 'If provided, video will be saved to specified path')
    parser.add_argument('-r', '--output_path_raw',
                        default = False,
                        type = Path,
                        help = 'If provided, raw video will be saved to specified path')

    args = parser.parse_args()

    # Calculate baseline contrast
    baseline_contrast = None
    if args.baseline:
        print("Calculating baseline contrast")
        baseline_img = skvideo.io.vread(args.baseline.as_posix())[:,:,:,1]
        baseline_contrast, _ = temporal_contrast(baseline_img, args.window_size)
        baseline_contrast = np.mean(baseline_contrast, axis=0)

    # Start reading camera
    try:
        # if on Windows, use the provided setup script to add the DLLs folder to the PATH
        from windows_setup import configure_path
        configure_path()
    except ImportError:
        configure_path = None

    with TLCameraSDK() as sdk:
        available_cameras = sdk.discover_available_cameras()
        if len(available_cameras) < 1:
            print("no cameras detected")
        else:
            stack = deque(maxlen=args.window_size)
            sum_s = 0
            sum_s2 = 0
            
            with sdk.open_camera(available_cameras[0]) as camera:
                # Get camera ready for acquisition
                # print('data rate: ', camera.data_rate) # Currently set to 30 fps  (other option is 50 fps)
                camera.exposure_time_us = 10000  # set exposure to 10 ms #TODO this should eventually be a parameter
                camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
                camera.image_poll_timeout_ms = 1000  # 1 second polling timeout
                # print('exposure time: ', camera.exposure_time_us)
                dimensions = (camera.image_height_pixels, camera.image_width_pixels)

                camera.arm(2)
                camera.issue_software_trigger()

                with tempfile.TemporaryDirectory() as tmp:
                    hdf5_raw_path = Path(tmp)/ 'tmp_raw_frames.h5'
                    hdf5_k_path = Path(tmp)/ 'tmp_k_frames.h5'

                    raw_write_thread = threading.Thread(target=writer_thread, args=(hdf5_raw_path, dimensions, raw_frame_queue))
                    k_write_thread = threading.Thread(target=writer_thread, args=(hdf5_k_path, dimensions, k_frame_queue))

                    raw_write_thread.start()
                    k_write_thread.start()

                    try:
                        while True:
                            start_time = time.time()
                            frame = camera.get_pending_frame_or_null()
                            print('frame number: ', frame.frame_count)
                            if frame is not None:
                                stack.append(np.copy(frame.image_buffer).astype(np.float32))

                                # print('checkpoint 1: ', time.time()-start_time)
                                
                                if len(stack) < args.window_size:
                                    # only start processing once we have as many frames as the window size
                                    # but start calculating the sum
                                    new_frame = stack[-1]
                                    sum_s += new_frame
                                    sum_s2 += new_frame**2
                                    old_frame = stack[0]
                                    continue

                                # Our stack should always be as long as the window size
                                assert len(stack) == args.window_size

                                # PROCESS IMAGE
                                processing_start = time.time()

                                new_frame = stack[-1]
                                sum_s, sum_s2 = update_sums(sum_s, sum_s2, new_frame, old_frame)

                                contrast_start = time.time()
                                contrast_frame = calculate_contrast_from_sums(sum_s, sum_s2, args.window_size)
                                contrast_frame = contrast_frame/2 # TODO this shouldn't be here... see what to do about range

                                # print('checkpoint 1.5: ', time.time()-contrast_start)
                                #print("min: {}, max: {}".format(contrast_frame.min(), contrast_frame.max()))

                                display_prep_start = time.time()
                                frame_to_display = cv2.cvtColor(contrast_frame, cv2.COLOR_GRAY2BGR) # Repeats image over 3 channels

                                # frame_to_display = cv2.applyColorMap(contrast_frame, cv2.COLORMAP_BONE) # Colormap

                                if not baseline_contrast is None:
                                    # subtract baseline if provided
                                    frame_to_display = frame_to_display - baseline_contrast

                                # frame_to_display = (frame_to_display*255).astype(np.uint8)
                                # print('display prep time: ', time.time()-display_prep_start)

                                # print('checkpoint 2: ', time.time()-processing_start)
                                show_start = time.time()

                                #print("min: {}, max: {}".format(frame_to_display.min(), frame_to_display.max()))
                                raw_frame = stack[int(args.window_size/2)]
                                raw_frame = raw_frame * (255/np.max(raw_frame))
                                raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR).astype(np.uint8)
                                # print('raw frame min: ', np.min(raw_frame))
                                # print('raw frame max: ', np.max(raw_frame))
                                # print('raw frame shape: ', raw_frame.shape)

                                cv2.imshow("Image From TSI Cam", raw_frame)
                                cv2.waitKey(1) # Continuous feed
                                # print('checkpoint 3: ', time.time()-show_start)

                                # Save all processed frames to numpy array if we want to save as video
                                # if args.output_path:
                                #     continue
                                save_start = time.time()
                                if args.output_path_raw:
                                    raw_frame = stack[int(args.window_size/2)]
                                    try:
                                        raw_frame_queue.put_nowait(raw_frame)
                                    except queue.Full:
                                        print("Frame dropped from save due to full queue")
                                if args.output_path:
                                    try:
                                        k_frame_queue.put_nowait(contrast_frame)
                                    except queue.Full:
                                        print("Frame dropped from save due to full queue")
                                # print('checkpoint 4: ', time.time()-save_start)

                                old_frame = stack[0] # defined for next loop
                                # print("time elapsed: ", time.time()-start_time)

                            else:
                                print("Unable to acquire image, program exiting...")
                                exit()

                    except KeyboardInterrupt:
                        print("loop terminated. Waiting for saving thread to finish")
                        stop_flag=True
                        raw_write_thread.join()
                        k_write_thread.join()


                    if args.output_path:
                    ## Save video
                        print("Saving processed video")
                        os.makedirs(args.output_path.parent, exist_ok=True)

                        with h5py.File(hdf5_k_path, 'r') as h5file:
                            full_sequence_k = h5file['frames'][:]

                        # VideoWriter setup
                        fps = 30  # frames per second #TODO change this
                        frame_size = (camera._local_image_width_pixels, camera.image_height_pixels)
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
                        out = cv2.VideoWriter(args.output_path.as_posix(), fourcc, fps, frame_size)
                        
                        full_sequence_k = full_sequence_k * (255/np.max(full_sequence_k)) # TODO unsure if this is the best method
                        for i in range(full_sequence_k.shape[0]):
                            
                            frame = full_sequence_k[i].astype(np.uint8)
                            # print('after range: ', np.min(frame), ' -- ', np.max(frame))
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            out.write(frame)

                        out.release()

                    if args.output_path_raw:
                    ## Save video
                        print("Saving raw frames")
                        os.makedirs(args.output_path_raw, exist_ok=True)

                        with h5py.File(hdf5_raw_path, 'r') as h5file:
                            full_sequence_raw = h5file['frames'][:]
                        
                        print('sequence shape: ', full_sequence_raw.shape)
                        print('dtype: ', full_sequence_raw.dtype)
                        print('min: ', np.min(full_sequence_raw))
                        print('max: ', np.max(full_sequence_raw))
                        full_sequence_raw=full_sequence_raw/full_sequence_raw.max()*255 #TODO this isn't ideal

                        fps = 30  # frames per second #TODO change this
                        frame_size = (camera._local_image_width_pixels, camera.image_height_pixels)
                        # print('frame_size: ', frame_size)
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
                        out = cv2.VideoWriter((args.output_path_raw/f'raw.avi').as_posix(), fourcc, fps, frame_size) #TODO change this filename

                        # print('raw range: ', np.min(full_sequence_raw), ' -- ', np.max(full_sequence_raw))
                        for i in range(full_sequence_raw.shape[0]):
                            frame = full_sequence_raw[i]
                            # Save as png
                            cv2.imwrite((args.output_path_raw/f'frame{i:04d}.png').as_posix(), frame)
                            # Save as avi
                            frame = frame.astype(np.uint8)
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            out.write(frame)
                        out.release()

                cv2.destroyAllWindows()
                camera.disarm()

    print("program completed")

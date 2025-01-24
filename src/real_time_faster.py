"""
Polling Example

This example shows how to open a camera, adjust some settings, and poll for images. It also shows how 'with' statements
can be used to automatically clean up camera and SDK resources.

"""
import sys
sys.path.append('..')
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

import skvideo
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE

from utils import calculate_contrast_from_sums, temporal_contrast

frame_queue = queue.Queue(maxsize=100)
stop_flag = False

def writer_thread(output_file:Path, dimensions:tuple):
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
    description = 'Get temporal contrast image from raw video',
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
        print("Calling config path")
        configure_path()
    except ImportError:
        print("ERROR")
        configure_path = None


    with TLCameraSDK() as sdk:
        available_cameras = sdk.discover_available_cameras()
        if len(available_cameras) < 1:
            print("no cameras detected")

        with sdk.open_camera(available_cameras[0]) as camera:
            camera.exposure_time_us = 10000  # set exposure to 10 ms
            camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
            camera.image_poll_timeout_ms = 1000  # 1 second polling timeout
            #camera.frame_rate_control_value = 10
            #camera.is_frame_rate_control_enabled = True

            camera.arm(2)
            camera.issue_software_trigger()

            stack = None
            i=0
            sum_s = 0
            sum_s2 = 0
            with tempfile.TemporaryDirectory() as tmp:
                hdf5_raw_path = Path(tmp)/ 'tmp_raw_frames.h5'
                dimensions = (camera.image_height_pixels, camera.image_width_pixels)
                thread = threading.Thread(target=writer_thread, args=(hdf5_raw_path, dimensions))
                thread.start()
                # with h5py.File(hdf5_raw_path, 'w') as h5_raw_file:
                #     raw_dataset = h5_raw_file.create_dataset( #TODO we could create another dataset for processed frames
                #         'raw_frames',
                #         shape=(0, camera.image_height_pixels, camera.image_width_pixels),
                #         maxshape=(None, camera.image_height_pixels, camera.image_width_pixels),
                #         dtype=np.float32)

                try:
                    frame_count=0
                    while True:
                        start_time = time.time()
                        frame = camera.get_pending_frame_or_null()
                        if frame is not None:
                            #print("frame #{} received!".format(frame.frame_count))
                            image_buffer_copy = np.copy(frame.image_buffer).astype(np.float32) #TODO see if astype is necessary
                            numpy_shaped_image = image_buffer_copy.reshape(camera.image_height_pixels, camera.image_width_pixels)

                            if not stack is None:
                                stack = np.concatenate((stack, numpy_shaped_image[np.newaxis,:,:]), axis=0) # changed this so that the frames are stacked along axis 0
                            else:
                                stack = numpy_shaped_image[np.newaxis,:,:]
                            
                            if stack.shape[0] < args.window_size:
                                # only start processing once we have as many frames as the window size
                                # but start calculating the sum
                                new_frame = stack[-1]
                                sum_s += new_frame
                                sum_s2 += new_frame**2
                                old_frame = stack[0]
                                continue

                            # Our stack should always be as long as the window size
                            assert stack.shape[0] == args.window_size

                            # PROCESS IMAGE
                            # start_time = time.time()

                            new_frame = stack[-1]
                            sum_s += new_frame - old_frame # pylint: disable=used-before-assignment
                            sum_s2 += new_frame**2 - old_frame**2

                            contrast_frame = calculate_contrast_from_sums(sum_s, sum_s2, args.window_size)
                            contrast_frame = contrast_frame/2
                            #print("min: {}, max: {}".format(contrast_frame.min(), contrast_frame.max()))

                            frame_to_display = cv2.cvtColor(contrast_frame, cv2.COLOR_GRAY2BGR) # Grayscale
                            #frame_to_display = cv2.applyColorMap(contrast_frame, cv2.COLORMAP_BONE) # Colormap

                            if not baseline_contrast is None:
                                # subtract baseline if provided
                                frame_to_display = frame_to_display - baseline_contrast

                            frame_to_display = (frame_to_display*255).astype(np.uint8)
                            #print("min: {}, max: {}".format(frame_to_display.min(), frame_to_display.max()))
                            cv2.imshow("Image From TSI Cam", new_frame/np.max(new_frame))
                            cv2.waitKey(1)

                            # Save all processed frames to numpy array if we want to save as video
                            # if args.output_path:
                            #     continue
                            if args.output_path_raw:
                                raw_frame = stack[int(args.window_size/2)]
                                try:
                                    frame_queue.put_nowait(raw_frame)
                                except queue.Full:
                                    print("Frame dropped from save due to full queue")

                            stack = np.delete(stack, 0, axis=0) # remove the first frame from stack (we're only keeping the current window)
                            i += 1
                            old_frame = stack[0] # defined for next loop
                            print("time elapsed: ", time.time()-start_time)

                        else:
                            print("Unable to acquire image, program exiting...")
                            exit()

                except KeyboardInterrupt:
                    print('path exists: ', hdf5_raw_path.exists())
                    print("loop terminated. Waiting for saving thread to finish")
                    stop_flag=True
                    thread.join()
                    print('path exists: ', hdf5_raw_path.exists())


                # if args.output_path:
                # ## Save video
                #     print("Saving processed video")
                #     print(full_sequence.dtype)
                #     print(full_sequence.shape)
                #     os.makedirs(args.output_path.parent, exist_ok=True)

                #     # VideoWriter setup
                #     fps = 10  # frames per second
                #     frame_size = (1920, 1080)
                #     fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
                #     out = cv2.VideoWriter(args.output_path, fourcc, fps, frame_size)

                #     for i in range(full_sequence.shape[0]):
                #         out.write(full_sequence[i])

                #     out.release()

                if args.output_path_raw:
                ## Save video
                    print("Saving raw frames")
                    os.makedirs(args.output_path_raw, exist_ok=True)
                    print('path exists: ', hdf5_raw_path.exists())

                    with h5py.File(hdf5_raw_path, 'r') as h5file:
                        full_sequence_raw = h5file['frames'][:]

                    print('sequence shape: ', full_sequence_raw.shape)
                    print('min: ', np.min(full_sequence_raw))
                    print('max: ', np.max(full_sequence_raw))
                    full_sequence_raw=full_sequence_raw/full_sequence_raw.max()*255

                    for i in range(full_sequence_raw.shape[0]):
                        frame = full_sequence_raw[i]
                        cv2.imwrite((args.output_path_raw/f'frame{i:04d}.png').as_posix(), frame)

            cv2.destroyAllWindows()
            camera.disarm()

    print("program completed")

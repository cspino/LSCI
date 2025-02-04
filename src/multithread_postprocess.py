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
import subprocess

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE
from thorlabs_tsi_sdk.tl_camera_enums import DATA_RATE
from utils import calculate_contrast_from_sums, temporal_contrast, update_sums

def save_frames_hdf5(output_file:Path, dimensions:tuple, frame_queue:queue.Queue, stop:threading.Event):
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
        while not stop.is_set() or not frame_queue.empty():
            start_time = time.time()
            try:
                frame = frame_queue.get(timeout=1)
                dataset.resize((frame_count+1, dimensions[0], dimensions[1]))
                dataset[frame_count] = frame
                frame_count +=1
                frame_queue.task_done()
            except queue.Empty:
                pass
            # print('Saving time: ', time.time()-start_time)
        print('Save thread stopping...')

def save_frames(output_file:Path, dimensions:tuple, frame_queue:queue.Queue, stop:threading.Event):
    with open(output_file, 'wb') as f:
        while not stop.is_set() or not frame_queue.empty():
            start_time = time.time()
            try:
                frame = frame_queue.get(timeout=1)
                f.write(frame.tobytes())
                frame_queue.task_done()
            except queue.Empty:
                pass
            # print('Saving time: ', time.time()-start_time)
        print('Save thread stopping...')

def capture_frames(queue_raw_save:queue.Queue, queue_raw_display:deque, stop:threading.Event):
    """Process to acquire frames from the camera"""
    # Start reading camera
    try:
        # if on Windows, use the provided setup script to add the DLLs folder to the PATH
        from windows_setup import configure_path
        configure_path()
    except ImportError:
        print('IMPORT ERROR')
        configure_path = None

    with TLCameraSDK() as sdk:
        available_cameras = sdk.discover_available_cameras()
        if len(available_cameras) < 1:
            print("No cameras detected.")
            stop.set()
            return

        with sdk.open_camera(available_cameras[0]) as camera:
            camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
            camera.exposure_time_us = 10000  # set exposure to 10 ms #TODO this should eventually be a parameter
            camera.image_poll_timeout_ms = 1000  # 1 second polling timeout

            camera.arm(2)
            camera.issue_software_trigger()

            while not stop.is_set():
                frame = camera.get_pending_frame_or_null()
                if frame is not None:
                    # print('frame number: ', frame.frame_count)
                    frame = np.copy(frame.image_buffer).astype(np.float32)
                    # Send raw frame to queue
                    try:
                        queue_raw_save.put_nowait(frame)  # Send latest frame
                        queue_raw_display.append(frame)  # Send latest frame
                    except queue.Full:
                        print('Raw frame queue is full, frame skipped')
                        pass
            print('Capture thread stopping...')

def display_frames(frame_queue:deque, stop:threading.Event):
    while not stop.is_set():
        try:
            frame = frame_queue.popleft()
            frame = frame * (255/np.max(frame)) #TODO not ideal
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            cv2.imshow("Image From TSI Cam", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                stop.set() # Stop when q is pressed
            frame_queue.task_done()
        except:
            pass
    cv2.destroyAllWindows()
    print('Display thread stopping...')


if __name__=="__main__":
    parser = ArgumentParser(
    prog = 'process_image',
    description = 'Get temporal contrast image from thorcam video feed',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', '--window_size',
                        default = 10,
                        type = int,
                        help = 'Temporal window size. 10 by default')
    parser.add_argument('-o', '--output_dir',
                        default = None,
                        type = Path,
                        help = 'Directory where raw and processed videos will be saved.')
    parser.add_argument('-nr', '--no-raw',
                        dest = "raw",
                        action = "store_false",
                        help = 'Use this tag to disable saving of the raw video.')
    parser.add_argument('-nk', '--no-k',
                        dest = "process",
                        action = "store_false",
                        help = 'Use this tag to disable processing into speckle contrast.')


    args = parser.parse_args()

    raw_frame_queue_saving = queue.Queue(maxsize=100)
    raw_frame_queue_display = deque(maxlen=1) # We only need to display the latest frame
    stop_event = threading.Event()

    with tempfile.TemporaryDirectory() as tmp:
        binary_path = Path(tmp)/ 'tmp_raw_frames.raw'

        capture_thread = threading.Thread(target=capture_frames, args=(raw_frame_queue_saving, raw_frame_queue_display, stop_event))
        save_thread_raw = threading.Thread(target=save_frames, args=(binary_path, (1080,1920), raw_frame_queue_saving, stop_event))
        display_thread = threading.Thread(target=display_frames, args=(raw_frame_queue_display, stop_event))

        capture_thread.start()
        save_thread_raw.start()
        display_thread.start()

        try:
            while not stop_event.is_set():
                time.sleep(1)
                # We wait for the stop signal

        except KeyboardInterrupt:
            print("Stopping program...")
            stop_event.set()
        
        # Wait for all threads to finish
        save_thread_raw.join()
        display_thread.join()
        capture_thread.join()

        if args.raw:
            # Read the whole sequence from the binary file and save as pngs as well as avi
            data = np.fromfile(binary_path, dtype=np.float32)
            num_frames = data.size // (1080*1920)
            full_sequence_raw = data.reshape((num_frames, 1080, 1920))

            # Save video
            print("Saving raw frames")
            os.makedirs(args.output_dir/'raw_frames', exist_ok=True)

            # print('sequence shape: ', full_sequence_raw.shape)
            # print('dtype: ', full_sequence_raw.dtype)
            # print('min: ', np.min(full_sequence_raw))
            # print('max: ', np.max(full_sequence_raw))
            full_sequence_raw=(full_sequence_raw/full_sequence_raw.max())*255 #TODO this isn't ideal

            fps = 30  # frames per second #TODO change this
            frame_size = (1920, 1080) # CAREFUL! THIS NEEDS TO  BE INVERTED!!
            # print('frame_size: ', frame_size)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
            out = cv2.VideoWriter((args.output_dir/f'raw.avi').as_posix(), fourcc, fps, frame_size, False)

            # full_sequence_raw = cv2.cvtColor(full_sequence_raw, cv2.COLOR_GRAY2BGR)
            print('raw range when saving: ', np.min(full_sequence_raw), ' -- ', np.max(full_sequence_raw))
            # print('dtype ----- ', full_sequence_raw.dtype)

            for i in range(full_sequence_raw.shape[0]):
                frame = full_sequence_raw[i]
                # Save as png
                cv2.imwrite((args.output_dir/'raw_frames'/f'frame{i:04d}.png').as_posix(), frame)
                # print('dtype ----- ', frame.dtype)
                # Save as avi
                frame = frame.astype(np.uint8)
                out.write(frame)
            out.release()

        if args.process:
            command = ['python', 'temporal_contrast.py',
                        '-v', (args.output_dir/'raw_frames').as_posix(),
                        # '-v', binary_path.as_posix(),
                        '-o', args.output_dir.as_posix(),
                        '-w', str(args.window_size),
                        '--show']
            subprocess.run(command, shell=True)

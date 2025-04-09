import sys
sys.path.append('..') # this is to be able to import files in project folder

from multiprocessing import shared_memory
import time
import numpy as np
import pyqtgraph as pg
import cv2
import queue
from pathlib import Path
import threading
import multiprocessing

from utils.utils import spatial_bfi_frame

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QMenu

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

LIVE_DTYPE = np.float32

class InteractiveColorBar(pg.ColorBarItem):
        def __init__(self, main_window, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.main_window = main_window

        def mousePressEvent(self, event):
            """
            Adds a menu window with window level options
            """
            if event.button() == Qt.RightButton:
                menu = QMenu()
                set_window = menu.addAction("Auto set the window level")
                auto_window = menu.addAction("Toggle auto adjust")
                action = menu.exec_(event.screenPos())

                if action == set_window:
                    clip_to = np.percentile(self.main_window.data, 95)
                    self.main_window.colorbar.setLevels((0, clip_to))
                    self.main_window.auto_window_level = False
                elif action == auto_window:
                    self.main_window.auto_window_level = not self.main_window.auto_window_level

            else:
                super().mousePressEvent(event)


class ProcessMonitor(QThread):
    finished_signal = pyqtSignal(int)

    def __init__(self, process_id, process):
        super().__init__()
        self.process_id = process_id
        self.process = process

    def run(self):
        self.process.join()
        self.finished_signal.emit(self.process_id)


class DisplayProcessed(QThread):
    ImageUpdate = pyqtSignal(np.ndarray)
    FirstFrame = pyqtSignal(str)

    def __init__(self, signal_queue:multiprocessing.Queue):
        super().__init__()
        self.signal_queue = signal_queue
        self.ThreadActive = False
        self.dtype = LIVE_DTYPE

        self.processed_shm = shared_memory.SharedMemory(name='processed')
        self.processed_frame = np.ndarray((1920,1080), dtype=self.dtype, buffer=self.processed_shm.buf)

    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            # Wait for new frame signal and update frame
            signal = self.signal_queue.get()
            if signal:
                # If signal is True, we're receiving the first frame
                self.FirstFrame.emit('magma')
            
            self.ImageUpdate.emit(self.processed_frame)
        self.processed_shm.close()

    def stop(self):
        self.ThreadActive = False


class CameraWorker(QThread):
    ImageUpdate = pyqtSignal(np.ndarray)
    FirstFrame = pyqtSignal(str)
    CameraError = pyqtSignal()

    def __init__(self, frame_queue, recording:threading.Event, display_raw=True):
        super().__init__()
        self.frame_queue = frame_queue
        self.ThreadActive = False
        self.recording = recording
        self.display_raw = display_raw
        self.max_attempts = 4

        if not display_raw:
            self.raw_shm = shared_memory.SharedMemory(create=False, name='spatial')
            self.raw_frame = np.ndarray((1920,1080), dtype=LIVE_DTYPE, buffer=self.raw_shm.buf)

    def run(self):
        self.ThreadActive = True

        # Start reading camera
        try:
            # add the DLLs folder to the PATH
            from config.windows_setup import configure_path
            configure_path()
        except ImportError:
            print('IMPORT ERROR')
            configure_path = None

        for _ in range(self.max_attempts):
            try:
                with TLCameraSDK() as sdk:
                    available_cameras = sdk.discover_available_cameras()
                    if len(available_cameras) < 1:
                        print("No cameras detected.")
                        self.stop()

                    with sdk.open_camera(available_cameras[0]) as camera:
                        camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
                        camera.exposure_time_us = 10000  # set exposure to 10 ms
                        camera.image_poll_timeout_ms = 1000  # 1 second polling timeout

                        camera.arm(2)
                        camera.issue_software_trigger()

                        first_frame = True
                        while self.ThreadActive:
                            frame = camera.get_pending_frame_or_null()
                            if frame is not None:
                                frame_float32 = np.copy(frame.image_buffer).astype(np.float32)
                                if self.recording.is_set():
                                    # Currently recording
                                    self.frame_queue.put(frame_float32)
                                frame_float32 = cv2.transpose(frame_float32)
                                frame_float32 = cv2.flip(frame_float32, -1)

                                frame = ((frame_float32-frame_float32.min())/(frame_float32.max()-frame_float32.min()) * 255).astype(np.uint8)
                                
                                if self.display_raw:
                                    self.ImageUpdate.emit(frame)
                                    if first_frame:
                                        self.FirstFrame.emit('gray')
                                else:
                                    # Live processing is activated
                                    self.raw_frame[:] = frame_float32[:]

                                first_frame = False
                        if not self.display_raw:
                            self.raw_shm.close()
                    break
            except:
                time.sleep(1) # Wait 1 second before trying again

        else:
            print('Failed to access camera after multiple attempts.')
            self.CameraError.emit()

    
    def stop(self):
        self.ThreadActive = False
        # if not self.display_raw:
        #     self.raw_shm.close()
        # self.quit()


class SaveThread(QThread):
    finished = pyqtSignal(Path)  # Define a signal that will be emitted when saving is finished

    def __init__(self, output_file: Path, frame_queue: queue.Queue):
        super().__init__()
        self.output_file = output_file
        self.frame_queue = frame_queue

    def run(self):
        with open(self.output_file, 'wb') as f:
            while not (self.frame_queue.empty() and self.isInterruptionRequested()):
                try:
                    frame = self.frame_queue.get(timeout=1)
                    f.write(frame.tobytes())
                    self.frame_queue.task_done()
                except queue.Empty:
                    pass

        self.finished.emit(self.output_file)  # Emit the signal when saving is finished


class SaveAsVideo(QThread):
    def __init__(self, raw_path: Path):
        super().__init__()
        self.raw_path = raw_path
        self.bytes_per_frame = 1920*1080*np.dtype(np.float32).itemsize
        self.batch_size = 300
        self.fps = 30

        self.norm_max = None

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
        out = cv2.VideoWriter((self.raw_path.with_suffix('.avi')).as_posix(), fourcc, self.fps, (1920, 1080), False)

        with open(self.raw_path, 'rb') as f:
            while True:
                buffer = f.read(self.bytes_per_frame*self.batch_size)
                if not buffer:
                    break

                video = np.frombuffer(buffer, dtype=np.float32)
                video = video.reshape((-1, 1080, 1920))

                if not self.norm_max:
                    self.norm_max = video.max()
                
                video = (video/self.norm_max)*255

                for i in range(video.shape[0]):
                    frame = video[i]
                    frame = frame.astype(np.uint8)
                    out.write(frame)

            out.release()


def spatial_worker(kernel_size:int, out_queue:multiprocessing.Queue, stop_event:multiprocessing.Event,
                    raw_mem:str="spatial", processed_mem:str="processed"):
    """
    Used for live processing

    Parameters
    -------------
    kernel_size: int
        Spatial window size
    out_queue: multiprocessing.Queue
        Queue used to send a signal anytime a new frame was processed
    stop_event: multiprocessing.Event
        Stop signal
    raw_mem: str
        Name of the shared memory where raw frames are stored
    processed_mem: str
        Name of the shared memory where processed frames are stored
    """

    raw_shm = shared_memory.SharedMemory(create=False, name=raw_mem)
    raw_frame = np.ndarray((1920,1080), dtype=LIVE_DTYPE, buffer=raw_shm.buf)

    processed_shm = shared_memory.SharedMemory(create=False, name=processed_mem)
    processed_frame = np.ndarray((1920,1080), dtype=LIVE_DTYPE, buffer=processed_shm.buf)
    
    first_frame = True
    while not stop_event.is_set():

        working_copy = raw_frame.copy()
        processed_frame[:] = spatial_bfi_frame(working_copy, kernel_size)[:]
        
        out_queue.put(first_frame) # Send signal through the queue
        first_frame=False

    raw_shm.close()
    processed_shm.close()

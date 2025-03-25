from multiprocessing import Process, shared_memory
import os
import subprocess
import skvideo
from PyQt5.QtWidgets import *
import sys

from utils import spatial_bfi_frame, spatial_one_frame
# from utils import launch_processing
sys.path.append('..') # this is to be able to import files in project folder
# from windows_setup import configure_path
# configure_path()
import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore
# from pyqtgraph import ptime

import cv2
import toml
import queue
from pathlib import Path
import threading
from app_ui import Ui_MainWindow
import multiprocessing
import process_videos

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE
from thorlabs_tsi_sdk.tl_camera_enums import DATA_RATE

LIVE_DTYPE = np.float32

pg.setConfigOptions(useOpenGL=True, enableExperimental=True) # Reduces lag in displayed stream

class MyApp(QApplication):
    def __init__(self, *args):
        super().__init__(*args)
        self.aboutToQuit.connect(self.cleanup)
        self.main_window = None

    def set_main_window(self, window):
        self.main_window = window

    def cleanup(self):
        if self.main_window.live_processing:
            self.main_window.stop_live_processing.set()
            print('cleanup done')


# Image View class
class ImageView(pg.ImageView):
    # constructor which inherit original ImageView
    def __init__(self, *args, **kwargs):
        pg.ImageView.__init__(self, *args, **kwargs)

class Window(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()

        self.config = toml.load('../config/default_config.toml')
        self.output_dir = Path(self.config['output_dir'])

        self.setWindowTitle("PyQtGraph Video Display")
        self.setGeometry(100, 100, 600, 500)

        icon = QIcon("skin.png")  # Replace this if you don't have an icon
        self.setWindowIcon(icon)
        self.setupUi(self)
        self.UiComponents()
        self.show()

        self.frame_queue = queue.Queue()
        self.save_thread = None
        self.save_as_vid_threads = []

        self.recording = threading.Event()

        # To keep track of processing launched (post)
        self.processes = {} # Stores processes
        self.monitor_threads = {} # Stores QThreads

        # Live processing
        self.live_processing = False
        self.signal_queue = multiprocessing.Queue()
        self.DisplayProcessed = None # This stores the process responsible for real time processing

    # method for components
    def UiComponents(self):
        # widget = QWidget()
        # TODO this can be moved to the UI document if we pass the config
        _translate = QtCore.QCoreApplication.translate
        self.output_dir_line_edit.setText(_translate("MainWindow", self.config['output_dir']))


        # self.StartButton = QPushButton("Start Acquisition")
        self.StartButton.clicked.connect(self.ToggleAcquisition)

        # Output directory selection
        # self.output_dir_line_edit = QLineEdit(self)
        # self.output_dir_line_edit.setPlaceholderText("Choose output directory")

        # self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_directory)

        self.RecordButton.clicked.connect(self.ToggleRecord)

        self.SpatialBox.toggled.connect(self.check_bfi_state)
        self.TemporalBox.toggled.connect(self.check_bfi_state)

        # Synchronize both spin boxes for spatial window size
        self.spatial_window1.valueChanged.connect(self.sync_spatial_window_boxes)
        self.spatial_window2.valueChanged.connect(self.sync_spatial_window_boxes)

        self.process_table.setColumnWidth(0,25)
        self.process_table.setColumnWidth(1,150)

        # Radio buttons for display
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.radio_raw)
        self.button_group.addButton(self.radio_temporal)
        self.button_group.addButton(self.radio_spatial)


        pg.setConfigOptions(antialias=True)
        # self.win = pg.GraphicsLayoutWidget()
        self.win.ci.setContentsMargins(0, 0, 0, 0)
        # self.p1 = self.win.addPlot()
        # self.img = pg.ImageItem()
        # self.p1.addItem(self.img)
        # self.p1.setMouseEnabled(x=False,y=False)
        # self.p1.setAspectLocked(True)

        # self.p1.hideAxis('left')
        # self.p1.hideAxis('bottom')
        # self.p1.hideButtons()

        # self.p1.scale(1.0, 1.0)
        # self.p1.setLimits(xMin=0, xMax=1920, yMin=0, yMax=1080)


        self.view = self.win.addViewBox()
        # view.setBackgroundColor('w')
        self.view.setMouseEnabled(x=False, y=False)
        self.view.setAspectLocked(True)
        self.view.enableAutoRange()
        self.view.setRange(QRectF(0, 0, 1920, 1080))  #TODO store this in config file

        self.img = pg.ImageItem()
        # self.img.setPos(-frac*1920, -frac*1080)
        self.view.addItem(self.img)


    def sync_spatial_window_boxes(self):
        sender = self.sender()  # Get the widget that triggered the signal

        if sender == self.spatial_window1:
            self.spatial_window2.setValue(self.spatial_window1.value())
        elif sender == self.spatial_window2:
            self.spatial_window1.setValue(self.spatial_window2.value())

    def check_bfi_state(self):
        if self.SpatialBox.isChecked() or self.TemporalBox.isChecked():
            self.BFI_CheckBox.setEnabled(True)
        else:
            self.BFI_CheckBox.setEnabled(False)

    def browse_directory(self):
        """Open a directory dialog to select an output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir.as_posix())
        if dir_path:
            self.output_dir = Path(dir_path)  # Set the selected directory
            self.output_dir_line_edit.setText(str(self.output_dir))

    def ToggleAcquisition(self):
        if self.StartButton.isChecked():
            # Start the acquisition
            print("Starting acquisition...")
            self.StartButton.setText("Stop")

            # Disable radio buttons
            for button in self.button_group.buttons():
                button.setEnabled(False)

            if not self.radio_raw.isChecked():
                n_bytes = np.empty((1920,1080), dtype=LIVE_DTYPE).nbytes
                self.spatial_mem = shared_memory.SharedMemory(create=True, name='spatial', size=n_bytes)
                self.processed_mem = shared_memory.SharedMemory(create=True, name='processed', size=n_bytes)
                print('mem files created')
            
            # Start worker to handle video feed
            self.CameraWorker = CameraWorker(self.frame_queue, self.recording, self.radio_raw.isChecked())
            self.CameraWorker.start()

            if self.radio_raw.isChecked():
                self.CameraWorker.ImageUpdate.connect(self.ImageUpdateSlot)

            elif self.radio_spatial.isChecked():
                self.live_processing = True
                self.stop_live_processing = multiprocessing.Event()
                # print('kwargs: ', {'kernel_size':self.temp_window.value(), 'out_queue': self.signal_queue, "stop_event": self.stop_live_processing})
                self.live_process = multiprocessing.Process(target=spatial_worker, args=(self.temp_window.value(),
                                                                                          self.signal_queue,
                                                                                          self.stop_live_processing))
                # self.live_process = SpatialWorker(self.temp_window.value(), self.signal_queue, self.stop_live_processing)
                self.live_process.start()

                self.DisplayProcessed = DisplayProcessed(self.signal_queue)
                self.DisplayProcessed.start()
                self.DisplayProcessed.ImageUpdate.connect(lambda image: self.ImageUpdateSlot(image, processed=True))

            self.RecordButton.setEnabled(True)

        else:
            # Stop the acquisition
            print("Stopping acquisition...")
            self.StartButton.setText("Start")
            self.RecordButton.setEnabled(False)

            # Enable radio buttons
            for button in self.button_group.buttons():
                button.setEnabled(True)
            
            if self.CameraWorker:
                self.CameraWorker.stop()  # Stop the video feed worker

            if self.DisplayProcessed:
                self.live_processing = False
                self.DisplayProcessed.stop()
                self.stop_live_processing.set()
                self.processed_mem.close()
                self.spatial_mem.close()
                self.DisplayProcessed = None
                print('2')

    def ToggleRecord(self):
        if  self.RecordButton.isChecked():
            self.RecordButton.setText("Stop Record")
            # self.RecordButton.setChecked(True)
            self.StartButton.setEnabled(False)

            self.recording.set()
            # Start the save worker to save frames
            self.save_thread = SaveThread(self.output_dir/'raw_frames.raw', self.frame_queue)
            self.save_thread.finished.connect(self.save_raw_as_vid)  # Connect the signal for when save finishes
            self.save_thread.start()

        else:
            # Stop saving
            self.RecordButton.setText("Start Record")
            # self.RecordButton.setChecked(False)
            
            self.save_thread.requestInterruption()
            self.recording.clear()
            self.StartButton.setEnabled(True)


    def ImageUpdateSlot(self, image, processed=False):
        # If we're showing processed frames (processed=True), 
        # then we want to make sure that self.live_processing hasnt been 
        # set to false to avoid acessing the shared memory that has 
        # been discarded
        if processed:
            if self.live_processing:
                self.img.setImage(image)
                # self.img.setRect(QRectF(0, 0, 1920, 1080))
                self.view.autoRange()
        else:
            self.img.setImage(image)
            self.view.autoRange()

    def CancelFeed(self):
        self.CameraWorker.stop()

    def save_raw_as_vid(self, raw_path):
        print("Saving finished, starting another thread...")
        # Save .raw vid as .avi
        thread = SaveAsVideo(raw_path)
        self.save_as_vid_threads.append(thread)
        thread.finished.connect(lambda: self._cleanup_thread(thread))
        thread.start()

        if self.SpatialBox.isChecked() or self.TemporalBox.isChecked():
            # Launch processing
            # launch_processing(raw_path,
            #                   self.SpatialBox.isChecked(),
            #                   self.TemporalBox.isChecked(),
            #                   self.BFI_CheckBox.isEnabled() and self.BFI_CheckBox.isChecked(),
            #                   (self.spatial_window1.value(), self.spatial_window2.value()),
            #                   self.temp_window.value())

            params = {'output_dir':None,
                      'videos':[raw_path],
                      'temporal':self.TemporalBox.isChecked(),
                      'temporal_window':self.temp_window.value(),
                      'spatial':self.SpatialBox.isChecked(),
                      'spatial_window':(self.spatial_window1.value(), self.spatial_window2.value()),
                      'bfi':self.BFI_CheckBox.isEnabled() and self.BFI_CheckBox.isChecked()}
            process = multiprocessing.Process(target=process_videos.main, kwargs=params)
            process.start()

            process_id = len(self.processes) + 1
            self.processes[process_id] = process
            # item = QListWidgetItem(f"Video {process_id}: Processing...")
            row = self.process_table.rowCount()
            self.process_table.insertRow(row)
            self.process_table.setRowHeight(row, 25)
            self.process_table.setItem(row, 0, QTableWidgetItem(f"{process_id}"))
            self.process_table.setItem(row, 1, QTableWidgetItem("Processing..."))

            monitor_thread = ProcessMonitor(process_id, process)
            monitor_thread.finished_signal.connect(lambda pid=process_id: self.update_thread_status(pid))
            monitor_thread.start()
            self.monitor_threads[process_id] = monitor_thread


    def update_thread_status(self, process_id):
        print("HERE")
        for row in range(self.process_table.rowCount()):
            item = self.process_table.item(row, 0)
            if item and int(item.text()) == process_id:
                self.process_table.setItem(row, 1, QTableWidgetItem("Finished!"))
                break


    def _cleanup_thread(self, thread):
        self.save_as_vid_threads.remove(thread)
        print(f"Thread for {thread.raw_path} has finished.")

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

    def __init__(self, signal_queue:multiprocessing.Queue):
        super().__init__()
        print('DisplayProcessed initialized')
        self.signal_queue = signal_queue
        self.ThreadActive = False
        self.dtype = LIVE_DTYPE

        self.processed_shm = shared_memory.SharedMemory(name='processed')
        self.processed_frame = np.ndarray((1920,1080), dtype=self.dtype, buffer=self.processed_shm.buf)

    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            # Wait for new frame signal and update frame
            self.signal_queue.get()
            self.ImageUpdate.emit(self.processed_frame)
            # print('got a frame')
            # print(self.ThreadActive)
        print('out of while in DisplayProcessed')
        self.processed_shm.close()

    def stop(self):
        self.ThreadActive = False
        print('4')
        # self.quit()


class CameraWorker(QThread):
    ImageUpdate = pyqtSignal(np.ndarray)

    def __init__(self, frame_queue, recording:threading.Event, display_raw=True):
        super().__init__()
        self.frame_queue = frame_queue
        self.ThreadActive = False
        self.recording = recording
        self.display_raw = display_raw

        if not display_raw:
            self.raw_shm = shared_memory.SharedMemory(create=False, name='spatial')
            self.raw_frame = np.ndarray((1920,1080), dtype=LIVE_DTYPE, buffer=self.raw_shm.buf)

    def run(self):
        self.ThreadActive = True

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
                self.stop()

            with sdk.open_camera(available_cameras[0]) as camera:
                camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
                camera.exposure_time_us = 10000  # set exposure to 10 ms #TODO this should eventually be a parameter
                camera.image_poll_timeout_ms = 1000  # 1 second polling timeout

                camera.arm(2)
                camera.issue_software_trigger()

                while self.ThreadActive:
                    frame = camera.get_pending_frame_or_null()
                    if frame is not None:
                        # print('frame number: ', frame.frame_count)
                        frame_float32 = np.copy(frame.image_buffer).astype(np.float32)
                        if self.recording.is_set():
                            # Currently recording
                            self.frame_queue.put(frame_float32)
                        frame_float32 = cv2.transpose(frame_float32)
                        frame_float32 = cv2.flip(frame_float32, -1)

                        frame = ((frame_float32-frame_float32.min())/(frame_float32.max()-frame_float32.min()) * 255).astype(np.uint8)
                        
                        # ConvertToQtFormat = QImage(FlippedImage.data, 
                        #                            FlippedImage.shape[1], 
                        #                            FlippedImage.shape[0], 
                        #                            QImage.Format_Grayscale8)
                        # Pic = ConvertToQtFormat.scaled(1920/2, 1080/2, Qt.KeepAspectRatio)
                        if self.display_raw:
                            self.ImageUpdate.emit(frame)
                        else:
                            # Live processing is activated
                            self.raw_frame[:] = frame_float32[:]
                if not self.display_raw:
                    self.raw_shm.close()
                        
    
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
        print('Save thread stopping...')
        self.finished.emit(self.output_file)  # Emit the signal when saving is finished


class SaveAsVideo(QThread):
    def __init__(self, raw_path: Path):
        super().__init__()
        self.raw_path = raw_path

    def run(self):
        # Read the whole sequence from the binary file and save as pngs as well as avi
            data = np.fromfile(self.raw_path, dtype=np.float32)
            num_frames = data.size // (1080*1920)
            full_sequence_raw = data.reshape((num_frames, 1080, 1920))

            # Save video
            print("Saving raw frames")
            # (args.output_dir/'raw_frames').mkdir(parents=True, exist_ok=True)

            # print('sequence shape: ', full_sequence_raw.shape)
            # print('dtype: ', full_sequence_raw.dtype)
            # print('min: ', np.min(full_sequence_raw))
            # print('max: ', np.max(full_sequence_raw))
            full_sequence_raw=(full_sequence_raw/full_sequence_raw.max())*255 #TODO this isn't ideal

            fps = 30  # frames per second #TODO change this
            frame_size = (1920, 1080) # CAREFUL! THIS NEEDS TO  BE INVERTED!!
            # print('frame_size: ', frame_size)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
            out = cv2.VideoWriter((self.raw_path.with_suffix('.avi')).as_posix(), fourcc, fps, frame_size, False)

            # full_sequence_raw = cv2.cvtColor(full_sequence_raw, cv2.COLOR_GRAY2BGR)
            # print('raw range when saving: ', np.min(full_sequence_raw), ' -- ', np.max(full_sequence_raw)) 
            # print('dtype ----- ', full_sequence_raw.dtype)

            for i in range(full_sequence_raw.shape[0]):
                frame = full_sequence_raw[i]
                # Save as png
                # cv2.imwrite((args.output_dir/'raw_frames'/f'frame{i:04d}.png').as_posix(), frame)
                # print('dtype ----- ', frame.dtype)
                # Save as avi
                frame = frame.astype(np.uint8)
                out.write(frame)
            out.release()

def spatial_worker(kernel_size, out_queue:multiprocessing.Queue, stop_event,
                raw_mem:str="spatial", processed_mem:str="processed"):

    raw_shm = shared_memory.SharedMemory(create=False, name=raw_mem)
    raw_frame = np.ndarray((1920,1080), dtype=LIVE_DTYPE, buffer=raw_shm.buf)

    processed_shm = shared_memory.SharedMemory(create=False, name=processed_mem)
    processed_frame = np.ndarray((1920,1080), dtype=LIVE_DTYPE, buffer=processed_shm.buf)
        
    while not stop_event.is_set():
        working_copy = raw_frame.copy()
        processed_frame[:] = spatial_bfi_frame(working_copy, kernel_size)[:]
        out_queue.put('signal')
        # print('frame processed and signal sent')
    raw_shm.close()
    processed_shm.close()
    print('3')


if __name__ == "__main__":
    # create pyqt5 app
    App = MyApp(sys.argv) #TODO Add wrapper to cleanup live processing workers upon exit

    # create the instance of our Window
    window = Window()

    App.set_main_window(window)

    # start the app
    sys.exit(App.exec_())

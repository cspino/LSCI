from multiprocessing import Process, shared_memory
import os
import subprocess
import time
import skvideo
from PyQt5.QtWidgets import QApplication, QMainWindow, QButtonGroup, QFileDialog, QMessageBox, QTableWidgetItem, QPushButton
import sys

from utils import read_video, spatial_bfi_frame
# from utils import launch_processing
sys.path.append('..') # this is to be able to import files in project folder
# from windows_setup import configure_path
# configure_path()
import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QRectF, QThread, pyqtSignal
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

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

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
        if self.main_window.processed_mem:
            self.main_window.processed_mem.close()
        if self.main_window.spatial_mem:
            self.main_window.spatial_mem.close()
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
        self.CameraWorker = None
        self.data = None
        self.frame_counter = 0
        self.avg_max = 0 # to keep track of the average profile max when recording
        self.cmap = 'gray' #keep track of current colormap

        # To load a video from raw file
        self.loaded_vid = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.read_from_loaded_vid)
        self.playback_index = 0
        self.play_pause_button.clicked.connect(self.togglePlayPause)
        self.frame_slider.valueChanged.connect(self.goToFrame)
        self.playback_running = False

        # Slider and play/pause button are kept hidden until we load a file
        self.play_pause_button.hide()
        self.frame_slider.hide()
        self.cmap_button.hide()

        # To keep track of processing launched (post)
        self.processes = {} # Stores processes
        self.monitor_threads = {} # Stores QThreads

        # Live processing
        self.live_processing = False
        self.signal_queue = multiprocessing.Queue()
        self.DisplayProcessed = None # This stores the process responsible for real time processing
        self.processed_mem = None
        self.spatial_mem = None

    # method for components
    def UiComponents(self):
        # Apply default config
        # TODO this can be moved to the UI document if we pass the config
        _translate = QtCore.QCoreApplication.translate
        self.output_dir_line_edit.setText(_translate("MainWindow", self.config['output_dir']))
        self.spatial_window1.setProperty("value", self.config['spatial_window'])
        self.spatial_window2.setProperty("value", self.config['spatial_window'])
        self.temp_window.setProperty("value", self.config['temporal_window'])

        self.TemporalBox.setChecked(self.config['temporal'])
        self.SpatialBox.setChecked(self.config['spatial'])
        self.BFI_CheckBox.setChecked(self.config['bfi'])
        self.check_bfi_state()

        self.StartButton.clicked.connect(lambda: self.ToggleAcquisition(False))
        self.load_button.clicked.connect(lambda: self.LoadFile(self.output_dir))

        self.browse_button.clicked.connect(lambda: self.browse_directory(self.output_dir))

        self.RecordButton.clicked.connect(self.ToggleRecord)

        self.SpatialBox.toggled.connect(self.check_bfi_state)
        self.TemporalBox.toggled.connect(self.check_bfi_state)

        # Synchronize both spin boxes for spatial window size
        self.spatial_window1.valueChanged.connect(self.sync_spatial_window_boxes)
        self.spatial_window2.valueChanged.connect(self.sync_spatial_window_boxes)

        self.process_table.setColumnWidth(0,25)
        self.process_table.setColumnWidth(1,117)
        self.process_table.setColumnWidth(2,3)

        # Radio buttons for display
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.radio_raw)
        self.button_group.addButton(self.radio_spatial)


        pg.setConfigOptions(antialias=True)
        # self.win = pg.GraphicsLayoutWidget()
        self.win.ci.setContentsMargins(0, 0, 0, 0)

        self.profile_plot = self.win.addPlot(row=0,col=1)
        self.profile_plot.setMaximumWidth(100)
        self.profile = self.profile_plot.plot()

        self.view = self.win.addViewBox(row=0, col=2)
        # view.setBackgroundColor('w')
        self.view.setMouseEnabled(x=False, y=False)
        self.view.setAspectLocked(True)
        self.view.enableAutoRange()
        self.view.setRange(QRectF(0, 0, 1920, 1080))  #TODO store this in config file

        self.img = pg.ImageItem()
        # self.img.setPos(-frac*1920, -frac*1080)
        self.view.addItem(self.img)

        # label for plot max and average max 
        self.profile_max_label = pg.TextItem("ROI Profile max: ", anchor=(0.5,0))
        self.view.addItem(self.profile_max_label)

        self.profile_max_avg_label = pg.TextItem("average max: ", anchor=(0.5,0))
        self.view.addItem(self.profile_max_avg_label)

        self.profile_max_label.setPos(1920/3, 0)
        self.profile_max_avg_label.setPos(1920*2/3, 0)


        self.gradient_bar = pg.ColorBarItem(interactive=False)
        self.gradient_bar.setImageItem(self.img)
        # self.gradient_bar.setOrientation('horizontal')  # Horizontal bar
        self.win.addItem(self.gradient_bar, row=0, col=3)

        # Custom ROI for selecting an image region
        self.roi = pg.ROI([0, 0], [1920, 1080], maxBounds=QRectF(0,0,1920,1080))
        self.roi_indices = [0,1920,0,1080]
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.view.addItem(self.roi)
        self.roi.setZValue(10)  # make sure ROI is drawn above image

        self.roi.sigRegionChanged.connect(self.updateRoi)

        icon = QIcon()
        icon.addPixmap(QPixmap("../resources/cmap_button_icon.png"), QIcon.Normal, QIcon.Off)
        self.cmap_button.setIcon(icon)
        self.cmap_button.clicked.connect(self.ToggleColorMap)

        self.open_file_icon = QIcon()
        self.open_file_icon.addPixmap(QPixmap("../resources/open_file_icon.png"), QIcon.Normal, QIcon.Off)


    def updateRoi(self):
        self.reset_average()
        pos = self.roi.pos()
        size = self.roi.size()
        self.roi_indices = [int(pos.x()), int(pos.x()+size.x()), int(pos.y()), int(pos.y()+size.y())] #x1, x2, y1, y2
        # print(self.roi_indices)

        if not self.data is None:
            self.updatePlot()


    def updatePlot(self):
        # selected = self.roi.getArrayRegion(self.data, self.img)
        # print("y1: ", self.roi_indices[3])
        # print("y2: ", self.roi_indices[2])
        # print("x1: ", self.roi_indices[0])
        # print("x2: ", self.roi_indices[1])

        selected = self.data[self.roi_indices[0]:self.roi_indices[1],
                             self.roi_indices[2]:self.roi_indices[3]]
        

        y_values = selected.mean(axis=0)
        x_values = np.arange(len(y_values))
        self.profile.setData(y_values, x_values)
        self.profile_max_label.setText("ROI Profile max: {:.3f}".format(np.max(y_values)))

        if self.recording.is_set() or self.playback_running:
            if self.frame_counter == 0:
                self.avg_max = float(np.max(y_values))
            else:
                self.avg_max = (self.avg_max*(self.frame_counter)+float(np.max(y_values)))/(self.frame_counter+1)
            self.profile_max_avg_label.setText("average max: {:.3f}".format(self.avg_max))
            self.frame_counter += 1

    def reset_average(self):
        self.avg_max = 0
        self.frame_counter = 0


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

    def browse_directory(self, directory:Path):
        """Open a directory dialog to select an output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", directory.as_posix())
        if dir_path:
            self.output_dir = Path(dir_path)  # Set the selected directory
            self.output_dir_line_edit.setText(str(self.output_dir))

    def SetColorMap(self, cmap_name:str, range_max):
        if cmap_name == 'gray':
            colormap = pg.colormap.getFromMatplotlib('gray')
        elif cmap_name == 'magma':
            colormap = pg.colormap.get('magma')
        else:
            try:
                colormap = pg.colormap.get(cmap_name)
            except:
                print("WARNING: Colormap {colormap} is not valid.")
                return
        self.cmap = cmap_name
        self.gradient_bar.setColorMap(colormap)
        # self.gradient_bar.setLevels(low=0, high=range_max) #TODO this might not be necessary

    def ToggleColorMap(self):
        if self.cmap == 'gray':
            colormap = pg.colormap.get('magma')
            self.cmap = 'magma'
        else:
            colormap = pg.colormap.getFromMatplotlib('gray')
            self.cmap = 'gray'

        self.gradient_bar.setColorMap(colormap)


    def camera_error(self):
        print("IN ERROR FUNCTION")
        self.ToggleAcquisition(force_off=True) # Turn off Acquisition

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Camera Detection Error")
        msg.setText("No camera detected. Make sure it is properly connected and try again.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


    def LoadFile(self, directory:Path):
        # Open File Dialog to select file
        file_filter = "RAW Files (*.raw);;All Files (*.*)"
        filepath, _ = QFileDialog.getOpenFileName(self, "Select .raw file to open", directory.as_posix(),file_filter)
        if filepath:
            self.loaded_vid, self.fps = read_video(Path(filepath))

            if not self.loaded_vid is None:
                self.playback_index = 0 # reset index
                self.playback_len = self.loaded_vid.shape[0]
                self.frame_slider.setMaximum(self.playback_len-1)
                self.switch_cmap = True
                self.StartPlayback()
                self.radio_raw.blockSignals(False)
                self.radio_spatial.blockSignals(False)
                # Display start/stop button and slider
                self.play_pause_button.show()
                self.frame_slider.show()
                self.cmap_button.show()
                self.profile_max_avg_label.setText("average max: ")


    def create_shared_mems(self):
        # If the app crashed, shared memories might not have been cleaned up
        # so we catch FileExistsErrors
        n_bytes = np.empty((1920,1080), dtype=LIVE_DTYPE).nbytes
        if not self.spatial_mem:
            try:
                self.spatial_mem = shared_memory.SharedMemory(create=True, name='spatial', size=n_bytes)
            except FileExistsError:
                self.spatial_mem = shared_memory.SharedMemory(create=False, name='spatial')
        if not self.processed_mem:
            try:
                self.processed_mem = shared_memory.SharedMemory(create=True, name='processed', size=n_bytes)
            except FileExistsError:
                self.processed_mem = shared_memory.SharedMemory(create=False, name='processed')

    def PausePlayback(self):
        self.playback_running = False
        self.timer.stop()
        self.play_pause_button.setText("Play")

    
    def StartPlayback(self):
        self.playback_running = True
        self.play_pause_button.setText("Pause")
        self.timer.start(1000//self.fps)

    def togglePlayPause(self):
        if self.playback_running:
            self.PausePlayback()
        else:
            self.StartPlayback()

    def goToFrame(self):
        self.reset_average()
        self.playback_index = self.frame_slider.value()
        self.read_from_loaded_vid()


    def read_from_loaded_vid(self):
        # print("reading frame")
        if self.playback_index >= self.playback_len-1:
            # At the end of the vid
            self.PausePlayback()
        else:
            self.playback_index += 1

        frame = self.loaded_vid[self.playback_index-1,:,:]
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, -1)
        
        # frame = ((frame-frame.min())/(frame.max()-frame.min()) * 255).astype(np.uint8) #TODO unsure if needed for raw
        self.ImageUpdateSlot(frame, processed=False)
        self.update_frame_slider()


    def update_frame_slider(self):
        # We need to block signals when we change the value to avoid a feedback loop
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.playback_index)
        self.frame_slider.blockSignals(False)

    def ToggleAcquisition(self, force_off=False):
        print(force_off)
        if self.StartButton.isChecked() and not force_off:
            # Start the acquisition
            print("Starting acquisition...")
            self.StartButton.setText("Stop")

            if self.playback_running:
                #From a file
                print("pausing playback here")
                self.PausePlayback()
            
            self.play_pause_button.hide()
            self.frame_slider.hide()
            self.cmap_button.hide()
            self.profile_max_avg_label.setText("average max: ")
            
            # Disable radio buttons
            for button in self.button_group.buttons():
                button.setEnabled(False)

            if self.radio_spatial.isChecked():
                self.create_shared_mems()
            
            # Start worker to handle video feed
            self.CameraWorker = CameraWorker(self.frame_queue, self.recording, self.radio_raw.isChecked())
            self.CameraWorker.start()
            self.CameraWorker.CameraError.connect(self.camera_error)

            if self.radio_raw.isChecked():
                self.CameraWorker.ImageUpdate.connect(self.ImageUpdateSlot)
                self.CameraWorker.FirstFrame.connect(lambda cmap, r_max: self.SetColorMap(cmap, r_max))

            elif self.radio_spatial.isChecked():
                self.SetupLiveProcessing()

            self.RecordButton.setEnabled(True)

        else:
            if force_off:
                self.StartButton.setChecked(False)
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
                self.DisplayProcessed = None
                print('2')

    def SetupLiveProcessing(self):
        self.stop_live_processing = multiprocessing.Event()
        self.pause_live_processing = multiprocessing.Event()
        self.live_process = multiprocessing.Process(target=spatial_worker, args=(self.temp_window.value(),
                                                                                    self.signal_queue,
                                                                                    self.stop_live_processing))
        self.live_process.start()

        self.DisplayProcessed = DisplayProcessed(self.signal_queue)
        self.DisplayProcessed.start()
        self.DisplayProcessed.ImageUpdate.connect(lambda image: self.ImageUpdateSlot(image, processed=True))
        self.DisplayProcessed.FirstFrame.connect(lambda cmap, r_max: self.SetColorMap(cmap, r_max))
        self.live_processing = True
    

    def ToggleRecord(self):
        if  self.RecordButton.isChecked():
            # Start Saving
            self.RecordButton.setText("Stop Record")
            # self.RecordButton.setChecked(True)
            self.StartButton.setEnabled(False)

            self.recording.set()
            # Start the save worker to save frames
            self.save_thread = SaveThread(self.output_dir/'raw_frames.raw', self.frame_queue)
            self.save_thread.finished.connect(self.save_raw_as_vid)  # Connect the signal for when save finishes
            self.save_thread.start()

        else:
            # Stop Saving
            self.RecordButton.setText("Start Record")
            # self.RecordButton.setChecked(False)
            
            self.save_thread.requestInterruption()
            self.recording.clear()
            self.StartButton.setEnabled(True)
            self.reset_average()


    def ImageUpdateSlot(self, image, processed=False):
        # If we're showing processed frames (processed=True), 
        # then we want to make sure that self.live_processing hasnt been 
        # set to false to avoid acessing the shared memory that has 
        # been discarded
        if processed:
            if not self.live_processing:
                return
        self.data = image
        self.img.setImage(image, autoLevels=True)
        self.gradient_bar.setLevels(self.img.quickMinMax())
        self.view.autoRange()

        self.updatePlot()


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
            monitor_thread.finished_signal.connect(lambda pid= process_id, out=self.output_dir : self.update_thread_status(pid, out))
            monitor_thread.start()
            self.monitor_threads[process_id] = monitor_thread


    def update_thread_status(self, process_id, directory:Path):
        open_button = QPushButton('')
        open_button.clicked.connect(lambda: self.LoadFile(directory))
        open_button.setIcon(self.open_file_icon)

        for row in range(self.process_table.rowCount()):
            item = self.process_table.item(row, 0)
            if item and int(item.text()) == process_id:
                self.process_table.setItem(row, 1, QTableWidgetItem("Finished!"))
                self.process_table.setCellWidget(row, 2, open_button)
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
    FirstFrame = pyqtSignal(str, int)

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
            signal = self.signal_queue.get()
            if signal:
                # If signal is True, we're receiving the first frame
                self.FirstFrame.emit('magma', int(self.processed_frame.max()))
            
            self.ImageUpdate.emit(self.processed_frame)

        print('out of while in DisplayProcessed')
        self.processed_shm.close()

    def stop(self):
        self.ThreadActive = False
        print('4')


class CameraWorker(QThread):
    ImageUpdate = pyqtSignal(np.ndarray)
    FirstFrame = pyqtSignal(str, int)
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
            # if on Windows, use the provided setup script to add the DLLs folder to the PATH
            from windows_setup import configure_path
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
                        camera.exposure_time_us = 10000  # set exposure to 10 ms #TODO this should eventually be a parameter
                        camera.image_poll_timeout_ms = 1000  # 1 second polling timeout

                        camera.arm(2)
                        camera.issue_software_trigger()

                        first_frame = True
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
                                
                                if self.display_raw:
                                    self.ImageUpdate.emit(frame)
                                    if first_frame:
                                        self.FirstFrame.emit('gray', frame.max())
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
    
    first_frame = True
    while not stop_event.is_set():

        working_copy = raw_frame.copy()
        processed_frame[:] = spatial_bfi_frame(working_copy, kernel_size)[:]
        
        out_queue.put(first_frame)
        first_frame=False

    raw_shm.close()
    processed_shm.close()
    print('3')


if __name__ == "__main__":
    # create pyqt5 app
    App = MyApp(sys.argv) 

    # create the instance of our Window
    window = Window()

    App.set_main_window(window)

    # start the app
    sys.exit(App.exec_())

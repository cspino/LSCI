import os
import subprocess
from PyQt5.QtWidgets import *
import sys
sys.path.append('..') # this is to be able to import files in project folder
import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore
import pyqtgraph.ptime as ptime
import cv2
import toml
import queue
from pathlib import Path
import threading
from app_ui import Ui_MainWindow

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE
from thorlabs_tsi_sdk.tl_camera_enums import DATA_RATE

pg.setConfigOptions(useOpenGL=True, enableExperimental=True) # Reduces lag in displayed stream

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
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if dir_path:
            self.output_dir = Path(dir_path)  # Set the selected directory
            self.output_dir_line_edit.setText(str(self.output_dir))

    def ToggleAcquisition(self):
        if self.StartButton.isChecked():
            # Start the acquisition
            print("Starting acquisition...")
            self.StartButton.setText("Stop")
            # self.StartButton.setChecked(True)
            
            # Start worker to handle video feed
            self.CameraWorker = CameraWorker(self.frame_queue, self.recording)
            self.CameraWorker.start()
            self.CameraWorker.ImageUpdate.connect(self.ImageUpdateSlot)

            self.RecordButton.setEnabled(True)

        else:
            # Stop the acquisition
            print("Stopping acquisition...")
            self.StartButton.setText("Start")
            # # self.StartButton.setChecked(False)
            self.RecordButton.setEnabled(False)
            
            if self.CameraWorker:
                self.CameraWorker.stop()  # Stop the video feed worker
                

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


    def ImageUpdateSlot(self, image):
        # self.img.setScale(1)
        # self.img.setPos(0, 0)
        self.img.setImage(image)
        # self.img.setRect(QRectF(0, 0, 1920, 1080))
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
            # Start subprocess to process .raw vid
            command = ['python', 'process_videos.py',
                        '-v', raw_path.as_posix(),
                        '-o', (raw_path.parent).as_posix()]
            
            if self.SpatialBox.isChecked():
                command.append('--spatial')
                command.extend(['-sw', (self.spatial_window1.value(), self.spatial_window2.value())])

            if self.TemporalBox.isChecked():
                command.append('--temporal')
                command.extend(['-tw', self.temporal_window.value()])

            if self.BFI_CheckBox.isEnabled() and self.BFI_CheckBox.isChecked():
                command.append('--bfi')

            subprocess.Popen(command, shell=True)

    def _cleanup_thread(self, thread):
        self.save_as_vid_threads.remove(thread)
        print(f"Thread for {thread.raw_path} has finished.")


class CameraWorker(QThread):
    ImageUpdate = pyqtSignal(np.ndarray)

    def __init__(self, frame_queue, recording:threading.Event):
        super().__init__()
        self.frame_queue = frame_queue
        self.ThreadActive = False
        self.recording = recording

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
                        frame = np.copy(frame.image_buffer).astype(np.float32)
                        if self.recording.is_set():
                            # Currently recording
                            self.frame_queue.put(frame)
                        frame = ((frame-frame.min())/(frame.max()-frame.min()) * 255).astype(np.uint8)
                        frame = cv2.transpose(frame)
                        frame = cv2.flip(frame, -1)
                        # ConvertToQtFormat = QImage(FlippedImage.data, 
                        #                            FlippedImage.shape[1], 
                        #                            FlippedImage.shape[0], 
                        #                            QImage.Format_Grayscale8)
                        # Pic = ConvertToQtFormat.scaled(1920/2, 1080/2, Qt.KeepAspectRatio)
                        self.ImageUpdate.emit(frame)
    
    def stop(self):
        self.ThreadActive = False
        self.quit()


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
            print('raw range when saving: ', np.min(full_sequence_raw), ' -- ', np.max(full_sequence_raw)) 
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


if __name__ == "__main__":
    # create pyqt5 app
    App = QApplication(sys.argv)

    # create the instance of our Window
    window = Window()

    # start the app
    sys.exit(App.exec())

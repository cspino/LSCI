import sys
sys.path.append('..') # this is to be able to import files in project folder

import numpy as np
import pyqtgraph as pg
from multiprocessing import shared_memory
import multiprocessing
import cv2
import toml
import queue
from pathlib import Path
import threading

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QRectF
from PyQt5 import QtCore, QtWidgets

from utils.utils import read_video
from utils.app_utils import *
from app_ui import Ui_MainWindow
import process_videos


pg.setConfigOptions(useOpenGL=True, enableExperimental=True) # Reduces lag in displayed stream

class MyApp(QtWidgets.QApplication):
    def __init__(self, *args):
        super().__init__(*args)
        self.aboutToQuit.connect(self.cleanup)
        self.main_window = None

    def set_main_window(self, window):
        self.main_window = window

    def cleanup(self):
        """
        Called when the app shuts down. Closes the shared memories
        and shuts down live processing.
        """
        if self.main_window.live_processing:
            self.main_window.stop_live_processing.set()
        if self.main_window.processed_mem:
            self.main_window.processed_mem.close()
        if self.main_window.raw_mem:
            self.main_window.raw_mem.close()
        print('cleanup done')


class Window(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()

        self.config = toml.load('../config/default_config.toml')
        self.output_dir = Path(self.config['output_dir'])
        self.img_width = self.config['width']
        self.img_height = self.config['height']

        self.setupUi(self) # UI base, generated with PyQT Designer
        self.UiComponents() # Additional UI elements
        self.show()

        ## Initialize variables
        # For Live Mode
        self.saving_queue = queue.Queue() # Used to transfer frames from CameraWorker to SaveThread
        self.save_thread = None # Stores the worker SaveThread that saves raw frames
        self.save_as_vid_threads = [] # Stores SaveAsVideo threads (responsible for saving raw vid as .avi)

        self.recording = threading.Event()
        self.CameraWorker = None
        self.data = None # Stores the current frame 
        self.frame_counter = 0 # Used to compute the average profile max
        self.avg_max = 0 # Keeps track of the average profile max when recording
        self.cmap = 'gray' # Keeps track of current colormap
        self.auto_window_level = True

        # For Playback Mode
        self.loaded_vid = None
        self.timer = QtCore.QTimer(self) # Used for playback at set frame rate
        self.timer.timeout.connect(self.read_from_loaded_vid)
        self.playback_index = 0
        self.play_pause_button.clicked.connect(self.togglePlayPause)
        self.frame_slider.valueChanged.connect(self.goToFrame)
        self.playback_running = False

        # Slider and play/pause button are kept hidden until we load a file
        self.play_pause_button.hide()
        self.frame_slider.hide()
        self.cmap_button.hide()

        # To keep track of post-processing workers
        self.processes = {} # Stores processes
        self.monitor_threads = {} # Stores QThreads that monitor the processes

        # Live processing
        self.live_processing = False
        self.signal_queue = multiprocessing.Queue() # Used to communicate between DisplayProcessed and spatial_worker
        self.DisplayProcessed = None
        self.processed_mem = None 
        self.raw_mem = None

    def UiComponents(self):
        # Apply default config
        _translate = QtCore.QCoreApplication.translate
        self.output_dir_line_edit.setText(_translate("MainWindow", self.config['output_dir']))
        self.spatial_window1.setProperty("value", self.config['spatial_window'])
        self.spatial_window2.setProperty("value", self.config['spatial_window'])
        self.temp_window.setProperty("value", self.config['temporal_window'])

        self.TemporalBox.setChecked(self.config['temporal'])
        self.SpatialBox.setChecked(self.config['spatial'])
        self.Profile_CheckBox.setChecked(self.config['profile_plots'])
        self.check_profile_state()

        # Connect various buttons
        self.StartButton.clicked.connect(lambda: self.ToggleAcquisition(False))
        self.load_button.clicked.connect(lambda: self.LoadFile(self.output_dir))

        self.browse_button.clicked.connect(lambda: self.browse_directory(self.output_dir))

        self.RecordButton.clicked.connect(self.ToggleRecord)

        self.SpatialBox.toggled.connect(self.check_profile_state)
        self.TemporalBox.toggled.connect(self.check_profile_state)

        # Synchronize both spin boxes for spatial window size
        self.spatial_window1.valueChanged.connect(self.sync_spatial_window_boxes)
        self.spatial_window2.valueChanged.connect(self.sync_spatial_window_boxes)

        # Table that displays post-process worker status
        self.process_table.setColumnWidth(0,25)
        self.process_table.setColumnWidth(1,117)
        self.process_table.setColumnWidth(2,3)

        # Radio buttons for Live Display Mode
        self.button_group = QtWidgets.QButtonGroup(self)
        self.button_group.addButton(self.radio_raw)
        self.button_group.addButton(self.radio_spatial)

        pg.setConfigOptions(antialias=True)
        self.win.ci.setContentsMargins(0, 0, 0, 0)

        # Plot that displays x-axis profile
        self.profile_plot = self.win.addPlot(row=0,col=1)
        self.profile_plot.setMaximumWidth(100)
        self.profile = self.profile_plot.plot()

        # ViewBox to display the videos
        self.view = self.win.addViewBox(row=0, col=2)
        self.view.setMouseEnabled(x=False, y=False)
        self.view.setAspectLocked(True)
        self.view.enableAutoRange()
        self.view.setRange(QRectF(0, 0, self.img_width, self.img_height))

        # ImageItem in ViewBox stores the current image (frame)
        self.img = pg.ImageItem()
        self.view.addItem(self.img)

        # Labels for plot max and average max 
        self.profile_max_label = pg.TextItem("ROI Profile max: ", anchor=(0.5,0))
        self.view.addItem(self.profile_max_label)

        self.profile_max_avg_label = pg.TextItem("average max: ", anchor=(0.5,0))
        self.view.addItem(self.profile_max_avg_label)

        self.profile_max_label.setPos(self.img_width/3, 0)
        self.profile_max_avg_label.setPos(self.img_width*2/3, 0)

        # ColorBar
        self.colorbar = InteractiveColorBar(main_window=self, interactive=True)
        self.colorbar.setImageItem(self.img)
        self.win.addItem(self.colorbar, row=0, col=3)

        # Custom ROI for selecting an image region (used to update the profile plot)
        self.roi = pg.ROI([0, 0], [self.img_width, self.img_height], maxBounds=QRectF(0,0,self.img_width,self.img_height))
        self.roi_indices = [0,self.img_width,0,self.img_height]
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.view.addItem(self.roi)
        self.roi.setZValue(10)
        self.roi.sigRegionChanged.connect(self.updateRoi)

        # Configure cmap button
        icon = QIcon()
        icon.addPixmap(QPixmap("../resources/cmap_button_icon.png"), QIcon.Normal, QIcon.Off)
        self.cmap_button.setIcon(icon)
        self.cmap_button.clicked.connect(self.ToggleColorMap)

        # Icon for post-process table
        self.open_file_icon = QIcon()
        self.open_file_icon.addPixmap(QPixmap("../resources/open_file_icon.png"), QIcon.Normal, QIcon.Off)


    def updateRoi(self):
        """
        Update ROI indices and plot
        """
        self.reset_average()
        pos = self.roi.pos()
        size = self.roi.size()
        self.roi_indices = [int(pos.x()), int(pos.x()+size.x()), int(pos.y()), int(pos.y()+size.y())] #x1, x2, y1, y2
        # print(self.roi_indices)

        if not self.data is None:
            self.updatePlot()


    def updatePlot(self):
        """
        Update the profile plot using current ROI
        """
        selected = self.data[self.roi_indices[0]:self.roi_indices[1],
                             self.roi_indices[2]:self.roi_indices[3]]

        y_values = selected.mean(axis=0)
        x_values = np.arange(len(y_values))
        self.profile.setData(y_values, x_values)
        self.profile_max_label.setText("ROI Profile max: {:.3f}".format(np.max(y_values)))

        if self.recording.is_set() or self.playback_running:
            # Update average
            if self.frame_counter == 0:
                self.avg_max = float(np.max(y_values))
            else:
                self.avg_max = (self.avg_max*(self.frame_counter)+float(np.max(y_values)))/(self.frame_counter+1)
            self.profile_max_avg_label.setText("average max: {:.3f}".format(self.avg_max))
            self.frame_counter += 1

    def reset_average(self):
        """
        Reset profile plot average
        """
        self.avg_max = 0
        self.frame_counter = 0

    def sync_spatial_window_boxes(self):
        """
        Makes sure both boxes for the spatial window size always display the same value
        """
        sender = self.sender()  # Get the widget that triggered the signal

        if sender == self.spatial_window1:
            self.spatial_window2.setValue(self.spatial_window1.value())
        elif sender == self.spatial_window2:
            self.spatial_window1.setValue(self.spatial_window2.value())

    def check_profile_state(self):
        """
        Makes sure the Profile plot checkbox is only available if Spatial 
        or Temporal is checked
        """
        if self.SpatialBox.isChecked() or self.TemporalBox.isChecked():
            self.Profile_CheckBox.setEnabled(True)
        else:
            self.Profile_CheckBox.setEnabled(False)

    def browse_directory(self, directory:Path):
        """
        Open a directory dialog to select an output directory
        """
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory", directory.as_posix())
        if dir_path:
            self.output_dir = Path(dir_path)
            self.output_dir_line_edit.setText(str(self.output_dir))

    def SetColorMap(self, cmap_name:str):
        """
        Set the specified colormap 
        """
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
        self.colorbar.setColorMap(colormap)

    def ToggleColorMap(self):
        """
        Toggle between gray and magma colormaps
        """
        if self.cmap == 'gray':
            colormap = pg.colormap.get('magma')
            self.cmap = 'magma'
        else:
            colormap = pg.colormap.getFromMatplotlib('gray')
            self.cmap = 'gray'

        self.colorbar.setColorMap(colormap)

    def camera_error(self):
        """
        Show a pop up error message signaling a camera error
        """
        self.ToggleAcquisition(force_off=True) # Turn off Acquisition

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle("Camera Detection Error")
        msg.setText("No camera detected. Make sure it is properly connected and try again.")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def create_shared_mems(self):
        """
        Create the shared memory files needed for communication with the spatial_worker
        """
        # If the app crashed, shared memories might not have been cleaned up
        # so we catch FileExistsErrors
        n_bytes = np.empty((self.img_width,self.img_height), dtype=np.float32).nbytes
        if not self.raw_mem:
            try:
                self.raw_mem = shared_memory.SharedMemory(create=True, name='spatial', size=n_bytes)
            except FileExistsError:
                self.raw_mem = shared_memory.SharedMemory(create=False, name='spatial')
        if not self.processed_mem:
            try:
                self.processed_mem = shared_memory.SharedMemory(create=True, name='processed', size=n_bytes)
            except FileExistsError:
                self.processed_mem = shared_memory.SharedMemory(create=False, name='processed')


    ### FOR PLAYBACK MODE
    def LoadFile(self, directory:Path):
        """
        Used to initiate Playback Mode; Opens a FileDialog to select the file for playback
        """
        # Open File Dialog to select file
        file_filter = "RAW Files (*.raw);;All Files (*.*)" # Only show .raw files
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select .raw file to open", directory.as_posix(), file_filter)
        if filepath:
            self.loaded_vid, self.fps = read_video(Path(filepath))

            if not self.loaded_vid is None:
                ## Prepare for Playback Mode
                self.playback_index = 0 # Reset index
                self.playback_len = self.loaded_vid.shape[0]
                self.frame_slider.setMaximum(self.playback_len-1)
                self.StartPlayback()
                self.radio_raw.blockSignals(False)
                self.radio_spatial.blockSignals(False)

                # Display start/stop button and slider
                self.play_pause_button.show()
                self.frame_slider.show()
                self.cmap_button.show()
                self.profile_max_avg_label.setText("average max: ") # Reset average

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
        """
        Sets the current frame to slider value and resets average
        """
        self.reset_average()
        self.playback_index = self.frame_slider.value()
        self.read_from_loaded_vid()

    def read_from_loaded_vid(self):
        """
        Update the displayed image by getting the next frame
        """
        if self.playback_index >= self.playback_len-1:
            # At the end of the video
            self.PausePlayback()
        else:
            self.playback_index += 1

        frame = self.loaded_vid[self.playback_index-1,:,:]
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, -1)
        
        self.ImageUpdateSlot(frame, processed=False)
        self.update_frame_slider()

    def update_frame_slider(self):
        """
        Update the frame slider to the current index
        """
        # We need to block signals when we change the value to avoid a feedback loop
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.playback_index)
        self.frame_slider.blockSignals(False)

    def ToggleAcquisition(self, force_off=False):
        """
        Toggle Live Display Mode
        """
        print(force_off)
        if self.StartButton.isChecked() and not force_off:
            # Start the acquisition
            self.StartButton.setText("Stop")

            # Exit Playback Mode
            if self.playback_running:
                self.PausePlayback()
            self.play_pause_button.hide()
            self.frame_slider.hide()
            self.cmap_button.hide()
            self.profile_max_avg_label.setText("average max: ")
            self.auto_window_level = True
            
            # Disable radio buttons
            for button in self.button_group.buttons():
                button.setEnabled(False)

            if self.radio_spatial.isChecked():
                self.create_shared_mems()
            
            # Start worker to handle video feed
            self.CameraWorker = CameraWorker(self.saving_queue, self.recording, self.radio_raw.isChecked())
            self.CameraWorker.start()
            self.CameraWorker.CameraError.connect(self.camera_error)

            if self.radio_raw.isChecked():
                self.CameraWorker.ImageUpdate.connect(self.ImageUpdateSlot)
                self.CameraWorker.FirstFrame.connect(lambda cmap: self.SetColorMap(cmap))

            elif self.radio_spatial.isChecked():
                self.SetupLiveProcessing()

            self.RecordButton.setEnabled(True)

        else:
            if force_off:
                self.StartButton.setChecked(False)

            # Stop the acquisition
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

    def SetupLiveProcessing(self):
        """
        Start the Spatial Woker for real time processing, as well as the worker responsible
        for updating the displayed frame
        """
        self.stop_live_processing = multiprocessing.Event()
        self.pause_live_processing = multiprocessing.Event()
        self.live_process = multiprocessing.Process(target=spatial_worker, args=(self.temp_window.value(),
                                                                                    self.signal_queue,
                                                                                    self.stop_live_processing))
        self.live_process.start()

        self.DisplayProcessed = DisplayProcessed(self.signal_queue)
        self.DisplayProcessed.start()
        self.DisplayProcessed.ImageUpdate.connect(lambda image: self.ImageUpdateSlot(image, processed=True))
        self.DisplayProcessed.FirstFrame.connect(lambda cmap: self.SetColorMap(cmap))
        self.live_processing = True
    
    def ToggleRecord(self):
        """
        Start or stop recording
        """
        if  self.RecordButton.isChecked():
            # Start Saving
            self.RecordButton.setText("Stop Record")
            self.StartButton.setEnabled(False)

            self.recording.set()
            self.save_thread = SaveThread(self.output_dir/'raw_frames.raw', self.saving_queue)
            self.save_thread.finished.connect(self.save_raw_as_vid)
            self.save_thread.start()

        else:
            # Stop Saving
            self.RecordButton.setText("Start Record")            
            self.save_thread.requestInterruption()
            self.recording.clear()
            self.StartButton.setEnabled(True)
            self.reset_average()


    def ImageUpdateSlot(self, image, processed=False):
        """
        Update the displayed frame
        """
        # If we're showing processed frames (processed=True), 
        # then we want to make sure that self.live_processing hasnt been 
        # set to false to avoid acessing the shared memory that has 
        # been discarded
        if processed:
            if not self.live_processing:
                return
        self.data = image
        self.img.setImage(image, autoLevels=self.auto_window_level)
        if self.auto_window_level:
            self.colorbar.setLevels(self.img.quickMinMax())
        self.view.autoRange()

        self.updatePlot()


    def save_raw_as_vid(self, raw_path):
        """
        Called once the recording has stopped and every raw frame has been saved.
        A SaveAsVideo thread is started to re-save the video in avi format
        And a post-process Process is started if spatial or temporal was checked
        """
        print("Saving finished, starting another thread...")
        # Save .raw vid as .avi
        thread = SaveAsVideo(raw_path)
        self.save_as_vid_threads.append(thread)
        thread.finished.connect(lambda: self._cleanup_thread(thread))
        thread.start()

        if self.SpatialBox.isChecked() or self.TemporalBox.isChecked():
            # Start post-processing script
            params = {'output_dir':None,
                      'videos':[raw_path],
                      'temporal':self.TemporalBox.isChecked(),
                      'temporal_window':self.temp_window.value(),
                      'spatial':self.SpatialBox.isChecked(),
                      'spatial_window':(self.spatial_window1.value(), self.spatial_window2.value()),
                      'profile':self.Profile_CheckBox.isEnabled() and self.Profile_CheckBox.isChecked()}
            process = multiprocessing.Process(target=process_videos.main, kwargs=params)
            process.start()

            # Add new row in processing table
            process_id = len(self.processes) + 1
            self.processes[process_id] = process
            row = self.process_table.rowCount()
            self.process_table.insertRow(row)
            self.process_table.setRowHeight(row, 25)
            self.process_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{process_id}"))
            self.process_table.setItem(row, 1, QtWidgets.QTableWidgetItem("Processing..."))

            # Start a thread to monitor the post-processing Process and update the table once it's finished
            monitor_thread = ProcessMonitor(process_id, process)
            monitor_thread.finished_signal.connect(lambda pid= process_id, out=self.output_dir : self.update_process_status(pid, out))
            monitor_thread.start()
            self.monitor_threads[process_id] = monitor_thread


    def update_process_status(self, process_id, directory:Path):
        """
        When a process has finished, updates its status in the table
        """
        open_button = QtWidgets.QPushButton('')
        open_button.clicked.connect(lambda: self.LoadFile(directory))
        open_button.setIcon(self.open_file_icon)

        for row in range(self.process_table.rowCount()):
            item = self.process_table.item(row, 0)
            if item and int(item.text()) == process_id:
                self.process_table.setItem(row, 1, QtWidgets.QTableWidgetItem("Finished!"))
                self.process_table.setCellWidget(row, 2, open_button)
                break


    def _cleanup_thread(self, thread):
        """
        When a saving thread is done, it is removed from the list of current threads
        """
        self.save_as_vid_threads.remove(thread)


if __name__ == "__main__":
    # create pyqt5 app
    App = MyApp(sys.argv) 

    # create the instance of our Window
    window = Window()

    App.set_main_window(window)

    # start the app
    sys.exit(App.exec_())

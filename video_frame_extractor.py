import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QSpinBox, QDoubleSpinBox, QGroupBox, QMessageBox,
                             QProgressBar, QLineEdit, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import cv2
import numpy as np
from datetime import timedelta


class RectangleSelector(QLabel):
    """Widget for selecting rectangular region on video frame"""
    
    roi_selected = pyqtSignal(int, int, int, int)  # x, y, width, height
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.roi = None
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            
    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_point = event.pos()
            self.update()
            self._update_roi()
            
    def _update_roi(self):
        if self.start_point and self.end_point:
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            
            x = min(x1, x2)
            y = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Get parent widget size for validation
            parent_size = self.parent().size() if self.parent() else self.size()
            
            # Clamp to widget bounds
            x = max(0, min(x, parent_size.width() - 1))
            y = max(0, min(y, parent_size.height() - 1))
            width = min(width, parent_size.width() - x)
            height = min(height, parent_size.height() - y)
            
            if width > 0 and height > 0:
                self.roi = (x, y, width, height)
                self.roi_selected.emit(x, y, width, height)
                
    def paintEvent(self, event):
        super().paintEvent(event)  # Draw the pixmap first
        if self.start_point and self.end_point:
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
            painter.setPen(pen)
            
            x = min(self.start_point.x(), self.end_point.x())
            y = min(self.start_point.y(), self.end_point.y())
            width = abs(self.end_point.x() - self.start_point.x())
            height = abs(self.end_point.y() - self.start_point.y())
            
            painter.drawRect(x, y, width, height)
            painter.end()
            
    def clear_selection(self):
        self.start_point = None
        self.end_point = None
        self.roi = None
        self.update()


class FrameExtractorThread(QThread):
    """Background thread for extracting frames"""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, video_path, output_dir, interval_seconds, frames_per_second, roi=None):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.interval_seconds = interval_seconds
        self.frames_per_second = frames_per_second
        self.roi = roi  # (x, y, width, height) in pixel coordinates
        
    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"Failed to open video: {self.video_path}")
                return
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # ROI is already in video coordinates (converted in on_roi_selected)
            if self.roi:
                roi_x, roi_y, roi_w, roi_h = self.roi
                # Ensure ROI is within video bounds
                roi_x = max(0, min(roi_x, video_width - 1))
                roi_y = max(0, min(roi_y, video_height - 1))
                roi_w = min(roi_w, video_width - roi_x)
                roi_h = min(roi_h, video_height - roi_y)
            else:
                roi_x, roi_y, roi_w, roi_h = 0, 0, video_width, video_height
            
            frame_interval = self.interval_seconds if self.interval_seconds > 0 else video_duration
            current_time = 0.0
            frame_count = 0
            
            os.makedirs(self.output_dir, exist_ok=True)
            
            while current_time < video_duration:
                # Calculate frame numbers for this interval
                start_frame = int(current_time * fps)
                end_frame = int(min((current_time + frame_interval) * fps, total_frames))
                
                # Extract frames within this interval
                frames_in_interval = min(self.frames_per_second * frame_interval, end_frame - start_frame)
                if frames_in_interval > 0:
                    frame_indices = np.linspace(start_frame, end_frame - 1, int(frames_in_interval), dtype=int)
                else:
                    frame_indices = [start_frame] if start_frame < total_frames else []
                
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Apply ROI if specified
                    if self.roi and roi_w > 0 and roi_h > 0:
                        frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    
                    if frame.size > 0:
                        # Save frame
                        timestamp = current_time
                        frame_filename = f"frame_{frame_count:06d}_t{timestamp:.2f}s.jpg"
                        frame_path = os.path.join(self.output_dir, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        frame_count += 1
                        
                        # Emit progress
                        progress = int((current_time / video_duration) * 100) if video_duration > 0 else 0
                        self.progress.emit(min(progress, 100))
                
                current_time += frame_interval
                if frame_interval <= 0:
                    break
                    
            cap.release()
            self.finished.emit(frame_count)
            
        except Exception as e:
            self.error.emit(f"Error extracting frames: {str(e)}")


class VideoFrameExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.video_fps = 0
        self.video_duration = 0
        self.display_scale = 1.0
        self.roi = None
        
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def init_ui(self):
        self.setWindowTitle("Video Frame Extractor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Video display
        left_panel = QVBoxLayout()
        
        # Video display area with rectangle selector
        self.video_label = RectangleSelector()
        self.video_label.setText("No video loaded")
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.roi_selected.connect(self.on_roi_selected)
        
        left_panel.addWidget(self.video_label)
        
        # Video controls
        video_controls = QHBoxLayout()
        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)
        self.clear_roi_btn = QPushButton("Clear Selection")
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        self.clear_roi_btn.setEnabled(False)
        
        video_controls.addWidget(self.load_btn)
        video_controls.addWidget(self.clear_roi_btn)
        video_controls.addStretch()
        
        left_panel.addLayout(video_controls)
        
        # Right panel - Configuration
        right_panel = QVBoxLayout()
        
        # Video info
        info_group = QGroupBox("Video Information")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("No video loaded")
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        right_panel.addWidget(info_group)
        
        # Extraction settings
        settings_group = QGroupBox("Extraction Settings")
        settings_layout = QVBoxLayout()
        
        # Interval in seconds
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Interval (seconds):"))
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setMinimum(1.0)
        self.interval_spin.setMaximum(3600.0)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setDecimals(2)
        interval_layout.addWidget(self.interval_spin)
        settings_layout.addLayout(interval_layout)
        
        # Frames per second
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frames per second:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setMinimum(1)
        self.fps_spin.setMaximum(60)
        self.fps_spin.setValue(1)
        fps_layout.addWidget(self.fps_spin)
        settings_layout.addLayout(fps_layout)
        
        settings_group.setLayout(settings_layout)
        right_panel.addWidget(settings_group)
        
        # Output directory
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText("extracted_frames")
        self.output_dir_btn = QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_btn)
        output_layout.addLayout(output_dir_layout)
        
        output_group.setLayout(output_layout)
        right_panel.addWidget(output_group)
        
        # Extract button
        self.extract_btn = QPushButton("Extract Frames")
        self.extract_btn.clicked.connect(self.extract_frames)
        self.extract_btn.setEnabled(False)
        self.extract_btn.setMinimumHeight(40)
        right_panel.addWidget(self.extract_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        right_panel.addWidget(self.progress_bar)
        
        right_panel.addStretch()
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open video file")
                return
            
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration = total_frames / self.video_fps if self.video_fps > 0 else 0
            
            video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Update info label
            duration_str = str(timedelta(seconds=int(self.video_duration)))
            self.info_label.setText(
                f"File: {os.path.basename(file_path)}\n"
                f"Resolution: {video_width}x{video_height}\n"
                f"FPS: {self.video_fps:.2f}\n"
                f"Duration: {duration_str}\n"
                f"Total Frames: {total_frames}"
            )
            
            # Update interval maximum
            self.interval_spin.setMaximum(self.video_duration)
            
            # Display first frame
            self.update_frame()
            self.extract_btn.setEnabled(True)
            
    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
    def display_frame(self, frame):
        """Display frame in the video label"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)
        
    def on_roi_selected(self, x, y, width, height):
        """Handle ROI selection"""
        # Convert display coordinates to video coordinates
        if self.current_frame is not None and self.video_label.pixmap():
            pixmap = self.video_label.pixmap()
            label_size = self.video_label.size()
            
            # Get actual displayed pixmap rect
            pixmap_size = pixmap.size()
            x_offset = (label_size.width() - pixmap_size.width()) // 2
            y_offset = (label_size.height() - pixmap_size.height()) // 2
            
            # Adjust coordinates relative to pixmap
            x_adj = x - x_offset
            y_adj = y - y_offset
            
            if x_adj < 0 or y_adj < 0 or x_adj >= pixmap_size.width() or y_adj >= pixmap_size.height():
                return
            
            # Calculate scale factors
            video_height, video_width = self.current_frame.shape[:2]
            scale_x = video_width / pixmap_size.width() if pixmap_size.width() > 0 else 1.0
            scale_y = video_height / pixmap_size.height() if pixmap_size.height() > 0 else 1.0
            
            # Convert to video coordinates
            roi_x = int(x_adj * scale_x)
            roi_y = int(y_adj * scale_y)
            roi_w = int(width * scale_x)
            roi_h = int(height * scale_y)
            
            # Clamp to video bounds
            roi_x = max(0, min(roi_x, video_width - 1))
            roi_y = max(0, min(roi_y, video_height - 1))
            roi_w = min(roi_w, video_width - roi_x)
            roi_h = min(roi_h, video_height - roi_y)
            
            self.roi = (roi_x, roi_y, roi_w, roi_h)
            self.clear_roi_btn.setEnabled(True)
            
    def clear_roi(self):
        self.roi = None
        self.video_label.clear_selection()
        self.clear_roi_btn.setEnabled(False)
        
    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)
            
    def extract_frames(self):
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please load a video first")
            return
            
        interval_seconds = self.interval_spin.value()
        frames_per_second = self.fps_spin.value()
        output_dir = self.output_dir_edit.text()
        
        if not output_dir:
            QMessageBox.warning(self, "Warning", "Please specify an output directory")
            return
        
        # Disable extract button during extraction
        self.extract_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Create extraction thread
        self.extractor_thread = FrameExtractorThread(
            self.video_path, 
            output_dir, 
            interval_seconds, 
            frames_per_second,
            self.roi
        )
        self.extractor_thread.progress.connect(self.progress_bar.setValue)
        self.extractor_thread.finished.connect(self.on_extraction_finished)
        self.extractor_thread.error.connect(self.on_extraction_error)
        self.extractor_thread.start()
        
    def on_extraction_finished(self, frame_count):
        self.progress_bar.setValue(100)
        self.extract_btn.setEnabled(True)
        QMessageBox.information(
            self, 
            "Success", 
            f"Successfully extracted {frame_count} frames!"
        )
        
    def on_extraction_error(self, error_message):
        self.extract_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", error_message)
        
    def closeEvent(self, event):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'extractor_thread') and self.extractor_thread.isRunning():
            self.extractor_thread.terminate()
            self.extractor_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = VideoFrameExtractor()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

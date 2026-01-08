# Video Frame Extractor

A PyQt5-based application for extracting frames from video files with configurable intervals, frames per second, and region of interest (ROI) selection.

## Features

- Load video files (MP4, AVI, MOV, MKV, FLV, WMV)
- Extract frames at configurable intervals (in seconds, from 1 to full video length)
- Select rectangular area (ROI) on the video to crop frames
- Configure how many frames to extract per second
- Save extracted frames as JPEG images

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python video_frame_extractor.py
```

2. **Load Video**: Click "Load Video" button and select a video file.

3. **Configure Extraction Settings**:
   - **Interval (seconds)**: Set the time interval for frame extraction (default: 1.0 second). Range: 1.0 to video duration.
   - **Frames per second**: Set how many frames to extract per second within each interval (default: 1 frame/second).

4. **Select ROI (Optional)**:
   - Click and drag on the video display to select a rectangular region.
   - Only the selected area will be saved in extracted frames.
   - Click "Clear Selection" to remove ROI selection.

5. **Set Output Directory**: 
   - Enter the output directory path or click "Browse..." to select a folder.
   - Default: "extracted_frames"

6. **Extract Frames**: Click "Extract Frames" button to start the extraction process.
   - Progress will be shown in the progress bar.
   - Extracted frames will be saved with filenames like: `frame_000001_t0.00s.jpg`

## How It Works

- The application extracts frames at specified intervals throughout the video.
- Within each interval, it extracts the configured number of frames per second.
- If ROI is selected, only the selected rectangular area is saved in each frame.
- Frame extraction runs in a background thread to keep the UI responsive.

## Example

- Video duration: 60 seconds
- Interval: 5 seconds
- Frames per second: 2
- Result: Extracts 2 frames every 5 seconds = 24 frames total (2 frames × 12 intervals)

# Golf Swing Analysis System

AI-powered golf swing analyzer using pose estimation technology to help golfers improve their swing technique through detailed biomechanical analysis.

## Overview

This project provides automated analysis of golf swing videos using advanced pose estimation models. The system tracks 33 body keypoints, segments swing phases, and compares movements against established golf technique standards to provide actionable feedback for improvement.

## Features

- **Real-time Pose Estimation**: Uses MediaPipe for high-fidelity tracking of 33 body keypoints
- **Automatic Swing Segmentation**: Identifies and analyzes key swing phases (address, backswing, contact, follow-through)
- **Angle Analysis**: Calculates critical body angles and positions throughout the swing
- **Comparative Analysis**: Compares swing mechanics against professional golf standards
- **Visual Feedback**: Generates color-coded analysis videos with technical annotations
- **Web Interface**: Flask-based web server for easy video upload and analysis
- **Professional Validation**: Tested with 173 professional golf swing videos

## Technology Stack

- **Python 3.x**: Core programming language
- **MediaPipe**: Primary pose estimation model for body tracking
- **OpenCV**: Video processing and computer vision operations
- **Flask**: Web framework for API and file handling
- **NumPy & Pandas**: Data processing and numerical computations
- **Matplotlib**: Visualization and plotting
- **APScheduler**: Background task scheduling for file cleanup

## Project Structure

```
golf-analysis/
├── server.py              # Flask web server and API endpoints
├── MediaPipe_class.py      # MediaPipe pose estimation implementation
├── process_swing.py        # Swing analysis and data processing logic
├── run_swing_analyer.py    # Main analysis orchestration script
├── uploads/               # Directory for uploaded video files
├── processed/             # Directory for processed analysis results
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Dependencies

Install required packages:

```bash
pip install flask flask-cors opencv-python mediapipe numpy pandas matplotlib apscheduler
```

### Alternative Installation

If you have a requirements.txt file:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

1. Run the Flask server:
```bash
python server.py
```

2. The server will start on `http://localhost:5000` by default

### API Endpoints

#### Upload Video for Analysis
```http
POST /analyze
Content-Type: multipart/form-data

Parameters:
- file: Video file (MP4, MOV, AVI formats supported)
```

#### Check Analysis Status
```http
GET /status/{file_id}
```

#### Download Analysis Results
```http
GET /download/{file_id}
```

### Supported Video Formats

- MP4
- MOV
- AVI

### Analysis Process

1. **Video Upload**: Submit golf swing video through the web interface
2. **Pose Detection**: MediaPipe extracts 33 body keypoints frame-by-frame
3. **Swing Segmentation**: Algorithm identifies key swing phases:
   - Address position
   - Top of backswing
   - Ball contact
   - Follow-through
4. **Angle Calculation**: Computes critical body angles and positions
5. **Analysis Generation**: Creates detailed report with visual feedback
6. **Result Download**: Processed video and analysis data available for download

## Configuration

### File Management

- **Upload Directory**: `uploads/` (automatically created)
- **Processing Directory**: `processed/` (automatically created)
- **File Cleanup**: Automatic cleanup of old files after 24 hours

### Server Configuration

Modify `server.py` to adjust:
- Port number (default: 5000)
- File size limits
- Cleanup intervals
- CORS settings

## Technical Details

### Pose Estimation

The system uses MediaPipe's pose estimation model which provides:
- 33 body landmarks with X, Y, Z coordinates
- Real-time processing capabilities
- High accuracy for sports motion analysis

### Swing Analysis Methodology

1. **Keypoint Tracking**: Continuous tracking of body positions
2. **Phase Detection**: Identifies swing transitions using wrist trajectory analysis
3. **Angle Computation**: Calculates joint angles using three-point geometry
4. **Comparison Analysis**: Evaluates positions against optimal swing mechanics
5. **Feedback Generation**: Creates visual and numerical feedback

### Performance Considerations

- **Video Quality**: Higher resolution videos provide better pose detection accuracy
- **Lighting**: Well-lit environments improve tracking reliability
- **Camera Angle**: Side-view angles work best for swing analysis
- **Background**: Minimal background clutter improves pose detection

## Development

### Running in Development Mode

```bash
# Enable Flask debug mode
export FLASK_ENV=development
python server.py
```

### Code Structure

- **MediaPipe_class.py**: Handles pose estimation and angle calculations
- **process_swing.py**: Contains data processing and swing segmentation logic
- **server.py**: Web server, file handling, and API endpoints
- **run_swing_analyer.py**: Coordinates the analysis workflow

## Limitations

- Works best with controlled practice session recordings
- Optimal performance requires side-view camera angles
- Processing time varies with video length and quality
- Current model optimized for standard golf swing techniques

## Future Enhancements

- **Multi-Sport Support**: Adaptation for tennis, baseball, and other sports
- **Real-time Analysis**: Live swing feedback during practice
- **Mobile Application**: Dedicated mobile app for field use
- **Advanced Analytics**: Machine learning-based personalized recommendations
- **3D Analysis**: Enhanced depth perception and spatial analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is based on the original work from [23206-final-pose-estimation-for-swing-improvement](https://github.com/Strojove-uceni/23206-final-pose-estimation-for-swing-improvement).

## Support

For issues, questions, or feature requests, please create an issue in the repository.

## Acknowledgments

- Original research project: 23206 Final Pose Estimation for Swing Improvement
- MediaPipe team for pose estimation technology
- Golf professionals who provided swing analysis validation
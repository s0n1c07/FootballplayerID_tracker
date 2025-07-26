# Advanced RAFT-Enhanced Multi-Object Tracking

A sophisticated multi-object tracking system that combines YOLOv11 object detection with RAFT optical flow estimation and ResNet18-based re-identification for robust tracking of football game entities.

## Features

- **YOLOv11 Detection**: State-of-the-art object detection for football, goalkeeper, player, and referee detection
- **RAFT Optical Flow**: Advanced motion estimation using RAFT (Recurrent All-Pairs Field Transforms) for improved tracking accuracy
- **ResNet18 Re-ID**: Appearance-based re-identification to maintain track consistency across occlusions
- **Kalman Filtering**: Predictive motion modeling for smooth trajectory estimation
- **Hungarian Algorithm**: Optimal assignment for track-detection association
- **Multi-Modal Tracking**: Combines IoU, optical flow, and appearance features for robust tracking

## Supported Object Classes

The system is specifically designed for football match analysis and can track:

- **Football** (Ball)
- **Goalkeeper**
- **Player**
- **Referee**

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- Minimum 8GB GPU memory for RAFT large model
- CPU fallback available but significantly slower

### Software
See `requirements.txt` for complete dependencies:

```
torch
ultralytics
numpy
raft
scipy
opencv-python
pandas
torchvision
pillow
torchreid
gdown
tensorboard
deep_sort_realtime
setuptools
wheel
cython
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/s0n1c07/FootballplayerID_tracker
cd FootballplayerID_tracker
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Git LFS (for model weights)**
```bash
git lfs install
git lfs pull
```

## Usage

### Basic Usage

1. **Prepare your input video**
   - Place your input video as `15sec_input_720p.mp4` in the project directory
   - Or modify the `VIDEO_PATH` variable in `main.py`

2. **Ensure model weights are available**
   - The YOLOv11 model weights should be in `best.pt`
   - This file is managed by Git LFS

3. **Run the tracking system**
```bash
python main.py
```

4. **Output**
   - Processed video will be saved as `output_raft_advanced_reid_tracking.mp4`

### Configuration

Key parameters that can be modified in `main.py`:

```python
# File paths
MODEL_PATH = 'best.pt'                    # YOLOv11 model weights
VIDEO_PATH = '15sec_input_720p.mp4'       # Input video
OUTPUT_VIDEO_PATH = 'output_raft_advanced_reid_tracking.mp4'  # Output video

# Detection parameters
conf_threshold = 0.4                      # Detection confidence threshold
iou_threshold = 0.5                       # NMS IoU threshold

# Tracking parameters
max_age = 90                              # Maximum frames to keep lost tracks
iou_match_threshold = 0.5                 # IoU threshold for track matching
reid_distance_threshold = 0.5             # Re-ID cosine distance threshold
```

## System Architecture

### 1. Detection Pipeline
- **YOLOv11**: Performs object detection on each frame
- **Confidence Filtering**: Filters detections based on confidence threshold
- **Class Filtering**: Only processes specified target classes

### 2. Tracking Pipeline
- **Kalman Filter Prediction**: Predicts object positions based on motion model
- **RAFT Flow Enhancement**: Uses optical flow to improve position predictions
- **Association**: Uses Hungarian algorithm with IoU matrix for primary matching
- **Re-ID Fallback**: Uses ResNet18 embeddings for appearance-based matching of unmatched detections

### 3. Track Management
- **Track Initialization**: Creates new tracks for unmatched detections
- **Track Update**: Updates existing tracks with new detections
- **Track Termination**: Removes tracks that exceed maximum age without updates

## Model Details

### YOLOv11 Detection Model
- Custom trained model for football match analysis
- Detects: ball, goalkeeper, player, referee
- Input resolution: 640×640
- Stored as `best.pt` using Git LFS

### RAFT Optical Flow
- **Primary**: RAFT Large model (better accuracy)
- **Fallback**: RAFT Small model (lower memory requirements)
- Provides dense optical flow for motion prediction

### ResNet18 Re-ID
- **Architecture**: ResNet18 with final classification layer removed
- **Input Size**: 256×128 (standard person re-identification size)
- **Output**: 512-dimensional feature vectors
- **Distance Metric**: Cosine distance for similarity comparison

## Performance Considerations

### GPU Memory Usage
- **RAFT Large**: ~6-8GB GPU memory
- **RAFT Small**: ~3-4GB GPU memory
- **ResNet18**: ~1-2GB GPU memory
- **YOLOv11**: ~2-3GB GPU memory

### Speed Optimization
- Model inference is batched where possible
- Optical flow computation is the main bottleneck
- Consider reducing input resolution for faster processing

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Solution: The system automatically falls back to RAFT small model
   - Alternative: Reduce input video resolution

2. **Git LFS Issues**
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

4. **Video Codec Issues**
   - Ensure OpenCV is compiled with proper codec support
   - Try converting input video to standard MP4 format

## File Structure

```
├── .gitattributes          # Git LFS configuration
├── .gitignore             # Git ignore rules
├── best.pt                # YOLOv11 model weights (Git LFS)
├── main.py                # Main tracking application
├── requirements.txt       # Python dependencies
├── 15sec_input_720p.mp4   # Input video (not included)
└── output_raft_advanced_reid_tracking.mp4  # Output video (generated)
```

## Technical Details

### Tracking Algorithm Flow
1. **Frame Processing**: Convert frame to RGB and create tensor
2. **Detection**: Run YOLOv11 inference
3. **Prediction**: Use Kalman filters + optical flow for track prediction
4. **Embedding**: Extract ResNet18 features for each detection
5. **Association**: 
   - Primary: IoU-based Hungarian assignment
   - Secondary: Re-ID based Hungarian assignment for unmatched objects
6. **Update**: Update matched tracks, create new tracks, remove old tracks
7. **Visualization**: Draw bounding boxes and track IDs

### Color Coding
- **Football**: Orange (0, 165, 255)
- **Goalkeeper**: Yellow (0, 200, 200)
- **Player**: Green (0, 255, 0)
- **Referee**: Red (255, 0, 0)

## Future Improvements

- [ ] Support for additional sports and object classes
- [ ] Real-time processing optimization
- [ ] Multi-camera fusion
- [ ] Advanced trajectory analysis
- [ ] Export tracking data to JSON/CSV formats
- [ ] Integration with sports analytics platforms

## License

This project is provided as-is for research and educational purposes. Please ensure you have appropriate licenses for all dependencies and model weights.

## Contributing

Contributions are welcome! Please ensure all code follows the existing style and includes appropriate documentation.

## Acknowledgments

- YOLOv11 by Ultralytics
- RAFT optical flow by Princeton Vision & Learning Lab
- ResNet architecture by Microsoft Research
- OpenCV computer vision library

# OpenCV: Real-Time Object Detection & Shape/Color Tracking

This project provides various Python scripts for live object detection and shape/color tracking using a webcam. More may be added that I find interesting and worthwhile to use.

1. **`live_dnn_detection.py`** – Uses a pre-trained DNN for object detection.  
2. **`webcam.py`** – Custom multi-color shape detection with dictionary-based labeling and HSV tuning.  

---

## Features

### live_dnn_detection.py
- Detects objects using a pre-trained deep neural network (DNN).  
- Provides bounding boxes and labels for standard objects.  
- Real-time confidence on webcam input.  

### webcam.py
- **Multi-color detection**: Red, Blue, Green, Yellow, Orange, Purple.  
- **Shape recognition**: Triangles, rectangles, circles, pentagons, hexagons, and generic polygons.  
- **Intelligent labeling**: Combines color and shape for descriptive labels (e.g., "Blue Triangle").  
- **Real-time HSV tuning**: Adjust color detection ranges via sliders.  
- **Visual feedback**: Shows bounding boxes, contours, centers, masks, and labeled objects.  
- **Keyboard controls**:  
  - `1-6` → Toggle detection for specific colors  
  - `R, B, G, Y, O, P` → Select color for HSV tuning  
  - `Q` → Quit application  

---

## Requirements

- Python 3.x  
- OpenCV (`cv2`)  
- Numpy  

Install dependencies with:

```bash
pip install opencv-python numpy

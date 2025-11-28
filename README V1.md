**Project Overview**
- An enhanced real-time object detection and tracking demo built on YOLOv8 (Ultralytics) with advanced visualization, Kalman filtering, clustering, zone alerts, recording and analytics.
- **Purpose**: Provide a robust, extendable playground for experimenting with real-time detection, object tracking, and monitoring (camera or video file input).


**Key Features**
- **Real-time detection**: Uses `ultralytics.YOLO` for inference with configurable model (e.g., `yolov8n.pt`, `yolov8s.pt`, ...).
- **Enhanced tracking**: Custom tracker combining centroid association, IoU, area consistency and optional Kalman filtering.
- **Visuals & HUD**: Draws bounding boxes, motion trails, velocity arrows, multi-line labels and an on-frame HUD showing FPS, inference time and system stats.
- **Detection zones & alerts**: Add polygon zones and optional alerting on entry.
- **Recording & snapshot**: Start/stop video recording and save frames from keyboard controls.
- **Config management**: Save/load configuration via JSON using `--save-config` / `--load-config`.
- **Robustness**: Graceful fallbacks when optional dependencies (e.g. `scipy`, `filterpy`) are missing.

**Quick Setup (Windows / PowerShell)**
1. Install Python 3.8+ (recommended 3.10/3.11). If you need GPU acceleration, install a matching `torch` with CUDA support.

2. Install required packages (system/global install as you requested):

```powershell
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python-headless opencv-python numpy torch torchvision scikit-learn scipy filterpy
```

Notes:
- If you want CUDA support for `torch`, follow the official install instructions at https://pytorch.org (choose the correct CUDA version). Example (CPU-only wheels shown above).
- `opencv-python-headless` can be used on headless systems; if you need GUI (`cv2.imshow`) keep `opencv-python`.

3. Ensure a YOLO weight file exists. The project expects `yolov8n.pt` by default in the working directory. You can download official Ultralytics models or point `--model` to a different path.

**Run The App**
- Run with the default webcam (camera `0`) and default model:

```powershell
python "Object_Detection_V1.py" --model "yolov8n.pt" --device cpu --camera 0 --width 1280 --height 720
```

- Run using GPU (if `torch` with CUDA is installed):

```powershell
python "Object_Detection_V1.py" --model "yolov8n.pt" --device cuda --camera 0
```

- Run on a video file instead of the webcam:

```powershell
python "Object_Detection_V1.py" --model "yolov8n.pt" --camera "path\\to\\video.mp4"
```

- Save current runtime configuration to JSON:

```powershell
python "Object_Detection_V1.py" --save-config config.json
```

- Load configuration from JSON:

```powershell
python "Object_Detection_V1.py" --load-config config.json
```

**Keyboard Controls (while window focused)**
- `Q`: Quit
- `R`: Reset tracking
- `+` / `=`: Increase confidence threshold
- `-`: Decrease confidence threshold
- `S`: Save current frame to `detection_YYYYMMDD_HHMMSS.jpg`
- `V`: Start/Stop recording; file saved as `recording_YYYYMMDD_HHMMSS.mp4`
- `C`: Clear statistics
- `Z`: Add a centered detection zone (demo) - you can modify `_interactive_zone_creation` to use mouse callbacks
- `H`: Toggle HUD overlay
- `I`: Show detailed information panel in logs

**Configuration Options**
Important command-line flags (see `--help` for full list):
- `--model`: Path to YOLO model (default `yolov8n.pt`)
- `--confidence`: Confidence score threshold (default `0.5`)
- `--iou`: NMS IoU threshold (default `0.45`)
- `--device`: `cpu` or `cuda` (default `cpu`)
- `--imgsz`: Inference image size (default `640`)
- `--half`: Use FP16 (only works on CUDA)
- Tracking params: `--max-disappeared`, `--max-distance`, `--min-hits`, `--use-kalman`
- Camera params: `--camera` (ID or path), `--width`, `--height`

**Optional Dependencies & Recommendations**
- `scipy` (for Hungarian assignment via `scipy.optimize.linear_sum_assignment`) — improves association quality.
- `filterpy` (Kalman filter) — enables more robust motion smoothing and velocity estimation.
- `torch` with CUDA — provides GPU acceleration for much faster inference; install the matching CUDA wheel from the PyTorch site.

If these packages are not installed, the script logs warnings and uses fallbacks (greedy assignment, centroid-only tracking).

**Troubleshooting**
- "Cannot open camera": Verify camera index, ensure no other app is using the camera, and test with a simple `cv2.VideoCapture(0)` script.
- "Model file not found": Make sure `--model` points to a valid `.pt` file. Download from Ultralytics or point to a custom-trained model.
- Slow FPS / high latency: Try `--device cuda` with a properly installed CUDA `torch`, or use a smaller model like `yolov8n.pt`.
- cv2.imshow blank/black: Use `opencv-python` (not headless) on desktop systems to enable GUI windows.

**Extending & Tuning**
- Add mouse callbacks in `_interactive_zone_creation` to draw polygons interactively.
- Export per-frame detection logs (CSV or JSON) for offline analytics.
- Integrate a lightweight web UI (Streamlit or Flask) to view stats remotely.
- Swap the tracker with a learned ReID + SORT/DeepSORT implementation for better re-identification across occlusions.

**Files of Interest**
- `Object_Detection_V1.py` — main script (this file)
- `yolov8n.pt` — default model file (place in same folder or pass full path with `--model`)


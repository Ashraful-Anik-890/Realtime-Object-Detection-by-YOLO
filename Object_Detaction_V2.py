import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
from collections import defaultdict, deque
import threading
import queue
import math
import json
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.cluster import DBSCAN
import torch
import sys
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """Configuration class for detection parameters"""
    model_path: str = 'yolov8n.pt'
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = 'cpu'
    max_det: int = 300
    imgsz: int = 640
    half: bool = False  # Use FP16 inference

@dataclass
class TrackingConfig:
    """Configuration class for tracking parameters"""
    max_disappeared: int = 30
    max_distance: float = 100
    min_hits: int = 3
    max_age: int = 1
    use_kalman: bool = True

@dataclass
class ObjectInfo:
    """Enhanced object information structure"""
    object_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    velocity: Tuple[float, float]
    first_seen: float
    last_seen: float
    hit_streak: int
    time_since_update: int
    area: float
    aspect_ratio: float

class KalmanFilter:
    """Kalman filter for object tracking with constant velocity model"""

    def __init__(self):
        try:
            from filterpy.kalman import KalmanFilter as KF
            from filterpy.common import Q_discrete_white_noise

            # State: [x, y, vx, vy] - position and velocity
            self.kf = KF(dim_x=4, dim_z=2)
            dt = 1.0  # time step

            # Transition matrix (constant velocity model)
            self.kf.F = np.array([[1, 0, dt, 0],
                                  [0, 1, 0, dt],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]], dtype=np.float64)

            # Measurement matrix (we only observe position)
            self.kf.H = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0]], dtype=np.float64)

            # Measurement noise
            self.kf.R = np.eye(2, dtype=np.float64) * 10

            # Process noise
            self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1, block_size=2)

            # Initial covariance
            self.kf.P *= 100

            self.initialized = True

        except ImportError:
            logger.warning("filterpy not available, using simple tracking")
            self.initialized = False

    def predict(self):
        """Predict next state"""
        if not self.initialized:
            return None
        try:
            self.kf.predict()
            return self.kf.x[:2].copy()  # Return predicted position
        except Exception as e:
            logger.error(f"Kalman predict error: {e}")
            return None

    def update(self, measurement):
        """Update with measurement"""
        if not self.initialized:
            return
        try:
            # Ensure measurement is correct shape
            measurement = np.array(measurement, dtype=np.float64).reshape(-1, 1)
            if measurement.shape[0] != 2:
                logger.error(f"Invalid measurement shape: {measurement.shape}")
                return
            self.kf.update(measurement)
        except Exception as e:
            logger.error(f"Kalman update error: {e}")

    def get_state(self):
        """Get current state [x, y, vx, vy]"""
        if not self.initialized:
            return None
        return self.kf.x.copy()

class EnhancedObjectTracker:
    """Enhanced object tracker with Kalman filtering and deep association"""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.next_id = 0
        self.trackers = {}
        self.max_disappeared = config.max_disappeared
        self.max_distance = config.max_distance
        self.min_hits = config.min_hits
        self.max_age = config.max_age
        self.use_kalman = config.use_kalman

    def _create_tracker(self, detection: Dict) -> Dict:
        """Create a new tracker for an object"""
        centroid = detection['centroid']

        tracker = {
            'id': self.next_id,
            'class_name': detection['class_name'],
            'centroid': centroid,
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'velocity': (0.0, 0.0),
            'first_seen': time.time(),
            'last_seen': time.time(),
            'hit_streak': 1,
            'time_since_update': 0,
            'area': detection['area'],
            'aspect_ratio': detection['aspect_ratio'],
            'kalman': None
        }

        if self.use_kalman:
            try:
                kf = KalmanFilter()
                if kf.initialized:
                    # Initialize Kalman filter with current position
                    kf.kf.x = np.array([centroid[0], centroid[1], 0, 0], dtype=np.float64).reshape(-1, 1)
                    tracker['kalman'] = kf
                else:
                    logger.warning("Kalman filter not initialized, falling back to centroid tracking")
                    self.use_kalman = False
            except Exception as e:
                logger.warning(f"Kalman filter creation failed: {e}, falling back to centroid tracking")
                self.use_kalman = False

        self.next_id += 1
        return tracker

    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _compute_distance_matrix(self, trackers: List, detections: List) -> np.ndarray:
        """Compute distance matrix between trackers and detections"""
        if not trackers or not detections:
            return np.array([])

        distances = np.zeros((len(trackers), len(detections)))

        for i, tracker in enumerate(trackers):
            for j, detection in enumerate(detections):
                # Centroid distance
                t_cent = tracker['centroid']
                d_cent = detection['centroid']
                centroid_dist = math.sqrt((t_cent[0] - d_cent[0])**2 + (t_cent[1] - d_cent[1])**2)

                # IoU similarity (converted to distance)
                iou = self._compute_iou(tracker['bbox'], detection['bbox'])
                iou_dist = 1.0 - iou

                # Class consistency
                class_penalty = 0.0 if tracker['class_name'] == detection['class_name'] else 0.5

                # Area consistency
                area_ratio = min(tracker['area'], detection['area']) / max(tracker['area'], detection['area'])
                area_penalty = 1.0 - area_ratio

                # Combined distance
                distances[i, j] = centroid_dist * 0.4 + iou_dist * 100 + class_penalty * 200 + area_penalty * 50

        return distances

    def update(self, detections: List[Dict]) -> List[ObjectInfo]:
        """Update trackers with new detections"""
        # Predict step for Kalman filters
        for tracker in self.trackers.values():
            if tracker['kalman'] is not None:
                try:
                    predicted_pos = tracker['kalman'].predict()
                    if predicted_pos is not None:
                        tracker['centroid'] = (int(predicted_pos[0]), int(predicted_pos[1]))
                except Exception as e:
                    logger.error(f"Kalman predict failed: {e}")
                    # Disable Kalman for this tracker
                    tracker['kalman'] = None
            tracker['time_since_update'] += 1

        matched_indices = []
        unmatched_trackers = []
        unmatched_detections = list(range(len(detections)))

        if self.trackers and detections:
            tracker_list = list(self.trackers.values())
            distance_matrix = self._compute_distance_matrix(tracker_list, detections)

            if distance_matrix.size > 0:
                # Hungarian algorithm for optimal assignment
                try:
                    from scipy.optimize import linear_sum_assignment
                    row_indices, col_indices = linear_sum_assignment(distance_matrix)
                except ImportError:
                    logger.warning("scipy not available, using greedy assignment")
                    # Fallback to greedy assignment
                    row_indices, col_indices = self._greedy_assignment(distance_matrix)

                matched_indices = []
                for row, col in zip(row_indices, col_indices):
                    if distance_matrix[row, col] <= self.max_distance:
                        matched_indices.append((row, col))
                        unmatched_detections.remove(col)
                    else:
                        unmatched_trackers.append(row)

                unmatched_trackers.extend([i for i in range(len(tracker_list))
                                         if i not in [m[0] for m in matched_indices]])
        else:
            unmatched_trackers = list(range(len(self.trackers)))

        # Update matched trackers
        for tracker_idx, detection_idx in matched_indices:
            tracker_id = list(self.trackers.keys())[tracker_idx]
            tracker = self.trackers[tracker_id]
            detection = detections[detection_idx]

            # Update tracker with detection
            old_centroid = tracker['centroid']
            new_centroid = detection['centroid']

            # Calculate velocity
            dt = time.time() - tracker['last_seen']
            if dt > 0:
                vx = (new_centroid[0] - old_centroid[0]) / dt
                vy = (new_centroid[1] - old_centroid[1]) / dt
                tracker['velocity'] = (vx, vy)

            tracker['centroid'] = new_centroid
            tracker['bbox'] = detection['bbox']
            tracker['confidence'] = detection['confidence']
            tracker['last_seen'] = time.time()
            tracker['hit_streak'] += 1
            tracker['time_since_update'] = 0
            tracker['area'] = detection['area']
            tracker['aspect_ratio'] = detection['aspect_ratio']

            # Update Kalman filter
            if tracker['kalman'] is not None:
                try:
                    tracker['kalman'].update(new_centroid)
                except Exception as e:
                    logger.error(f"Kalman update failed: {e}")
                    # Fall back to simple tracking
                    tracker['kalman'] = None

        # Create new trackers for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            new_tracker = self._create_tracker(detection)
            self.trackers[new_tracker['id']] = new_tracker

        # Remove old trackers
        to_remove = []
        for tracker_id, tracker in self.trackers.items():
            if tracker['time_since_update'] > self.max_disappeared:
                to_remove.append(tracker_id)

        for tracker_id in to_remove:
            del self.trackers[tracker_id]

        # Return valid tracked objects
        valid_objects = []
        for tracker in self.trackers.values():
            if tracker['hit_streak'] >= self.min_hits or tracker['time_since_update'] == 0:
                obj_info = ObjectInfo(
                    object_id=tracker['id'],
                    class_name=tracker['class_name'],
                    confidence=tracker['confidence'],
                    bbox=tracker['bbox'],
                    centroid=tracker['centroid'],
                    velocity=tracker['velocity'],
                    first_seen=tracker['first_seen'],
                    last_seen=tracker['last_seen'],
                    hit_streak=tracker['hit_streak'],
                    time_since_update=tracker['time_since_update'],
                    area=tracker['area'],
                    aspect_ratio=tracker['aspect_ratio']
                )
                valid_objects.append(obj_info)

        return valid_objects

    def _greedy_assignment(self, distance_matrix):
        """Fallback greedy assignment when scipy is not available"""
        rows, cols = [], []
        used_rows, used_cols = set(), set()

        # Create list of (distance, row, col) and sort by distance
        assignments = []
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[1]):
                assignments.append((distance_matrix[i, j], i, j))

        assignments.sort()  # Sort by distance (lowest first)

        for dist, row, col in assignments:
            if row not in used_rows and col not in used_cols:
                rows.append(row)
                cols.append(col)
                used_rows.add(row)
                used_cols.add(col)

        return np.array(rows), np.array(cols)

class DetectionProcessor:
    """Advanced detection processing with filtering and enhancement"""

    def __init__(self):
        self.detection_history = deque(maxlen=10)
        self.confidence_smoother = {}

    def process_detections(self, results, frame_shape) -> List[Dict]:
        """Process YOLO results into structured detections"""
        detections = []
        boxes = results.boxes

        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]

            # Calculate additional features
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height if height > 0 else 1.0
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Filter out small or invalid detections
            if area < 100 or width < 10 or height < 10:
                continue

            # Boundary check
            if x1 < 0 or y1 < 0 or x2 >= frame_shape[1] or y2 >= frame_shape[0]:
                continue

            detection = {
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'class_name': class_name,
                'class_id': class_id,
                'centroid': centroid,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'width': width,
                'height': height
            }

            detections.append(detection)

        # Apply temporal smoothing
        detections = self._apply_temporal_smoothing(detections)

        return detections

    def _apply_temporal_smoothing(self, detections: List[Dict]) -> List[Dict]:
        """Apply temporal smoothing to reduce detection jitter"""
        smoothed_detections = []

        for detection in detections:
            key = f"{detection['class_name']}_{detection['centroid'][0]//50}_{detection['centroid'][1]//50}"

            if key in self.confidence_smoother:
                # Smooth confidence over time
                old_conf = self.confidence_smoother[key]
                new_conf = 0.7 * old_conf + 0.3 * detection['confidence']
                detection['confidence'] = new_conf
                self.confidence_smoother[key] = new_conf
            else:
                self.confidence_smoother[key] = detection['confidence']

            smoothed_detections.append(detection)

        return smoothed_detections

class AdvancedObjectDetector:
    """Advanced real-time object detector with enhanced features"""

    def __init__(self, detection_config: DetectionConfig, tracking_config: TrackingConfig):
        self.detection_config = detection_config
        self.tracking_config = tracking_config

        # Initialize model with optimizations
        self.model = self._initialize_model()

        # Initialize components
        self.tracker = EnhancedObjectTracker(tracking_config)
        self.processor = DetectionProcessor()

        # Performance tracking
        self.fps_history = deque(maxlen=60)
        self.frame_count = 0
        self.start_time = time.time()
        self.inference_times = deque(maxlen=30)

        # Detection analytics
        self.detection_analytics = {
            'total_detections': 0,
            'unique_objects': set(),
            'class_counts': defaultdict(int),
            'confidence_history': defaultdict(list),
            'zones': {}
        }

        # Visualization
        self.colors = self._generate_colors(100)
        self.trail_history = defaultdict(lambda: deque(maxlen=30))

        # Threading for async processing
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        self.running = False

        # Zones and alerts
        self.detection_zones = []
        self.alert_system = AlertSystem()

        # Recording
        self.is_recording = False
        self.video_writer = None

    def _initialize_model(self) -> YOLO:
        """Initialize YOLO model with optimizations"""
        model = YOLO(self.detection_config.model_path)

        # Set device
        if self.detection_config.device == 'cuda' and torch.cuda.is_available():
            model.to('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            model.to('cpu')
            logger.info("Using CPU")

        # Enable optimizations
        if self.detection_config.half and self.detection_config.device == 'cuda':
            model.half()  # Use FP16 for faster inference

        return model

    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for different classes"""
        np.random.seed(42)
        colors = []
        for i in range(num_classes):
            hue = (i * 137.508) % 360  # Golden angle for better distribution
            saturation = 90 + (i % 3) * 10
            value = 90 + (i % 4) * 10

            # Convert HSV to BGR
            # OpenCV's cvtColor for uint8 expects: Hue: [0, 179], Saturation: [0, 255], Value: [0, 255]
            h_cv = int(hue / 2)  # Scale hue from 0-359 to 0-179
            # Scale saturation and value (originally percentages 90-120) to 0-255 range and cap at 255
            s_cv = min(255, int(saturation * 2.55))
            v_cv = min(255, int(value * 2.55))

            hsv = np.uint8([[[h_cv, s_cv, v_cv]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr)))

        return colors

    def add_detection_zone(self, points: List[Tuple[int, int]], name: str = "Zone"):
        """Add a detection zone (polygon)"""
        zone = {
            'name': name,
            'points': np.array(points, dtype=np.int32),
            'active': True,
            'alert_on_entry': False,
            'objects_in_zone': set()
        }
        self.detection_zones.append(zone)

    def _process_frame_async(self):
        """Async frame processing worker"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)

                # Inference
                start_time = time.time()
                try:
                    results = self.model(
                        frame,
                        conf=self.detection_config.confidence_threshold,
                        iou=self.detection_config.iou_threshold,
                        imgsz=self.detection_config.imgsz,
                        max_det=self.detection_config.max_det,
                        half=self.detection_config.half,
                        device=self.detection_config.device,
                        verbose=False
                    )[0]
                except Exception as e:
                    logger.error(f"YOLO inference error: {e}")
                    continue

                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)

                # Process detections
                try:
                    detections = self.processor.process_detections(results, frame.shape)
                except Exception as e:
                    logger.error(f"Detection processing error: {e}")
                    detections = []

                # Update tracker
                try:
                    tracked_objects = self.tracker.update(detections)
                except Exception as e:
                    logger.error(f"Tracker update error: {e}")
                    tracked_objects = []

                self.result_queue.put({
                    'tracked_objects': tracked_objects,
                    'inference_time': inference_time,
                    'raw_results': results
                })

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Frame processing error: {e}")

    def _draw_enhanced_detections(self, frame: np.ndarray, tracked_objects: List[ObjectInfo]) -> np.ndarray:
        """Draw enhanced detection visualizations"""
        height, width = frame.shape[:2]

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            color = self.colors[hash(obj.class_name) % len(self.colors)]

            # Main bounding box
            thickness = 3 if obj.hit_streak > 5 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Centroid
            cv2.circle(frame, obj.centroid, 4, color, -1)

            # Motion trail
            self.trail_history[obj.object_id].append(obj.centroid)
            trail = list(self.trail_history[obj.object_id])
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    pt1, pt2 = trail[i-1], trail[i]
                    cv2.line(frame, pt1, pt2, color, max(1, int(3 * alpha)))

            # Velocity arrow
            if obj.velocity[0] != 0 or obj.velocity[1] != 0:
                vel_scale = 10
                end_point = (
                    int(obj.centroid[0] + obj.velocity[0] * vel_scale),
                    int(obj.centroid[1] + obj.velocity[1] * vel_scale)
                )
                cv2.arrowedLine(frame, obj.centroid, end_point, color, 2, tipLength=0.3)

            # Enhanced label
            speed = math.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
            age = time.time() - obj.first_seen

            label_lines = [
                f"ID:{obj.object_id} {obj.class_name}",
                f"Conf:{obj.confidence:.2f} Age:{age:.1f}s",
                f"Speed:{speed:.1f}px/s"
            ]

            # Multi-line label background
            label_height = 20 * len(label_lines) + 10
            cv2.rectangle(frame, (x1, y1 - label_height), (x1 + 200, y1), color, -1)
            cv2.rectangle(frame, (x1, y1 - label_height), (x1 + 200, y1), (255, 255, 255), 1)

            # Draw label text
            for i, line in enumerate(label_lines):
                y_offset = y1 - label_height + 15 + i * 20
                cv2.putText(frame, line, (x1 + 5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw detection zones
        for zone in self.detection_zones:
            if zone['active']:
                cv2.polylines(frame, [zone['points']], True, (0, 255, 255), 2)
                cv2.putText(frame, zone['name'], tuple(zone['points'][0]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def _draw_advanced_hud(self, frame: np.ndarray, tracked_objects: List[ObjectInfo],
                          inference_time: float) -> np.ndarray:
        """Draw advanced heads-up display"""
        height, width = frame.shape[:2]

        # Calculate metrics
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        self.fps_history.append(current_fps)

        avg_fps = sum(self.fps_history) / len(self.fps_history)
        avg_inference = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0

        # Update analytics
        active_classes = list(set([obj.class_name for obj in tracked_objects]))
        for obj in tracked_objects:
            self.detection_analytics['unique_objects'].add(f"{obj.class_name}_{obj.object_id}")
            self.detection_analytics['class_counts'][obj.class_name] += 1
            self.detection_analytics['confidence_history'][obj.class_name].append(obj.confidence)

        # HUD Panel
        hud_height = 120
        hud_y = height - hud_height

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, hud_y), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Border
        cv2.rectangle(frame, (0, hud_y), (width, height), (0, 255, 0), 2)

        # Performance metrics (left column)
        perf_metrics = [
            f"FPS: {avg_fps:.1f} (Target: 30)",
            f"Inference: {avg_inference*1000:.1f}ms",
            f"Active Objects: {len(tracked_objects)}",
            f"Unique Objects: {len(self.detection_analytics['unique_objects'])}"
        ]

        for i, metric in enumerate(perf_metrics):
            y_pos = hud_y + 15 + i * 25
            color = (0, 255, 0) if i == 0 and avg_fps > 25 else (255, 255, 255)
            cv2.putText(frame, metric, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Detection info (center)
        center_x = width // 2
        active_classes_str = ", ".join(active_classes) if active_classes else "None"
        detection_info = f"Detected: {active_classes_str}"

        text_size = cv2.getTextSize(detection_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        x_pos = center_x - text_size[0] // 2
        cv2.putText(frame, detection_info, (x_pos, hud_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # System info (right column)
        system_info = [
            f"Model: {Path(self.detection_config.model_path).stem}",
            f"Device: {self.detection_config.device.upper()}",
            f"Conf: {self.detection_config.confidence_threshold:.2f}",
            f"Resolution: {width}x{height}"
        ]

        for i, info in enumerate(system_info):
            y_pos = hud_y + 15 + i * 25
            text_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            x_pos = width - text_size[0] - 10
            cv2.putText(frame, info, (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Recording indicator
        if self.is_recording:
            cv2.circle(frame, (width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 55, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    def run_detection(self, source: int = 0, width: int = 1280, height: int = 720):
        """Run enhanced real-time detection"""
        # Initialize camera
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {source}")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

        logger.info(f"Camera initialized: {width}x{height}")
        logger.info(f"Running on: {platform.system()} {platform.release()}")
        logger.info("Enhanced Controls:")
        logger.info("  Q: Quit")
        logger.info("  R: Reset tracking")
        logger.info("  +/-: Adjust confidence")
        logger.info("  S: Save frame")
        logger.info("  V: Start/Stop recording")
        logger.info("  C: Clear statistics")
        logger.info("  Z: Add detection zone")
        logger.info("  H: Toggle HUD")

        # Start async processing
        self.running = True
        processing_thread = threading.Thread(target=self._process_frame_async)
        processing_thread.daemon = True
        processing_thread.start()

        show_hud = True
        latest_objects = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Cannot read frame")
                    break

                self.frame_count += 1

                # Add frame to processing queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())

                # Get latest results
                try:
                    result = self.result_queue.get_nowait()
                    latest_objects = result['tracked_objects']
                    inference_time = result['inference_time']
                except queue.Empty:
                    inference_time = 0

                # Draw visualizations
                display_frame = frame.copy()
                display_frame = self._draw_enhanced_detections(display_frame, latest_objects)

                if show_hud:
                    display_frame = self._draw_advanced_hud(display_frame, latest_objects, inference_time)

                # Recording
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(display_frame)

                cv2.imshow('Enhanced Object Detection', display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_system()
                elif key == ord('+') or key == ord('='):
                    self.detection_config.confidence_threshold = min(0.95, self.detection_config.confidence_threshold + 0.05)
                    logger.info(f"Confidence threshold: {self.detection_config.confidence_threshold:.2f}")
                elif key == ord('-'):
                    self.detection_config.confidence_threshold = max(0.05, self.detection_config.confidence_threshold - 0.05)
                    logger.info(f"Confidence threshold: {self.detection_config.confidence_threshold:.2f}")
                elif key == ord('s'):
                    self._save_frame(display_frame)
                elif key == ord('v'):
                    self._toggle_recording(width, height)
                elif key == ord('c'):
                    self._clear_statistics()
                elif key == ord('h'):
                    show_hud = not show_hud
                elif key == ord('z'):
                    self._interactive_zone_creation(frame)
                elif key == ord('i'):
                    self._show_info_panel()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.running = False
            cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
            cv2.destroyAllWindows()
            self._print_final_statistics()

    def _reset_system(self):
        """Reset entire detection system"""
        self.tracker = EnhancedObjectTracker(self.tracking_config)
        self.processor = DetectionProcessor()
        self.detection_analytics = {
            'total_detections': 0,
            'unique_objects': set(),
            'class_counts': defaultdict(int),
            'confidence_history': defaultdict(list),
            'zones': {}
        }
        self.trail_history.clear()
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history.clear()
        self.inference_times.clear()
        logger.info("System reset completed")

    def _save_frame(self, frame: np.ndarray):
        """Save current frame with detections"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Frame saved: {filename}")

    def _toggle_recording(self, width: int, height: int):
        """Toggle video recording"""
        if not self.is_recording:
            # Start recording
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            self.is_recording = True
            logger.info(f"Recording started: {filename}")
        else:
            # Stop recording
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.is_recording = False
            logger.info("Recording stopped")

    def _clear_statistics(self):
        """Clear detection statistics"""
        self.detection_analytics['unique_objects'].clear()
        self.detection_analytics['class_counts'].clear()
        self.detection_analytics['confidence_history'].clear()
        logger.info("Statistics cleared")

    def _interactive_zone_creation(self, frame: np.ndarray):
        """Interactive zone creation (simplified for demo)"""
        # This would typically involve mouse callbacks for polygon creation
        # For now, add a default zone in the center
        height, width = frame.shape[:2]
        center_zone = [
            (width//4, height//4),
            (3*width//4, height//4),
            (3*width//4, 3*height//4),
            (width//4, 3*height//4)
        ]
        self.add_detection_zone(center_zone, f"Zone_{len(self.detection_zones)+1}")
        logger.info(f"Detection zone added: Zone_{len(self.detection_zones)}")

    def _show_info_panel(self):
        """Display detailed information panel"""
        info = f"""
        ========== DETECTION SYSTEM INFO ==========
        Model: {self.detection_config.model_path}
        Device: {self.detection_config.device}
        Confidence Threshold: {self.detection_config.confidence_threshold:.2f}
        IoU Threshold: {self.detection_config.iou_threshold:.2f}

        Tracking Configuration:
        Max Disappeared: {self.tracking_config.max_disappeared}
        Max Distance: {self.tracking_config.max_distance}
        Min Hits: {self.tracking_config.min_hits}
        Use Kalman: {self.tracking_config.use_kalman}

        Current Statistics:
        Total Unique Objects: {len(self.detection_analytics['unique_objects'])}
        Active Zones: {len(self.detection_zones)}
        Average FPS: {sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0:.1f}
        ==========================================
        """
        logger.info(info)

    def _print_final_statistics(self):
        """Print comprehensive final statistics"""
        elapsed_time = time.time() - self.start_time
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        avg_inference = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0

        print("\n" + "="*60)
        print("ENHANCED DETECTION SYSTEM - FINAL REPORT")
        print("="*60)

        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  Total Frames Processed: {self.frame_count:,}")
        print(f"  Total Runtime: {elapsed_time:.1f} seconds")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Average Inference Time: {avg_inference*1000:.1f}ms")
        print(f"  Processing Efficiency: {(avg_inference / (1/avg_fps))*100:.1f}%")

        print(f"\n🎯 DETECTION ANALYTICS:")
        print(f"  Total Unique Objects: {len(self.detection_analytics['unique_objects']):,}")

        if self.detection_analytics['class_counts']:
            print(f"  Objects by Class:")
            for class_name, count in sorted(self.detection_analytics['class_counts'].items(),
                                          key=lambda x: x[1], reverse=True):
                avg_conf = sum(self.detection_analytics['confidence_history'][class_name]) / \
                          len(self.detection_analytics['confidence_history'][class_name])
                print(f"    {class_name}: {count:,} detections (avg conf: {avg_conf:.2f})")

        print(f"\n⚙️  SYSTEM CONFIGURATION:")
        print(f"  Model: {self.detection_config.model_path}")
        print(f"  Device: {self.detection_config.device}")
        print(f"  Confidence Threshold: {self.detection_config.confidence_threshold:.2f}")
        print(f"  IoU Threshold: {self.detection_config.iou_threshold:.2f}")
        print(f"  Detection Zones: {len(self.detection_zones)}")

        print("="*60)

class AlertSystem:
    """Advanced alert system for detection events"""

    def __init__(self):
        self.alerts = []
        self.alert_history = deque(maxlen=100)

    def add_alert(self, alert_type: str, message: str, severity: str = "INFO"):
        """Add an alert to the system"""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        self.alerts.append(alert)
        self.alert_history.append(alert)

        if severity == "CRITICAL":
            logger.critical(f"ALERT: {message}")
        elif severity == "WARNING":
            logger.warning(f"ALERT: {message}")
        else:
            logger.info(f"ALERT: {message}")

def create_configs_from_args(args) -> Tuple[DetectionConfig, TrackingConfig]:
    """Create configuration objects from command line arguments"""
    detection_config = DetectionConfig(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        device=args.device,
        imgsz=args.imgsz,
        half=args.half
    )

    tracking_config = TrackingConfig(
        max_disappeared=args.max_disappeared,
        max_distance=args.max_distance,
        min_hits=args.min_hits,
        use_kalman=args.use_kalman
    )

    return detection_config, tracking_config

def main():
    """Enhanced main function with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description='Enhanced Real-time Object Detection with Advanced Tracking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and detection parameters
    detection_group = parser.add_argument_group('Detection Parameters')
    detection_group.add_argument('--model', default='yolov8n.pt',
                                help='YOLOv8 model path (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    detection_group.add_argument('--confidence', type=float, default=0.5,
                                help='Confidence threshold (0.0-1.0)')
    detection_group.add_argument('--iou', type=float, default=0.45,
                                help='IoU threshold for NMS (0.0-1.0)')
    detection_group.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                                help='Device to use for inference')
    detection_group.add_argument('--imgsz', type=int, default=640,
                                help='Image size for inference')
    detection_group.add_argument('--half', action='store_true',
                                help='Use FP16 half-precision inference')

    # Tracking parameters
    tracking_group = parser.add_argument_group('Tracking Parameters')
    tracking_group.add_argument('--max-disappeared', type=int, default=30,
                               help='Maximum frames an object can disappear before removal')
    tracking_group.add_argument('--max-distance', type=float, default=100,
                               help='Maximum distance for object association')
    tracking_group.add_argument('--min-hits', type=int, default=3,
                               help='Minimum hits before object is considered tracked')
    tracking_group.add_argument('--use-kalman', action='store_true', default=True,
                               help='Use Kalman filtering for tracking')

    # Camera parameters
    camera_group = parser.add_argument_group('Camera Parameters')
    camera_group.add_argument('--camera', type=int, default=0,
                             help='Camera ID or video file path')
    camera_group.add_argument('--width', type=int, default=1280,
                             help='Camera width')
    camera_group.add_argument('--height', type=int, default=720,
                             help='Camera height')

    # System parameters
    system_group = parser.add_argument_group('System Parameters')
    system_group.add_argument('--save-config', type=str,
                             help='Save current configuration to file')
    system_group.add_argument('--load-config', type=str,
                             help='Load configuration from file')
    system_group.add_argument('--log-level', default='INFO',
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                             help='Logging level')

    # Modified to parse known arguments and ignore unknown ones from the Colab environment
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Discarding unknown arguments: {unknown}")

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load configuration if specified
    if args.load_config:
        with open(args.load_config, 'r') as f:
            config_data = json.load(f)
            # Update args with loaded config
            for key, value in config_data.items():
                if hasattr(args, key):
                    setattr(args, key, value)
        logger.info(f"Configuration loaded from {args.load_config}")

    # Create configurations
    detection_config, tracking_config = create_configs_from_args(args)

    # Save configuration if specified
    if args.save_config:
        config_data = {
            **asdict(detection_config),
            **asdict(tracking_config),
            'camera': args.camera,
            'width': args.width,
            'height': args.height
        }
        with open(args.save_config, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Configuration saved to {args.save_config}")

    # Check dependencies
    try:
        import scipy.optimize
        import filterpy.kalman
        logger.info("All optional dependencies available")
    except ImportError as e:
        logger.warning(f"Optional dependency missing: {e}")
        logger.warning("Some features may be limited")

    # Initialize and run detector
    try:
        detector = AdvancedObjectDetector(detection_config, tracking_config)
        detector.run_detection(args.camera, args.width, args.height)
    except Exception as e:
        logger.error(f"Detection system error: {e}")
        return 1

    return 0

if __name__ == "__main__":
   sys.exit(main())
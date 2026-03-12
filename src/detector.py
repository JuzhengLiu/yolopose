"""PyTorch-based object detection using YOLOv8/YOLO-World via ultralytics.

Modern replacement for the old HSV color tracking.
Supports YOLO-Worldv2 for open-vocabulary detection and YOLOv8-Pose for keypoints.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import sys

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

from src.config import Config
from src.utils import get_bbox_center


class Detection:
    """Single detection result."""

    def __init__(
        self, 
        bbox: Tuple[int, int, int, int], 
        confidence: float, 
        class_id: int, 
        class_name: str,
        track_id: Optional[int] = None,
        keypoints: Optional[np.ndarray] = None  # [新增] 关键点数据 (17, 2)
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.track_id = track_id
        self.keypoints = keypoints
        
        # [修改] 计算中心点逻辑：优先使用 Pose 的躯干中心，否则使用 BBox 中心
        self.center = self._calculate_center()

    def _calculate_center(self) -> Tuple[int, int]:
        """Calculate reliable center point."""
        # 默认使用 BBox 中心
        bbox_center = get_bbox_center(self.bbox)
        
        if self.keypoints is None:
            return bbox_center
            
        # 如果有关键点，计算躯干中心 (更鲁棒)
        # COCO 索引: 5:L-Shoulder, 6:R-Shoulder, 11:L-Hip, 12:R-Hip
        torso_indices = [5, 6, 11, 12]
        visible_points = []
        
        for idx in torso_indices:
            if idx < len(self.keypoints):
                x, y = self.keypoints[idx]
                # 过滤掉 (0,0) 的点
                if x > 1 and y > 1: 
                    visible_points.append((x, y))
        
        if len(visible_points) > 0:
            # 计算可见躯干点的平均值
            avg_x = sum(p[0] for p in visible_points) / len(visible_points)
            avg_y = sum(p[1] for p in visible_points) / len(visible_points)
            return (int(avg_x), int(avg_y))
        else:
            return bbox_center

    def __repr__(self) -> str:
        id_str = f", id={self.track_id}" if self.track_id is not None else ""
        pose_str = ", pose=True" if self.keypoints is not None else ""
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox}{id_str}{pose_str})"


class ObjectDetector:
    """Modern object detector using YOLOv8 or YOLO-World.

    Supports various YOLO models and can filter by target classes.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.device = config.device
        self.is_pose_model = False  # 标记是否为 Pose 模型

        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. " "Install with: pip install ultralytics")

        self._load_model()

    def _load_model(self) -> None:
        """Load YOLOv8/YOLO-World model."""
        print(f"Loading {self.config.model_name} on {self.device}...")
        
        # 检查是否为 Pose 模型
        if "pose" in self.config.model_name:
            self.is_pose_model = True
            print("Detected YOLO-Pose model. Keypoint detection enabled.")

        try:
            # 1. 加载模型
            model_file = f"{self.config.model_name}.pt"
            self.model = YOLO(model_file)
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")

            # 2. 如果是 YOLO-World，设置自定义类别 (Pose 模型不支持这个)
            if not self.is_pose_model and "world" in self.config.model_name and self.config.target_classes:
                print(f"Detected YOLO-World model. Setting custom classes: {self.config.target_classes}")
                try:
                    self.model.set_classes(self.config.target_classes)
                    print("Custom classes set successfully.")
                except ImportError:
                    print("\n[CRITICAL ERROR] Missing 'clip' library required for YOLO-World.")
                    print("Please run: pip install openai-clip\n")
                except Exception as e:
                    error_msg = str(e)
                    if "clip" in error_msg.lower() or "no module" in error_msg.lower():
                        print("\n[CRITICAL ERROR] Missing CLIP library.")
                        print("YOLO-World needs CLIP to understand text prompts.")
                        print("Please run this command in your terminal:")
                        print(">>> pip install openai-clip\n")
                    else:
                        print(f"Warning: Failed to set custom classes: {e}")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
            self.model.to(self.device)
            self.is_pose_model = False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of Detection objects
        """
        if self.model is None:
            return []

        # Run inference
        if self.config.tracking_method == "yolo":
            results = self.model.track(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
                persist=True, 
                tracker="bytetrack.yaml"
            )[0]
        else:
            results = self.model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )[0]

        detections = []

        # Parse results
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # 获取 track_ids
            track_ids = None
            if results.boxes.id is not None:
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                
            # 获取 Keypoints (如果是 Pose 模型)
            keypoints_data = None
            if self.is_pose_model and hasattr(results, 'keypoints') and results.keypoints is not None:
                # results.keypoints.xy 形状通常为 (N, 17, 2)
                keypoints_data = results.keypoints.xy.cpu().numpy()

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                # 获取类别名称
                if self.model.names and cls_id < len(self.model.names):
                    class_name = self.model.names[cls_id]
                else:
                    class_name = str(cls_id)

                # Filter by target classes if specified
                if self.config.target_classes:
                    if class_name not in self.config.target_classes:
                        continue
                
                track_id = track_ids[i] if track_ids is not None else None
                
                # 获取当前目标的 keypoints
                kpts = None
                if keypoints_data is not None and i < len(keypoints_data):
                    kpts = keypoints_data[i]

                x1, y1, x2, y2 = map(int, box)
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=class_name,
                    track_id=track_id,
                    keypoints=kpts  # 传入关键点
                )
                detections.append(detection)

        return detections

    def detect_best(self, frame: np.ndarray) -> Optional[Detection]:
        """Detect and return only the best (highest confidence) detection."""
        detections = self.detect(frame)

        if not detections:
            return None

        return max(detections, key=lambda d: d.confidence)

    def detect_closest_to_center(self, frame: np.ndarray) -> Optional[Detection]:
        """Detect and return the detection closest to frame center."""
        detections = self.detect(frame)

        if not detections:
            return None

        h, w = frame.shape[:2]
        frame_center = (w // 2, h // 2)

        # Find detection with center closest to frame center
        closest = min(
            detections,
            key=lambda d: np.sqrt(
                (d.center[0] - frame_center[0]) ** 2 + (d.center[1] - frame_center[1]) ** 2
            ),
        )

        return closest

    def get_model_info(self) -> Dict[str, any]:
        """Get information about loaded model."""
        if self.model is None:
            return {}

        return {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "class_names": list(self.model.names.values()),
            "num_classes": len(self.model.names),
            "is_pose": self.is_pose_model
        }


class HSVDetector:
    """Fallback HSV color-based detector.

    Kept for backwards compatibility and situations where YOLO is too heavy.
    """

    def __init__(self, config: Config, color: str = "green"):
        self.config = config
        self.color = color

        if color not in config.hsv_ranges:
            raise ValueError(
                f"Color {color} not in config. Available: {list(config.hsv_ranges.keys())}"
            )

        self.lower, self.upper = config.hsv_ranges[color]

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """Detect object using HSV color masking.

        Returns single detection (largest contour).
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < 100:
            return None

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest)
        bbox = (x, y, x + w, y + h)

        # Return as Detection object for consistency
        return Detection(
            bbox=bbox,
            confidence=1.0,  # HSV doesn't have confidence
            class_id=0,
            class_name=self.color,
        )


class HybridDetector:
    """Hybrid detector that can switch between YOLO and HSV.

    Useful for testing or fallback scenarios.
    """

    def __init__(self, config: Config, use_yolo: bool = True, hsv_color: str = "green"):
        self.config = config
        self.use_yolo = use_yolo and YOLO_AVAILABLE

        if self.use_yolo:
            self.detector = ObjectDetector(config)
            print("Using YOLO detector")
        else:
            self.detector = HSVDetector(config, hsv_color)
            print(f"Using HSV detector for {hsv_color}")

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """Detect using current detector."""
        if self.use_yolo:
            return self.detector.detect_closest_to_center(frame)
        else:
            return self.detector.detect(frame)

    def switch_mode(self) -> None:
        """Switch between YOLO and HSV detection."""
        if not YOLO_AVAILABLE:
            print("Cannot switch to YOLO - ultralytics not installed")
            return

        self.use_yolo = not self.use_yolo
        print(f"Switched to {'YOLO' if self.use_yolo else 'HSV'} detector")
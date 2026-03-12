#!/usr/bin/env python3
"""
Drone Demo - Autonomous object tracking with DJI Tello (ID Locking Version).

此版本已修改为：
1. 使用 ObjectTracker 进行多目标 ID 记忆。
2. 自动锁定画面中心的目标 ID。
3. 即使目标移动到边缘，只要 ID 未丢失，就持续跟随该 ID，忽略其他目标。
4. 按 'r' 键可以重置锁定，重新寻找中心目标。
5. 支持 YOLOv8-Pose 显示骨架。
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config, DroneConfig, get_drone_config
from src.detector import ObjectDetector
from src.drone_controller import DroneController, DroneState, MockDroneController
from src.tracker import ObjectTracker
from src.utils import (
    FPSCounter,
    draw_bbox,
    draw_crosshair,
    draw_info_panel,
    draw_trajectory,
    draw_vector,
    draw_pose, # [新增]
)


class DroneDemo:
    """Drone tracking demo application."""

    def __init__(
        self, config: Config, drone_config: DroneConfig, use_mock: bool = False
    ):
        self.config = config
        self.drone_config = drone_config

        # Initialize components
        print("Initializing detector...")
        self.detector = ObjectDetector(config)
        print(f"Loaded {config.model_name} on {config.device}")

        # Initialize Tracker
        self.tracker = ObjectTracker(config)
        self.locked_target_id = None  # 记录当前锁定的目标 ID

        self.fps_counter = FPSCounter()

        # Initialize drone controller
        if use_mock:
            print("Using mock drone controller (no hardware)")
            self.drone = MockDroneController(config, drone_config)
            self.use_mock = True
        else:
            print("Using real drone controller")
            self.drone = DroneController(config, drone_config)
            self.use_mock = False

        # State
        self.running = False
        self.show_hud = True
        self.show_fps = True
        self.show_telemetry = True
        self.recording = False
        self.video_writer = None

        # Manual control state
        self.manual_speed = 30

        self._print_controls()

    def _print_controls(self) -> None:
        """Print control instructions."""
        print("\n" + "=" * 60)
        print("CONTROLS")
        print("=" * 60)
        print("Flight:")
        print("  TAB       - Takeoff")
        print("  BACKSPACE - Land")
        print("  ESC       - Emergency stop")
        print("  SPACE     - Toggle tracking mode")
        print("\nManual control (tracking off):")
        print("  w/s       - Forward/Backward")
        print("  a/d       - Left/Right")
        print("  UP/DOWN   - Ascend/Descend")
        print("  LEFT/RIGHT- Rotate Left/Right")
        print("\nDisplay:")
        print("  h - Toggle HUD")
        print("  f - Toggle FPS")
        print("  t - Toggle telemetry")
        print("  r - Reset Lock (Find new target)")
        print("  v - Record video")
        print("  c - Take photo")
        print("  q - Quit (will land first)")
        print("=" * 60)
        print()

    def start(self) -> None:
        """Start the demo."""
        # Connect to drone
        if not self.drone.connect():
            print("Failed to connect to drone")
            return

        print("\nDrone connected! Ready to fly.")
        print("Press TAB to takeoff when ready.")

        self.running = True
        self.run_loop()

    def run_loop(self) -> None:
        """Main processing loop."""
        while self.running:
            # Get frame
            if self.use_mock:
                # For mock, use webcam
                if not hasattr(self, "mock_cap"):
                    self.mock_cap = cv2.VideoCapture(0)
                ret, frame = self.mock_cap.read()
                if not ret:
                    continue
            else:
                frame = self.drone.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

            # Update FPS
            self.fps_counter.update()

            # Process frame
            processed_frame = self.process_frame(frame)

            # Display
            cv2.imshow("Drone Tracking Demo", processed_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self.handle_keypress(key)

        self.cleanup()

    def process_frame(self, frame):
        """Process a single frame with ID locking logic."""
        display_frame = frame.copy()

        # Detect and track
        if self.drone.is_flying() or self.use_mock or True:
            # 1. 检测 (Detect)
            detections = self.detector.detect(frame)
            
            # [技巧] 为了在 update 之后还能找到 detection 的 keypoints (因为 TrackedObject 不一定存了)，
            # 我们先建立一个临时映射: track_id -> detection (如果使用 yolo 模式)
            # 或者通过坐标匹配。最简单的方法是修改 Tracker 让他存 keypoints，但这里我们用一个简单字典辅助显示
            # 注意: 如果是 custom 模式，ID 是 tracker 分配的，只有 update 后才知道。
            # 如果是 yolo 模式，detection 里已经有 track_id 了。
            
            # 2. 更新追踪器 (Tracker Update)
            tracked_objects = self.tracker.update(detections)
            
            # 建立 ID 到当前帧 Detection 的映射，以便获取 Keypoints
            # 注意：Tracker update 后，tracked_objects 里的 center 已经被 detection 的 center 更新了
            # 我们需要回溯找到对应的 detection 来拿 keypoints
            id_to_keypoints = {}
            if self.detector.is_pose_model:
                # 这是一个简化的匹配：我们假设 TrackedObject 的 center 和 Detection 的 center 是一样的
                # 或者如果 mode 是 yolo，直接用 track_id
                for det in detections:
                    if det.keypoints is not None:
                        # 尝试匹配 tracked_objects
                        for obj_id, obj in tracked_objects.items():
                             # 如果使用 YOLO 模式，det.track_id 应该等于 obj_id
                             if self.config.tracking_method == "yolo" and det.track_id == obj_id:
                                 id_to_keypoints[obj_id] = det.keypoints
                                 break
                             # 如果是 custom 模式，比较 bbox 或 center
                             elif self.config.tracking_method == "custom":
                                 # 简单比较 center 距离极小
                                 dist = (det.center[0]-obj.center[0])**2 + (det.center[1]-obj.center[1])**2
                                 if dist < 5: # 像素距离平方
                                     id_to_keypoints[obj_id] = det.keypoints
                                     break

            target = None
            
            # 3. ID 锁定逻辑 (ID Locking)
            if self.locked_target_id is not None:
                if self.locked_target_id in tracked_objects:
                    target = tracked_objects[self.locked_target_id]
                else:
                    pass # 目标暂时丢失
            
            # 4. 如果没有锁定目标，寻找新目标
            if self.locked_target_id is None or (target is None and self.locked_target_id not in tracked_objects):
                 if len(tracked_objects) > 0:
                    h, w = frame.shape[:2]
                    center = (w // 2, h // 2)
                    
                    best_obj = min(tracked_objects.values(), 
                                   key=lambda o: (o.center[0]-center[0])**2 + (o.center[1]-center[1])**2)
                    
                    self.locked_target_id = best_obj.id
                    target = best_obj
                    print(f"Locked on new target ID: {target.id}")

            # 5. 自主跟随
            if self.drone.tracking_enabled and target is not None:
                self.drone.track_target(target)
            elif self.drone.tracking_enabled and target is None:
                self.drone.hover()

            # 6. 可视化
            for obj in tracked_objects.values():
                is_locked = (obj.id == self.locked_target_id)
                
                if is_locked:
                    color = (0, 255, 0) if self.drone.tracking_enabled else (0, 139, 139)
                    thickness = 3
                else:
                    color = (0, 0, 255)
                    thickness = 1
                
                # [新增] 绘制骨架 (如果可用)
                if obj.id in id_to_keypoints:
                    display_frame = draw_pose(display_frame, id_to_keypoints[obj.id], color=color)
                    
                # 绘制边框
                label = f"{obj.class_name} ID:{obj.id}"
                display_frame = draw_bbox(
                    display_frame,
                    obj.bbox,
                    label,
                    obj.confidence,
                    color,
                    thickness=thickness,
                )

                if is_locked:
                    if len(obj.centers) > 1:
                        display_frame = draw_trajectory(
                            display_frame, list(obj.centers), color, thickness=2
                        )

                    if self.drone.tracking_enabled:
                        h, w = frame.shape[:2]
                        frame_center = (w // 2, h // 2)
                        display_frame = draw_vector(
                            display_frame,
                            obj.center,
                            frame_center,
                            (255, 0, 255),
                            thickness=2,
                        )

        # Draw frame center crosshair
        display_frame = draw_crosshair(display_frame, color=(0, 0, 255), size=30)

        # Draw HUD
        if self.show_hud:
            display_frame = self.draw_hud(display_frame)

        if self.recording and self.video_writer:
            self.video_writer.write(display_frame)

        return display_frame

    def draw_hud(self, frame):
        """Draw heads-up display."""
        info = {}

        # FPS
        if self.show_fps:
            info["FPS"] = f"{self.fps_counter.get_fps():.1f}"

        # Drone state
        info["State"] = self.drone.state.value.upper()

        # Tracking state
        if self.drone.is_flying():
            tracking_status = "ACTIVE" if self.drone.tracking_enabled else "MANUAL"
            info["Mode"] = tracking_status
        
        info["Algo"] = self.config.tracking_method.upper()

        # Target info
        if self.locked_target_id is not None and self.tracker.objects:
             obj = self.tracker.get_object(self.locked_target_id)
             if obj:
                info["Target"] = f"ID:{obj.id}"
                info["Conf"] = f"{obj.confidence:.2f}"
             else:
                info["Target"] = "Lost"
        else:
            info["Target"] = "Scanning"

        # Telemetry
        if self.show_telemetry and not self.use_mock:
            telemetry = self.drone.get_telemetry()
            info["Bat"] = f"{telemetry.get('battery', 0)}%"
            info["H"] = f"{telemetry.get('height', 0)}cm"

        if self.recording:
            info["REC"] = "●"

        frame = draw_info_panel(
            frame,
            info,
            position="top-left",
            bg_color=(0, 0, 0),
            text_color=(0, 255, 0) if self.drone.tracking_enabled else (255, 165, 0),
            alpha=0.7,
        )

        return frame

    def handle_keypress(self, key: int) -> None:
        """Handle keyboard input."""
        # Flight controls
        if key == 9:  # TAB
            if self.drone.state == DroneState.CONNECTED:
                print("Taking off...")
                self.drone.takeoff()

        elif key == 8:  # BACKSPACE
            if self.drone.is_flying():
                print("Landing...")
                self.drone.land()

        elif key == 27:  # ESC
            print("EMERGENCY STOP!")
            self.drone.emergency_stop()
            self.running = False

        elif key == ord(" "):  # SPACE
            if self.drone.is_flying():
                if self.drone.tracking_enabled:
                    self.drone.disable_tracking()
                    print("Tracking disabled - manual control active")
                else:
                    self.drone.enable_tracking()
                    print("Tracking enabled - autonomous mode")

        # Manual controls (only when not tracking)
        elif key == ord("w") and not self.drone.tracking_enabled:
            self.drone.manual_control(fb=self.manual_speed)

        elif key == ord("s") and not self.drone.tracking_enabled:
            self.drone.manual_control(fb=-self.manual_speed)

        elif key == ord("a") and not self.drone.tracking_enabled:
            self.drone.manual_control(lr=-self.manual_speed)

        elif key == ord("d") and not self.drone.tracking_enabled:
            self.drone.manual_control(lr=self.manual_speed)

        elif key == 82 and not self.drone.tracking_enabled:  # UP arrow
            self.drone.manual_control(ud=self.manual_speed)

        elif key == 84 and not self.drone.tracking_enabled:  # DOWN arrow
            self.drone.manual_control(ud=-self.manual_speed)

        elif key == 81 and not self.drone.tracking_enabled:  # LEFT arrow
            self.drone.manual_control(yaw=-self.manual_speed)

        elif key == 83 and not self.drone.tracking_enabled:  # RIGHT arrow
            self.drone.manual_control(yaw=self.manual_speed)

        # Display controls
        elif key == ord("h"):
            self.show_hud = not self.show_hud
            print(f"HUD {'shown' if self.show_hud else 'hidden'}")

        elif key == ord("f"):
            self.show_fps = not self.show_fps
            print(f"FPS display {'shown' if self.show_fps else 'hidden'}")

        elif key == ord("t"):
            self.show_telemetry = not self.show_telemetry
            print(f"Telemetry {'shown' if self.show_telemetry else 'hidden'}")

        elif key == ord("r"):
            print("Resetting tracker and lock...")
            self.tracker.reset()
            self.locked_target_id = None
        
        elif key == ord("v"):
            self.toggle_recording()

        elif key == ord("c"):
            self.take_photo()

        elif key == ord("q"):
            print("Quitting...")
            if self.drone.is_flying():
                print("Landing before quit...")
                self.drone.land()
            self.running = False

    def toggle_recording(self) -> None:
        """Toggle video recording."""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                self.config.fps,
                (self.config.frame_width, self.config.frame_height),
            )

            self.recording = True
            print(f"Recording started: {filename}")
        else:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            self.recording = False
            print("Recording stopped")

    def take_photo(self) -> None:
        """Take a photo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"

        frame = self.drone.get_frame()
        if frame is not None:
            cv2.imwrite(filename, frame)
            print(f"Photo saved: {filename}")
        else:
            print("Failed to capture photo")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.recording and self.video_writer:
            self.video_writer.release()

        if hasattr(self, "mock_cap"):
            self.mock_cap.release()

        self.drone.disconnect()
        cv2.destroyAllWindows()
        print("Demo stopped")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Drone demo for autonomous object tracking (ID Lock Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        # [修改] 默认值
        default="yolov8s-worldv2",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", 
                 "yolov8s-worldv2", "yolov8n-pose", "yolov8s-pose", "yolov8m-pose"],
        help="YOLO model to use (default: yolov8s-worldv2)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Detection confidence threshold (default: 0.6)",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Target classes to detect (e.g., person ball)",
    )

    parser.add_argument(
        "--tracking-method",
        type=str,
        default="custom",
        choices=["custom", "yolo"],
        help="Tracking method: 'custom' (simple distance) or 'yolo' (ByteTrack/BoT-SORT)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run model on (default: auto)",
    )

    parser.add_argument(
        "--speed", type=int, default=50, help="Drone movement speed 0-100 (default: 50)"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock drone (for testing without hardware)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create configurations
    config, drone_config = get_drone_config()
    config.model_name = args.model
    config.confidence_threshold = args.confidence
    config.drone_speed = args.speed
    
    config.tracking_method = args.tracking_method

    if args.classes:
        config.target_classes = args.classes

    if args.device:
        config.device = args.device

    # Safety check
    if not args.mock:
        print("\n" + "!" * 60)
        print("SAFETY WARNING")
        print("!" * 60)
        print("You are about to fly a real drone.")
        print("- Ensure you are in an open area")
        print("- Keep away from people and obstacles")
        print("- Monitor battery level")
        print("- Be ready to emergency stop (ESC key)")
        print("!" * 60)

        response = input("\nDo you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted")
            return

    # Create and start demo
    demo = DroneDemo(config, drone_config, use_mock=args.mock)

    try:
        demo.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        demo.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        demo.cleanup()


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# -------------------------------------------------
# DEVICE AUTO SELECT (GPU if available)
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------
# YOLO MODEL
# -------------------------------------------------
try:
    model = YOLO("yolov8n.pt")
    model.to(device)
    print("✓ YOLO model loaded successfully")
except Exception as e:
    print(f"✗ Error loading YOLO model: {e}")
    print("Please ensure yolov8n.pt is available")
    exit(1)

# -------------------------------------------------
# Gripper Simulation
# -------------------------------------------------
def draw_gripper(frame, center, width, grip_percent):
    try:
        if frame is None or center is None or width <= 0:
            return
        cx, cy = center
        if cx < 0 or cy < 0:
            return
        grip_percent = max(0, min(1, grip_percent))
        half = int(width * (1 - grip_percent))
        if half < 0:
            half = 0
        cv2.line(frame, (cx-half, cy), (cx-10, cy), (0,255,0), 4)
        cv2.line(frame, (cx+10, cy), (cx+half, cy), (0,255,0), 4)
    except Exception:
        pass

# -------------------------------------------------
# Mass Estimation using Depth Map (3D Volume)
# -------------------------------------------------


# Fallback: Heuristic mass estimation (if depth unavailable)
def estimate_mass(area, material, width):
    try:
        if area <= 0 or width <= 0:
            return 0.001
        
        base_mass_grams = area * 0.002
        
        density_map = {
            "Metal": 2.5,
            "Plastic": 0.7,
            "Glossy": 1.0,
            "Organic": 1.05
        }
        multiplier = density_map.get(material, 1.0)
        mass_grams = base_mass_grams * multiplier
        
        mass_grams = max(0.1, min(mass_grams, 2000))
        mass_kg = mass_grams / 1000
        return round(max(mass_kg, 0.001), 3)
    except Exception:
        return 0.001

# -------------------------------------------------
# Collision Risk Prediction
# -------------------------------------------------
def collision_risk(depth, width):
    try:
        if depth is None or width is None:
            return "UNKNOWN", (128,128,128)
        if depth < 0:
            depth = 0
        if width < 0:
            width = 0
        if depth < 15:
            return "TOO CLOSE", (0,0,255)
        if width > 250:
            return "OVERSIZED", (0,140,255)
        return "SAFE", (0,255,0)
    except Exception:
        return "ERROR", (128,128,128)

# -------------------------------------------------
# Gripper Orientation Optimizer
# -------------------------------------------------
def get_best_orientation(width, height):
    try:
        if width is None or height is None or width <= 0 or height <= 0:
            return "UNKNOWN", "?", 0
        
        if width < height:
            orientation = "HORIZONTAL"
            min_opening = width
            grip_dir = "↔"
        else:
            orientation = "VERTICAL"
            min_opening = height
            grip_dir = "↕"
        
        return orientation, grip_dir, min_opening
    except Exception:
        return "ERROR", "?", 0

# -------------------------------------------------
# Material Detection
# -------------------------------------------------
def detect_material(roi):
    try:
        if roi is None or roi.size == 0:
            return "Unknown"
        
        if roi.shape[0] < 2 or roi.shape[1] < 2:
            return "Too Small"
        
        if len(roi.shape) != 3 or roi.shape[2] < 3:
            return "Invalid"
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue = hsv[:,:,0].mean()
        sat = hsv[:,:,1].mean()
        val = hsv[:,:,2].mean()

        is_white_egg = (val > 150 and sat < 80)
        is_brown_egg = (15 <= hue <= 40 and sat < 120 and val > 120)
        
        if is_white_egg or is_brown_egg:
            return "Organic"
        elif val > 200 and sat < 100:
            return "Glossy"
        elif sat < 35 and val < 180:
            return "Metal"
        else:
            return "Plastic"
    except Exception:
        return "Error"

# -------------------------------------------------
# CAMERA INITIALIZATION
# -------------------------------------------------
print("\n→ Initializing camera...")

# Scan all available cameras
available_cameras = {}
for cam_idx in range(10):
    try:
        test_cam = cv2.VideoCapture(cam_idx)
        if test_cam.isOpened():
            ret, test_frame = test_cam.read()
            if ret and test_frame is not None:
                available_cameras[cam_idx] = test_frame.shape
                print(f"Camera {cam_idx}: {test_frame.shape[1]}x{test_frame.shape[0]}")
                test_cam.release()
    except:
        pass

if not available_cameras:
    print("✗ No cameras found!")
    exit(1)

print(f"\n✓ Found {len(available_cameras)} camera(s)\n")

# Use first available camera for RGB
rgb_idx = list(available_cameras.keys())[0]
print(f"→ Camera {rgb_idx}: RGB Camera (Material Inspection & Detection)")

# Initialize RGB camera with error handling
try:
    rgb = cv2.VideoCapture(rgb_idx)
    if not rgb.isOpened():
        raise Exception(f"Failed to open camera {rgb_idx}")
    
    rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    rgb.set(cv2.CAP_PROP_FPS, 30)
    rgb.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"✓ RGB Camera initialized (Camera {rgb_idx})")
except Exception as e:
    print(f"✗ Error initializing camera: {e}")
    exit(1)

print("\n" + "="*60)
print("MASS CALCULATION SYSTEM")
print("="*60)
print("Method: 2D Heuristic - Size-based mass estimation")
print("  Uses pixel area and material density estimates")
print("  Materials: Metal(2.5x), Plastic(0.7x), Glass(1.0x), Organic(1.05x)")
print()
print("OBJECT TRACKING - Single Object Mode:")
print("  • Locks to first detected object")
print("  • Tracks object across frames")
print("  • Press 'R' to release and track new object")
print("  • Press 'ESC' to exit")
print("="*60 + "\n")

# -------------------------------------------------
# OBJECT TRACKING - Single Object Lock
# -------------------------------------------------
locked_object = None  # Stores locked object info
lock_timeout = 30     # Frames without detection before auto-release
frame_count_no_lock = 0



class LockedObject:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.cx = (x1 + x2) // 2
        self.cy = (y1 + y2) // 2
        self.width = max(x2 - x1, 1)
        self.height = max(y2 - y1, 1)
        self.frames_tracked = 0
        self.lost_frames = 0
    
    def distance_to(self, x1, y1, x2, y2):
        try:
            new_cx = (x1 + x2) // 2
            new_cy = (y1 + y2) // 2
            cx_dist = abs(new_cx - self.cx)
            cy_dist = abs(new_cy - self.cy)
            return np.sqrt(cx_dist**2 + cy_dist**2)
        except Exception:
            return float('inf')
    
    def is_same_object(self, x1, y1, x2, y2, max_distance=100):
        try:
            dist = self.distance_to(x1, y1, x2, y2)
            new_width = max(x2 - x1, 1)
            new_height = max(y2 - y1, 1)
            
            width_ratio = new_width / self.width if self.width > 0 else 1
            height_ratio = new_height / self.height if self.height > 0 else 1
            
            size_ok = (0.5 < width_ratio < 2.0) and (0.5 < height_ratio < 2.0)
            distance_ok = dist < max_distance
            
            return size_ok and distance_ok
        except Exception:
            return False
    
    def update(self, x1, y1, x2, y2):
        try:
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
            self.cx = (x1 + x2) // 2
            self.cy = (y1 + y2) // 2
            self.width = max(x2 - x1, 1)
            self.height = max(y2 - y1, 1)
            self.frames_tracked += 1
            self.lost_frames = 0
        except Exception:
            pass

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
frame_count = 0
object_data = {}  # Store latest object data for display

while True:
    try:
        # ========== GET FRAME ==========
        ret, frame = rgb.read()
        if not ret or frame is None:
            print("⚠ Warning: Failed to read frame from camera")
            continue
        
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            continue
        
        # ========== YOLO DETECTION ==========
        try:
            results = model(frame, conf=0.5, imgsz=480)[0]
        except Exception as e:
            print(f"⚠ YOLO inference error: {e}")
            continue
        
        # Sort detections by size (largest first)
        detections = []
        try:
            if results.boxes is not None:
                for box in results.boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Validate coordinates
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(x1+1, min(x2, w))
                        y2 = max(y1+1, min(y2, h))
                        area = (x2 - x1) * (y2 - y1)
                        if area > 0:  # Only add valid detections
                            detections.append((area, x1, y1, x2, y2))
                    except Exception:
                        pass  # Skip invalid boxes
        except Exception:
            pass
        
        detections.sort(reverse=True, key=lambda x: x[0])  # Sort by area (largest first)
        
        # ========== OBJECT TRACKING LOGIC ==========
        processed_box = None
        matched = False  # Initialize matched variable
        
        if locked_object is None and detections:
            # No object locked - lock to the largest detected object
            area, x1, y1, x2, y2 = detections[0]
            locked_object = LockedObject(x1, y1, x2, y2)
            processed_box = (x1, y1, x2, y2)
            lock_status = "LOCKED - NEW"
            lock_color = (0, 255, 0)  # Green
            
        elif locked_object is not None:
            # Object already locked - try to find and track it
            matched = False
            for area, x1, y1, x2, y2 in detections:
                if locked_object.is_same_object(x1, y1, x2, y2):
                    # Found matching object
                    locked_object.update(x1, y1, x2, y2)
                    processed_box = (x1, y1, x2, y2)
                    lock_status = f"TRACKED ({locked_object.frames_tracked}f)"
                    lock_color = (0, 255, 0)  # Green
                    matched = True
                    break
            
            if not matched:
                # Object lost
                locked_object.lost_frames += 1
                if locked_object.lost_frames > lock_timeout:
                    # Auto-release after timeout
                    lock_status = "LOST - AUTO RELEASED"
                    lock_color = (0, 0, 255)  # Red
                    locked_object = None
                else:
                    # Still tracking last known position
                    lock_status = f"LOST ({locked_object.lost_frames}/{lock_timeout})"
                    lock_color = (0, 165, 255)  # Orange
                    processed_box = (locked_object.x1, locked_object.y1, locked_object.x2, locked_object.y2)

        # Process only the locked/selected object
        if processed_box is not None:
            try:
                x1, y1, x2, y2 = processed_box
                
                # Validate box coordinates
                x1, y1, x2, y2 = max(0, x1), max(0, y1), max(x1+1, x2), max(y1+1, y2)
                x1, y1, x2, y2 = min(x1, w-1), min(y1, h-1), min(x2, w), min(y2, h)
                
                width = x2 - x1
                height = y2 - y1
                
                # Skip if box is too small
                if width < 5 or height < 5:
                    object_data = {}
                    continue
                
                area = width * height

                # Size-based depth estimation (no depth camera)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                pixel_size = max(width, 5)
                depth_cm = int(np.interp(pixel_size, [10, 100, 250], [100, 35, 15]))

                # -------- LOCATION TRACKING (X, Y, Z) --------
                # X: Horizontal position in frame (0-640)
                # Y: Vertical position in frame (0-480)
                # Z: Depth in centimeters (estimated from size)
                location_x = cx
                location_y = cy
                location_z = depth_cm

                # Material Inspection: Extract ROI from RGB camera for material detection
                roi = frame[y1:y2, x1:x2].copy()
                if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
                    object_data = {}
                    continue

                # Detect material from RGB camera ROI
                material = detect_material(roi)
                
                # Calculate mass using heuristic method
                mass = estimate_mass(area, material, width)
                calc_method = "Heuristic"
                color = (0, 165, 255)  # Orange for heuristic

                # -------- Grip Logic --------
                object_width_ratio = width / 640.0
                grip_percent = min(max(object_width_ratio, 0), 0.9)
                
                # -------- Gripper Orientation --------
                orientation, grip_dir, min_opening = get_best_orientation(width, height)

                # -------- Safety --------
                risk_text, risk_color = collision_risk(depth_cm, width)

                # -------- Grasp Confidence --------
                confidence = 1.0
                if depth_cm > 60: confidence -= 0.3
                if width < 30: confidence -= 0.2
                confidence = round(max(confidence, 0), 2)

                # -------- Draw --------
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_gripper(frame, (cx, cy), width, grip_percent)
                
                # -------- Draw Location Marker --------
                # Draw crosshair at object center showing (X, Y) position
                marker_size = 15
                cv2.line(frame, (cx - marker_size, cy), (cx + marker_size, cy), (0, 200, 255), 2)
                cv2.line(frame, (cx, cy - marker_size), (cx, cy + marker_size), (0, 200, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 200, 255), -1)  # Center dot
                
                # Display location label near the object
                cv2.putText(frame, f"({location_x}, {location_y}, {location_z}cm)", 
                            (cx + 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                
                # Store object data for display at bottom right later
                object_data = {
                    'location_x': location_x,
                    'location_y': location_y,
                    'location_z': location_z,
                    'depth': depth_cm,
                    'material': material,
                    'mass': mass,
                    'calc_method': calc_method,
                    'calc_color': color,
                    'risk': risk_text,
                    'risk_color': risk_color,
                    'confidence': confidence,
                    'orientation': orientation,
                    'grip_dir': grip_dir,
                    'min_opening': min_opening
                }
            except Exception as e:
                # Graceful degradation - clear data and continue
                object_data = {}
                pass

        # ========== LOCK STATUS DISPLAY ==========
        # Display object tracking status at top left of frame
        if locked_object is not None:
            cv2.putText(frame, f"LOCK: {lock_status}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, lock_color, 2)
        else:
            cv2.putText(frame, "READY - Waiting for object", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        
        # Instructions
        cv2.putText(frame, "[R] Release  [ESC] Exit", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # ========== DISPLAY OBJECT DATA ==========
        if object_data:
            # Starting position for data display (bottom left area)
            start_x = 20
            start_y = max(frame.shape[0] - 160, 100)  # Start from bottom with safety margin
            line_height = 16
            font_size = 0.4
            font_thickness = 1
            
            # Color codes
            color_value = (0, 255, 255)    # Cyan
            color_warning = (0, 165, 255)  # Orange
            color_location = (0, 200, 255)  # Light Orange for location
            
            y_pos = start_y
            
            # ========== LOCATION (X, Y, Z) ==========
            loc_x = object_data.get('location_x', 'N/A')
            loc_y = object_data.get('location_y', 'N/A')
            loc_z = object_data.get('location_z', 'N/A')
            cv2.putText(frame, f"Location: X={loc_x} Y={loc_y} Z={loc_z}cm", 
                        (start_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_size, color_location, font_thickness)
            y_pos += line_height
            
            # Depth
            cv2.putText(frame, f"Depth: {object_data.get('depth', 'N/A')} cm", 
                        (start_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_size, color_value, font_thickness)
            y_pos += line_height
            
            # Material
            cv2.putText(frame, f"Material: {object_data.get('material', 'N/A')}", 
                        (start_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_size, color_value, font_thickness)
            y_pos += line_height
            
            # Mass
            cv2.putText(frame, f"Mass: ~{object_data.get('mass', 'N/A')} kg", 
                        (start_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_size, color_value, font_thickness)
            y_pos += line_height
            
            # Calculation method
            calc_color = object_data.get('calc_color', color_warning)
            cv2.putText(frame, f"Method: [{object_data.get('calc_method', 'N/A')}]", 
                        (start_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_size, calc_color, font_thickness)
            y_pos += line_height
            
            # Risk
            risk_color = object_data.get('risk_color', (0, 0, 255))
            cv2.putText(frame, f"Safety: {object_data.get('risk', 'N/A')}", 
                        (start_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_size, risk_color, font_thickness)
            y_pos += line_height
            
            # Confidence
            cv2.putText(frame, f"Confidence: {object_data.get('confidence', 'N/A')}", 
                        (start_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 200), font_thickness)
            y_pos += line_height
            
            # Orientation
            grip_dir = object_data.get('grip_dir', '')
            cv2.putText(frame, f"Orientation: {object_data.get('orientation', 'N/A')} {grip_dir}", 
                        (start_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 200, 255), font_thickness)
        
        # Display frame
        cv2.imshow("RGB Camera - Gripper Control", frame)

        # ========== KEY INPUT HANDLING ==========
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):  # Release lock
            if locked_object is not None:
                print("Object lock released by user")
                locked_object = None
    except KeyboardInterrupt:
        print("\n→ Interrupted by user")
        break
    except Exception as e:
        print(f"⚠ Error in main loop: {e}")
        continue

# Cleanup
print("\n→ Shutting down...")

try:
    if rgb is not None:
        rgb.release()
        print("✓ RGB camera released")
except Exception as e:
    print(f"⚠ Error releasing camera: {e}")

try:
    cv2.destroyAllWindows()
except Exception as e:
    print(f"⚠ Error closing windows: {e}")

print("✓ System shutdown complete.")
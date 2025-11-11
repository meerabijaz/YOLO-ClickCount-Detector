import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        self.classes_to_detect = ["person", "car"]
        self.class_ids = [0, 2]  # COCO: 0=person, 2=car
        
        # Vehicle counting
        self.car_count_in = 0
        self.car_count_out = 0
        self.counted_ids = set()
        self.line_position = None
        self.previous_positions = defaultdict(lambda: {"x": 0, "y": 0})
        
        # Click detection
        self.clicked_regions = []
        self.click_radius = 50
        self.detected_objects = []
        
    def set_counting_line(self, y_position):
        """Set the horizontal line for vehicle counting"""
        self.line_position = y_position
    
    def detect_objects(self, frame):
        """Detect objects using YOLO"""
        results = self.model(frame)
        self.detected_objects = []
        
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                if class_id in self.class_ids:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    class_name = self.classes_to_detect[0] if class_id == 0 else self.classes_to_detect[1]
                    
                    self.detected_objects.append({
                        "bbox": (x1, y1, x2, y2),
                        "center": (center_x, center_y),
                        "class": class_name,
                        "conf": conf,
                        "id": track_id
                    })
                    
                    # Count cars crossing the line
                    if class_name == "car" and self.line_position:
                        self._count_vehicle_crossing(track_id, center_y)
                    
                    self.previous_positions[track_id] = {"x": center_x, "y": center_y}
        
        return self.detected_objects
    
    def _count_vehicle_crossing(self, track_id, current_y):
        """Count vehicles crossing a line"""
        if track_id in self.previous_positions:
            prev_y = self.previous_positions[track_id]["y"]
            threshold = 5
            
            if track_id not in self.counted_ids:
                if prev_y < self.line_position <= current_y:
                    self.car_count_in += 1
                    self.counted_ids.add(track_id)
                elif prev_y > self.line_position >= current_y:
                    self.car_count_out += 1
                    self.counted_ids.add(track_id)
    
    def is_in_clicked_region(self, obj_center):
        """Check if object is within any clicked region"""
        for click_point in self.clicked_regions:
            distance = np.sqrt((obj_center[0] - click_point[0])**2 + 
                             (obj_center[1] - click_point[1])**2)
            if distance <= self.click_radius:
                return True
        return False
    
    def draw_detections(self, frame, use_click_filter=False):
        """Draw bounding boxes and labels"""
        for obj in self.detected_objects:
            if use_click_filter and not self.is_in_clicked_region(obj["center"]):
                continue
            
            x1, y1, x2, y2 = obj["bbox"]
            color = (0, 255, 0) if obj["class"] == "person" else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{obj['class'].upper()} {obj['conf']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_counting_line(self, frame):
        """Draw the counting line"""
        if self.line_position:
            cv2.line(frame, (0, self.line_position), 
                    (frame.shape[1], self.line_position), (255, 255, 0), 2)
            cv2.putText(frame, f"IN: {self.car_count_in} OUT: {self.car_count_out}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def draw_clicked_regions(self, frame):
        """Draw clicked region circles"""
        for point in self.clicked_regions:
            cv2.circle(frame, point, self.click_radius, (255, 0, 255), 2)
        
        return frame

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks"""
    if event == cv2.EVENT_LBUTTONDOWN:
        detector, frame = param
        detector.clicked_regions.append((x, y))
        print(f"Clicked at ({x}, {y}) - Detection radius activated")

def main():
    detector = YOLODetector()
    
    # Open video file
    video_path = "road.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return
    
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector.set_counting_line(frame_height // 2)
    
    cv2.namedWindow("YOLO Detection")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (800, 600))
        
        # Set mouse callback
        param = (detector, frame)
        cv2.setMouseCallback("YOLO Detection", mouse_callback, param)
        
        # Detect objects
        detector.detect_objects(frame)
        
        # Draw detections with click filter
        use_click_filter = len(detector.clicked_regions) > 0
        frame = detector.draw_detections(frame, use_click_filter)
        frame = detector.draw_counting_line(frame)
        frame = detector.draw_clicked_regions(frame)
        
        # Display info
        mode = "CLICK MODE (Only clicked regions)" if use_click_filter else "ALL DETECTION"
        cv2.putText(frame, mode, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Left Click to activate detection region | C: Clear | Q: Quit", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("YOLO Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            detector.clicked_regions = []
            print("Cleared all clicked regions")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

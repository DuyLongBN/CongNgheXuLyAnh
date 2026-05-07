"""
Webcam-based license plate recognition - Simple Version
Nhận dạng biển số xe từ webcam - Phiên bản đơn giản
"""
import cv2
import numpy as np
from ultralytics import YOLO
from src.vietnamese_ocr import VietnameseOCREngine
from collections import deque

# Initialize model and Vietnamese OCR engine
model = YOLO("models/best.pt")
ocr = VietnameseOCREngine(use_gpu=False)

cap = cv2.VideoCapture(0)

# Temporal smoothing buffers
plate_text_buffer = deque(maxlen=15)
detection_frames = 0

def is_valid_plate_roi(frame_roi, x1, y1, x2, y2):
    """
    Validate if region is likely a license plate (not a light/reflector)
    """
    h, w = frame_roi.shape[:2]
    
    # Check aspect ratio
    aspect_ratio = w / h if h > 0 else 0
    is_horizontal = 2.5 <= aspect_ratio <= 6.0   # 1 dòng
    is_square = 0.5 <= aspect_ratio < 2.5         # 2 dòng (phổ biến VN)
    is_vertical = 0.15 <= aspect_ratio < 0.5      # dọc
    
    if not (is_horizontal or is_square or is_vertical):
        return False, f"Invalid aspect ratio {aspect_ratio:.2f}"
    
    # Check size
    if is_horizontal:
        if w < 50 or h < 10 or w > 1000 or h > 400:
            return False, "Invalid size (horizontal)"
    elif is_square:
        if w < 30 or h < 20 or w > 800 or h > 600:
            return False, "Invalid size (square)"
    else:  # vertical
        if w < 15 or h < 50 or w > 400 or h > 1000:
            return False, "Invalid size (vertical)"
    
    # Color analysis - reject pure red/yellow
    hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    
    # Red light
    red_mask = ((h_channel < 10) | (h_channel > 170)) & (s_channel > 100) & (v_channel > 100)
    red_ratio = np.sum(red_mask) / (w * h)
    if red_ratio > 0.5:
        return False, "Too much red"
    
    # Yellow light
    yellow_mask = (h_channel > 20) & (h_channel < 35) & (s_channel > 100) & (v_channel > 100)
    yellow_ratio = np.sum(yellow_mask) / (w * h)
    if yellow_ratio > 0.5:
        return False, "Too much yellow"
    
    # Brightness
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 30 or mean_brightness > 240:
        return False, "Extreme brightness"
    
    # Text pattern - edges
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (w * h)
    if edge_ratio < 0.05:
        return False, "Too few edges"
    
    return True, "Valid plate"

def smooth_recognition(current_text):
    """
    Smooth recognition using majority voting
    """
    if not current_text or current_text == 'UNKNOWN':
        return None
    
    plate_text_buffer.append(current_text)
    
    if len(plate_text_buffer) < 3:
        return None
    
    recent = list(plate_text_buffer)
    text_counts = {}
    for t in recent:
        if t and t != 'UNKNOWN':
            text_counts[t] = text_counts.get(t, 0) + 1
    
    if not text_counts:
        return None
    
    most_common = max(text_counts, key=text_counts.get)
    count = text_counts[most_common]
    
    if count >= 3:
        return most_common
    
    return None

# Main loop
frame_count = 0
stable_plate = None
stable_confidence = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Display frame
    frame_display = frame.copy()
    
    # YOLO detection
    results = model(frame, conf=0.25)
    
    best_detection = None
    best_score = 0
    all_detections = []
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        
        if len(boxes) == 0:
            continue
        
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Validate
            plate_roi = frame_display[y1:y2, x1:x2]
            if plate_roi.size == 0:
                continue
                
            is_valid, reason = is_valid_plate_roi(plate_roi, x1, y1, x2, y2)
            
            all_detections.append({
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'roi': plate_roi,
                'is_valid': is_valid
            })
            
            if not is_valid:
                continue
            
            # Score
            plate_area = (x2 - x1) * (y2 - y1)
            size_score = min(1.0, plate_area / 15000)
            combined_score = (conf * 0.7 + size_score * 0.3)
            
            if combined_score > best_score:
                best_score = combined_score
                best_detection = {
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'roi': plate_roi,
                    'is_valid': is_valid
                }
    
    # Draw invalid detections
    for det in all_detections:
        if not det['is_valid']:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 165, 255), 1)
    
    # Process best detection
    if best_detection:
        x1, y1, x2, y2 = best_detection['bbox']
        plate_roi = best_detection['roi']
        
        print(f"\n[FRAME {frame_count}] Detected at ({x1},{y1})-({x2},{y2})")
        
        # Resize for OCR
        h, w = plate_roi.shape[:2]
        if w < 250:
            scale = max(1.0, 250.0 / w)
            plate_roi_enhanced = cv2.resize(plate_roi, 
                                           (int(w * scale), int(h * scale)),
                                           interpolation=cv2.INTER_LINEAR)
        else:
            plate_roi_enhanced = plate_roi.copy()
        
        # OCR recognition
        ocr_result = ocr.recognize_and_format(plate_roi_enhanced)
        
        plate_text = ocr_result.get('formatted_text') or ocr_result.get('raw_ocr', 'UNKNOWN')
        confidence = ocr_result.get('confidence', 0.0)
        is_valid = ocr_result.get('success', False)
        
        print(f"  OCR: '{plate_text}' ({confidence:.1%})")
        
        # Temporal smoothing
        smoothed = smooth_recognition(plate_text if is_valid else 'UNKNOWN')
        
        if smoothed:
            stable_plate = smoothed
            stable_confidence = confidence
            detection_frames = 15
        
        # Draw bounding box
        color = (0, 255, 0) if is_valid else (0, 165, 255)
        thickness = 3 if is_valid else 2
        cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, thickness)
        
        # Draw text
        plate_display_text = plate_text if plate_text and plate_text != 'UNKNOWN' else 'DETECTED'
        text_size = cv2.getTextSize(f"BIEN SO: {plate_display_text}", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = max(10, x1)
        text_y = max(150, y1 - 50)
        
        cv2.rectangle(frame_display, 
                     (text_x - 5, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 5, text_y + 10),
                     (0, 0, 0), -1)
        
        cv2.putText(frame_display, f"BIEN SO: {plate_display_text}", (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        cv2.putText(frame_display, f"OCR:{confidence:.1%} YOLO:{best_detection['conf']:.2f}", 
                   (text_x, text_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Show stable result
    if stable_plate and detection_frames > 0:
        cv2.rectangle(frame_display, (5, 30), (400, 100), (0, 255, 0), 2)
        cv2.putText(frame_display, f"BIEN SO: {stable_plate}", (15, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        detection_frames -= 1
    
    # Info panel
    cv2.putText(frame_display, "License Plate Recognition", (10, frame_display.shape[0] - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    cv2.putText(frame_display, f"Frame: {frame_count} | Press Q to quit", 
               (10, frame_display.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Display
    cv2.imshow("License Plate Recognition", frame_display)
    
    # Exit
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Done")

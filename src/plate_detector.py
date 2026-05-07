"""
Plate detector module using Haar Cascade and YOLO
"""
import cv2
import numpy as np
import config
from src import utils


class HaarCascadePlateDetector:
    """Detect license plates using Haar Cascade classifier"""
    
    def __init__(self, cascade_path=config.CASCADE_PATH):
        """
        Initialize detector
        Args:
            cascade_path: Path to Haar Cascade XML file
        """
        try:
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if self.cascade.empty():
                utils.log_message(f"Warning: Cascade file not found at {cascade_path}", 'WARNING')
                self.cascade = None
        except Exception:
            self.cascade = None
            utils.log_message("Haar Cascade not available", 'WARNING')
    
    def detect(self, image, scale_factor=1.3, min_neighbors=5):
        """
        Detect license plates
        Args:
            image: Input image (BGR)
            scale_factor: Haar cascade scale factor
            min_neighbors: Cascade classifier parameter
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if self.cascade is None:
            return []
        
        gray = utils.convert_to_grayscale(image)
        
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(50, 15),
            maxSize=(500, 150)
        )
        
        return detections.tolist() if len(detections) > 0 else []


class DynamicPlateDetector:
    """Detect plates using morphological operations"""
    
    def __init__(self):
        pass
    
    def detect(self, image, min_area=config.MIN_CONTOUR_AREA, max_area=config.MAX_CONTOUR_AREA):
        """
        Detect license plates using morphological analysis
        Args:
            image: Input image
            min_area, max_area: Area constraints
        Returns:
            List of bounding boxes
        """
        # Convert to grayscale
        gray = utils.convert_to_grayscale(image)
        
        # Apply bilateral filter for edge preservation
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply threshold
        _, binary = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        plates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if min_area <= area <= max_area and 2.5 <= aspect_ratio <= 5.0:
                plates.append((x, y, w, h))
        
        return plates

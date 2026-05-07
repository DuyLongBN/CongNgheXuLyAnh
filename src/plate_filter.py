"""
Advanced plate detection filtering module
Mô-đun lọc phát hiện biển số nâng cao
Filters out false positives like motorcycle lights, reflectors, etc.
"""
import cv2
import numpy as np


class PlateValidator:
    """Validate if detected region is truly a license plate"""
    
    def __init__(self):
        """Initialize validator with default parameters"""
        # Size constraints (pixels) - support both orientations
        self.min_width = 15      # Min width (for vertical plates)
        self.max_width = 400     # Max width
        self.min_height = 15     # Min height (for horizontal plates)
        self.max_height = 400    # Max height (for vertical plates)
        
        # Aspect ratio ranges (width/height)
        # Horizontal: 2.5-6.0 (wide plates, 1-row)
        # Square: 0.5-2.5 (2-row plates, most common in Vietnam - many sizes)
        # Vertical: 0.15-0.5 (tall plates)
        self.horizontal_aspect_min = 2.5
        self.horizontal_aspect_max = 6.0
        self.square_aspect_min = 0.5
        self.square_aspect_max = 2.5
        self.vertical_aspect_min = 0.15
        self.vertical_aspect_max = 0.5
        
        # Color constraints (HSV)
        self.red_h_ranges = [(0, 10), (170, 180)]  # Red hue
        self.red_s_min = 100  # Red saturation threshold
        self.red_v_min = 100  # Red value threshold
        self.red_max_ratio = 0.5  # Max 50% red allowed
        
        self.yellow_h_min = 20  # Yellow hue
        self.yellow_h_max = 35
        self.yellow_s_min = 100
        self.yellow_v_min = 100
        self.yellow_max_ratio = 0.5  # Max 50% yellow allowed
        
        # Brightness constraints
        self.min_brightness = 30
        self.max_brightness = 240
        
        # Edge ratio constraints (for text detection)
        self.min_edge_ratio = 0.05  # At least 5% edges
        self.max_edge_ratio = 0.80  # At most 80% edges
    
    def validate(self, roi, verbose=False):
        """
        Validate if ROI is a license plate (horizontal, vertical, or square/2-row)
        Args:
            roi: Region of interest (numpy array)
            verbose: If True, return detailed reason
        Returns:
            Tuple: (is_valid: bool, reason: str)
        """
        h, w = roi.shape[:2]
        
        # Check 1: Size constraints (basic)
        if w < 15 or w > 400 or h < 15 or h > 400:
            return False, f"Size {w}x{h}px outside valid range"
        
        # Check 2: Aspect ratio - accept ALL three orientations
        aspect_ratio = w / h if h > 0 else 0
        is_horizontal = self.horizontal_aspect_min <= aspect_ratio <= self.horizontal_aspect_max
        is_square = self.square_aspect_min <= aspect_ratio <= self.square_aspect_max
        is_vertical = self.vertical_aspect_min <= aspect_ratio < self.square_aspect_min
        
        if not (is_horizontal or is_square or is_vertical):
            return False, f"Aspect ratio {aspect_ratio:.2f} invalid"
        
        # Check size based on orientation
        if is_horizontal:
            if w < 50 or h < 10 or w > 1000 or h > 400:
                return False, f"Horizontal plate size invalid"
        elif is_square:
            if w < 30 or h < 20 or w > 800 or h > 600:
                return False, f"Square plate size invalid"
        else:  # vertical
            if w < 15 or h < 50 or w > 400 or h > 1000:
                return False, f"Vertical plate size invalid"
        
        # Check 3: Color analysis - reject lights
        is_light, light_reason = self._check_light_color(roi)
        if is_light:
            return False, light_reason
        
        # Check 4: Brightness
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        mean_brightness = np.mean(gray)
        
        if mean_brightness < self.min_brightness or mean_brightness > self.max_brightness:
            return False, f"Brightness {mean_brightness:.0f} invalid"
        
        # Check 5: Edge/text pattern
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (w * h)
        
        if edge_ratio < self.min_edge_ratio:
            return False, f"Edge ratio {edge_ratio:.3f} too low"
        
        if edge_ratio > self.max_edge_ratio:
            return False, f"Edge ratio {edge_ratio:.3f} too high"
        
        # Check 6: Contrast (should have good text contrast)
        std_brightness = np.std(gray)
        if std_brightness < 10:
            return False, f"Contrast too low"
        
        # All checks passed
        return True, "Valid plate"
    
    def _check_light_color(self, roi):
        """
        Check if ROI is mostly a light (red/yellow)
        Returns: (is_light: bool, reason: str)
        """
        # Grayscale images cannot be analyzed for color — skip color check
        if len(roi.shape) != 3:
            return False, "Color OK (grayscale)"
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        h, w = roi.shape[:2]
        total_pixels = h * w
        
        # Red light detection
        red_mask = np.zeros_like(h_channel, dtype=bool)
        for h_min, h_max in self.red_h_ranges:
            red_mask |= (h_channel >= h_min) & (h_channel <= h_max)
        red_mask &= (s_channel > self.red_s_min) & (v_channel > self.red_v_min)
        
        red_ratio = np.sum(red_mask) / total_pixels
        if red_ratio > self.red_max_ratio:
            return True, f"Too much red color ({red_ratio:.1%})"
        
        # Yellow light detection
        yellow_mask = (h_channel >= self.yellow_h_min) & (h_channel <= self.yellow_h_max)
        yellow_mask &= (s_channel > self.yellow_s_min) & (v_channel > self.yellow_v_min)
        
        yellow_ratio = np.sum(yellow_mask) / total_pixels
        if yellow_ratio > self.yellow_max_ratio:
            return True, f"Too much yellow color ({yellow_ratio:.1%})"
        
        return False, "Color OK"
    
    def get_validation_details(self, roi):
        """
        Get detailed validation information
        Returns dict with all check results
        """
        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        details = {
            'width': w,
            'height': h,
            'aspect_ratio': w / h if h > 0 else 0,
            'brightness_mean': float(np.mean(gray)),
            'brightness_std': float(np.std(gray)),
            'edge_ratio': 0.0
        }
        
        # Calculate edge ratio
        edges = cv2.Canny(gray, 50, 150)
        details['edge_ratio'] = float(np.sum(edges > 0) / (w * h))
        
        # Color analysis — only for color images
        if len(roi.shape) == 3:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
        else:
            # Grayscale — no color info available
            details['red_ratio'] = 0.0
            details['yellow_ratio'] = 0.0
            return details
        
        red_mask = np.zeros_like(h_channel, dtype=bool)
        for h_min, h_max in self.red_h_ranges:
            red_mask |= (h_channel >= h_min) & (h_channel <= h_max)
        red_mask &= (s_channel > self.red_s_min) & (v_channel > self.red_v_min)
        details['red_ratio'] = float(np.sum(red_mask) / (w * h))
        
        yellow_mask = (h_channel >= self.yellow_h_min) & (h_channel <= self.yellow_h_max)
        yellow_mask &= (s_channel > self.yellow_s_min) & (v_channel > self.yellow_v_min)
        details['yellow_ratio'] = float(np.sum(yellow_mask) / (w * h))
        
        return details


class TemporalFilter:
    """Smooth recognition results across multiple frames"""
    
    def __init__(self, window_size=5):
        """
        Initialize temporal filter
        Args:
            window_size: Number of frames to track
        """
        self.window_size = window_size
        self.text_buffer = []
        self.confidence_buffer = []
    
    def add_detection(self, plate_text, confidence):
        """Add new detection result"""
        self.text_buffer.append(plate_text)
        self.confidence_buffer.append(confidence)
        
        # Keep only last N frames
        if len(self.text_buffer) > self.window_size:
            self.text_buffer.pop(0)
            self.confidence_buffer.pop(0)
    
    def get_stable_result(self, min_agreement=0.6):
        """
        Get stable result if multiple frames agree
        Args:
            min_agreement: Minimum ratio of frames that must agree (0-1)
        Returns:
            Tuple: (text, confidence, is_stable)
        """
        if not self.text_buffer:
            return None, 0.0, False
        
        # Count occurrences of each text
        text_counts = {}
        for text in self.text_buffer:
            text_counts[text] = text_counts.get(text, 0) + 1
        
        # Find most common text
        most_common = max(text_counts, key=text_counts.get)
        agreement_ratio = text_counts[most_common] / len(self.text_buffer)
        
        # Get average confidence for this text
        indices = [i for i, t in enumerate(self.text_buffer) if t == most_common]
        avg_confidence = np.mean([self.confidence_buffer[i] for i in indices])
        
        is_stable = agreement_ratio >= min_agreement
        
        return most_common, avg_confidence, is_stable
    
    def clear(self):
        """Clear buffers"""
        self.text_buffer.clear()
        self.confidence_buffer.clear()


class FrameAnalyzer:
    """Analyze frame characteristics for better detection"""
    
    @staticmethod
    def get_frame_brightness(frame):
        """Get overall frame brightness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    @staticmethod
    def get_frame_contrast(frame):
        """Get overall frame contrast"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.std(gray)
    
    @staticmethod
    def is_low_light(frame, threshold=80):
        """Check if frame is in low light conditions"""
        brightness = FrameAnalyzer.get_frame_brightness(frame)
        return brightness < threshold
    
    @staticmethod
    def is_high_glare(frame, threshold=200):
        """Check if frame has high glare/overexposure"""
        brightness = FrameAnalyzer.get_frame_brightness(frame)
        return brightness > threshold
    
    @staticmethod
    def detect_motion(frame, prev_frame=None, motion_threshold=30):
        """
        Detect motion in frame (helps ignore static objects like lights)
        Args:
            frame: Current frame
            prev_frame: Previous frame (if available)
            motion_threshold: Motion detection threshold
        Returns:
            float: Motion amount (0-1)
        """
        if prev_frame is None:
            return 0.0
        
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray_curr, gray_prev)
        motion = np.mean(diff) / 255.0
        
        return motion

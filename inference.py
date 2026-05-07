"""
Inference module for license plate recognition
Mô-đun suy diễn cho nhận dạng biển số xe
"""
import cv2
import numpy as np
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
from pathlib import Path
import easyocr
import config
from src import utils, image_processing, vietnamese_ocr
from yolo_plate_detector import YOLOPlateDetector


class LicensePlateRecognizer:
    """End-to-end license plate recognition system
    Hệ thống nhận dạng biển số xe hoàn chỉnh
    """
    
    def __init__(self, model_path=None, use_vietnamese_ocr=True, use_yolo=True):
        """
        Initialize recognizer
        Args:
            model_path: Path to trained model
            use_vietnamese_ocr: Whether to use Vietnamese OCR optimization
            use_yolo: Whether to use YOLO for plate detection
        """
        self.preprocessor = image_processing.ImagePreprocessor()
        self.plate_extractor = image_processing.PlateExtractor()
        self.char_segmentation = image_processing.CharacterSegmentation()
        
        # Initialize YOLO plate detector
        self.yolo_detector = None
        self.use_yolo = use_yolo
        if use_yolo:
            try:
                self.yolo_detector = YOLOPlateDetector(model_path='models/best.pt', conf_threshold=0.3)
                if self.yolo_detector.model is not None:
                    utils.log_message("YOLO plate detector initialized", 'INFO')
                    self.use_yolo = True
                else:
                    utils.log_message("YOLO model failed to load, falling back to contour detection", 'WARNING')
                    self.use_yolo = False
            except Exception as e:
                utils.log_message(f"Error initializing YOLO: {e}", 'WARNING')
                self.use_yolo = False
        
        # Load trained model if provided and TensorFlow is available
        self.model = None
        if TF_AVAILABLE and model_path and Path(model_path).exists():
            try:
                self.model = tf.keras.models.load_model(model_path)
                utils.log_message(f"Loaded model from {model_path}")
            except Exception as e:
                utils.log_message(f"Failed to load model: {e}", 'WARNING')
                self.model = None
        else:
            if not TF_AVAILABLE:
                utils.log_message("TensorFlow not available, using EasyOCR for recognition", 'INFO')
            else:
                utils.log_message("No trained model provided, using EasyOCR instead")
        
        # Initialize Vietnamese OCR if enabled
        self.use_vietnamese_ocr = use_vietnamese_ocr and config.VIETNAM_OCR_ENABLED
        self.vietnam_ocr = None
        self.ocr = None
        
        if self.use_vietnamese_ocr:
            self.vietnam_ocr = vietnamese_ocr.VietnameseOCREngine(use_gpu=False)
            utils.log_message("Vietnamese OCR engine initialized")
        else:
            # Fallback to standard EasyOCR
            self.ocr = easyocr.Reader(['vi', 'en'], gpu=False)
        
        self.char_to_idx = {char: idx for idx, char in enumerate(config.ALLOWED_CHARS)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
    
    def recognize_plate(self, image_path):
        """
        Recognize license plate from image
        Args:
            image_path: Path to vehicle image
        Returns:
            Dictionary with detection and recognition results
        """
        # Load image
        image = utils.load_image(str(image_path))
        
        # Detect plate regions - use YOLO if available
        if self.use_yolo and self.yolo_detector is not None:
            utils.log_message("Using YOLO for plate detection", 'DEBUG')
            detections = self.yolo_detector.detect(image)
            plates = self.yolo_detector.extract_plates(image, detections)
            utils.log_message(f"YOLO detected {len(plates)} plates", 'INFO')
        else:
            utils.log_message("Using contour-based plate detection", 'DEBUG')
            plates = self.plate_extractor.extract_all_plates(image)
            utils.log_message(f"Contour detection found {len(plates)} plates", 'INFO')
        
        results = {
            'image_path': str(image_path),
            'plates': []
        }
        
        for plate_data in plates:
            plate_img = plate_data['image']
            
            # Recognize using neural network or OCR
            if self.model is not None:
                recognition = self._recognize_with_model(plate_img)
            elif self.use_vietnamese_ocr:
                recognition = self._recognize_with_vietnamese_ocr(plate_img)
            else:
                recognition = self._recognize_with_ocr(plate_img)
            
            results['plates'].append({
                'bbox': plate_data['bbox'],
                'recognition': recognition
            })
        
        return results
    
    def _recognize_with_model(self, plate_image):
        """Recognize plate using trained neural network"""
        try:
            # Preprocess
            processed = self.preprocessor.preprocess(plate_image)
            
            # Segment characters
            characters = self.char_segmentation.segment_characters(processed)
            
            plate_number = ""
            confidences = []
            
            for char_data in characters:
                char_img = char_data['image']
                
                # Resize to model input size
                resized = self.char_segmentation.resize_character(char_img)
                resized = utils.normalize_image(resized)
                
                # Predict
                predictions = self.model.predict(np.expand_dims(resized, 0), verbose=0)
                char_idx = np.argmax(predictions[0])
                confidence = predictions[0][char_idx]
                
                if confidence > config.RECOGNITION_CONFIDENCE:
                    char = self.idx_to_char.get(char_idx, '?')
                    plate_number += char
                    confidences.append(float(confidence))
            
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'method': 'neural_network',
                'plate_number': plate_number,
                'confidence': avg_confidence
            }
        
        except Exception as e:
            utils.log_message(f"Error with neural network recognition: {e}", 'WARNING')
            if self.use_vietnamese_ocr:
                return self._recognize_with_vietnamese_ocr(plate_image)
            else:
                return self._recognize_with_ocr(plate_image)
    
    def _recognize_with_vietnamese_ocr(self, plate_image):
        """Recognize plate using Vietnamese-optimized OCR"""
        try:
            if self.vietnam_ocr is None:
                utils.log_message("Vietnamese OCR not initialized, falling back to EasyOCR", 'WARNING')
                return self._recognize_with_ocr(plate_image)
            
            # Use Vietnamese OCR engine
            result = self.vietnam_ocr.recognize_and_format(plate_image)
            
            return {
                'method': 'vietnamese_ocr',
                'plate_number': result.get('formatted_text', result.get('raw_ocr', '')),
                'raw_text': result.get('raw_ocr', ''),
                'confidence': result.get('confidence', 0.0),
                'validation': result.get('validation', {}),
                'is_valid': result.get('success', False)
            }
        
        except Exception as e:
            utils.log_message(f"Error with Vietnamese OCR: {e}", 'WARNING')
            return self._recognize_with_ocr(plate_image)
    
    def _recognize_with_ocr(self, plate_image):
        """Recognize plate using EasyOCR (fallback)"""
        try:
            # Lazily initialize EasyOCR if not yet created
            if self.ocr is None:
                self.ocr = easyocr.Reader(['vi', 'en'], gpu=False)
            
            # Preprocess for OCR
            gray = utils.convert_to_grayscale(plate_image)
            gray = cv2.resize(gray, (640, 320))
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Run OCR
            results = self.ocr.readtext(enhanced, detail=1)
            
            plate_number = ""
            confidences = []
            
            for detection in results:
                text = detection[1]
                confidence = detection[2]
                
                if confidence > config.RECOGNITION_CONFIDENCE:
                    plate_number += text
                    confidences.append(confidence)
            
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'method': 'easyocr',
                'plate_number': plate_number,
                'confidence': avg_confidence
            }
        
        except Exception as e:
            utils.log_message(f"Error with OCR recognition: {e}", 'ERROR')
            return {
                'method': 'error',
                'plate_number': 'UNKNOWN',
                'confidence': 0.0
            }
    
    def recognize_multiple(self, image_folder):
        """Recognize plates from multiple images"""
        image_files = utils.get_image_files(str(image_folder))
        
        all_results = []
        
        for image_file in image_files:
            utils.log_message(f"Processing {image_file.name}...")
            result = self.recognize_plate(str(image_file))
            all_results.append(result)
        
        return all_results
    
    def visualize_results(self, image_path, output_path=None):
        """Visualize recognition results"""
        image = utils.load_image(str(image_path))
        result = self.recognize_plate(str(image_path))
        
        # Draw detected plates and results
        for plate_info in result['plates']:
            x, y, w, h = plate_info['bbox']
            plate_number = plate_info['recognition']['plate_number']
            confidence = plate_info['recognition']['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw text
            text = f"{plate_number} ({confidence:.2f})"
            cv2.putText(image, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save or display
        if output_path:
            utils.save_image(str(output_path), image)
            utils.log_message(f"Saved result to {output_path}")
        else:
            utils.display_image("License Plate Recognition", image)
        
        return image


def main():
    """Main inference script"""
    utils.create_directories()
    
    # Example usage
    model_path = Path(config.MODELS_PATH) / 'cnn_model.h5'
    
    # Initialize recognizer
    recognizer = LicensePlateRecognizer(model_path=model_path if model_path.exists() else None)
    
    # Process sample image
    test_image = 'test_image.jpg'  # Replace with actual image path
    
    if Path(test_image).exists():
        utils.log_message(f"Processing {test_image}...")
        result = recognizer.recognize_plate(test_image)
        
        for plate_info in result['plates']:
            plate_num = plate_info['recognition']['plate_number']
            confidence = plate_info['recognition']['confidence']
            utils.log_message(f"Recognized: {plate_num} (Confidence: {confidence:.4f})")
        
        # Visualize
        recognizer.visualize_results(test_image, output_path='result.jpg')
    else:
        utils.log_message(f"Test image {test_image} not found", 'WARNING')


if __name__ == '__main__':
    main()

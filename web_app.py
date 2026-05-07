"""
Web Interface for License Plate Recognition
Giao diện web cho nhận dạng biển số xe
"""
import os
import cv2
import base64
import threading
import queue
import traceback
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Import existing modules
from inference import LicensePlateRecognizer
from video_recognition import VideoPlateRecognizer
from src import utils

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global variables for real-time processing
webcam_thread = None
is_streaming = False
stream_queue = queue.Queue(maxsize=2)
recognizer = LicensePlateRecognizer()
video_recognizer = VideoPlateRecognizer()

# Storage for detection history
detection_history = []
MAX_HISTORY = 100


def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    # buffer is already bytes from cv2.imencode, no need to call tobytes()
    return base64.b64encode(buffer).decode('utf-8')


def draw_detections(frame, detections):
    """Draw bounding boxes and plate numbers on frame"""
    frame_copy = frame.copy()
    
    for detection in detections:
        bbox = detection.get('bbox', (0, 0, 0, 0))
        plate_number = detection.get('plate_number', 'UNKNOWN')
        confidence = detection.get('confidence', 0.0)
        
        if bbox and bbox != (0, 0, 0, 0):
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw plate number with background
            text = f"{plate_number} ({confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = x1
            text_y = max(y1 - 10, 20)
            
            # Background rectangle for text
            cv2.rectangle(frame_copy, 
                         (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5),
                         (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame_copy, text, (text_x, text_y), 
                       font, font_scale, (0, 0, 0), thickness)
    
    return frame_copy


def webcam_worker():
    """Worker thread for webcam capture and processing"""
    global is_streaming
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    while is_streaming:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 2nd frame for performance
        if frame_count % 2 == 0:
            try:
                # Save temp frame
                temp_path = os.path.join(UPLOAD_FOLDER, '_temp_frame.jpg')
                cv2.imwrite(temp_path, frame)
                
                # Recognize plates
                result = recognizer.recognize_plate(temp_path)
                detections = result.get('plates', [])
                
                # Flatten detections
                flat_detections = []
                for plate_info in detections:
                    recognition = plate_info.get('recognition', {})
                    flat_detections.append({
                        'bbox': plate_info.get('bbox', (0, 0, 0, 0)),
                        'plate_number': recognition.get('plate_number', 'UNKNOWN'),
                        'confidence': recognition.get('confidence', 0.0)
                    })
                
                # Add to history
                if flat_detections:
                    for det in flat_detections:
                        detection_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'plate_number': det['plate_number'],
                            'confidence': det['confidence']
                        })
                    if len(detection_history) > MAX_HISTORY:
                        detection_history.pop(0)
                
                # Draw detections
                frame_with_boxes = draw_detections(frame, flat_detections)
                
                # Convert to base64 and queue
                frame_b64 = frame_to_base64(frame_with_boxes)
                
                try:
                    stream_queue.put_nowait({
                        'frame': frame_b64,
                        'detections': flat_detections,
                        'timestamp': datetime.now().isoformat()
                    })
                except queue.Full:
                    pass  # Skip frame if queue is full
                
            except Exception as e:
                utils.log_message(f"Error processing frame: {e}", 'ERROR')
    
    cap.release()


# ============ ROUTES ============

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/start-webcam', methods=['POST'])
def start_webcam():
    """Start webcam streaming"""
    global webcam_thread, is_streaming
    
    if is_streaming:
        return jsonify({'status': 'already_running'})
    
    is_streaming = True
    webcam_thread = threading.Thread(target=webcam_worker, daemon=True)
    webcam_thread.start()
    
    return jsonify({'status': 'started'})


@app.route('/api/stop-webcam', methods=['POST'])
def stop_webcam():
    """Stop webcam streaming"""
    global is_streaming
    is_streaming = False
    return jsonify({'status': 'stopped'})


@app.route('/api/stream')
def stream_frames():
    """Stream frames with Server-Sent Events"""
    def event_generator():
        while is_streaming:
            try:
                data = stream_queue.get(timeout=1)
                yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
            except Exception as e:
                utils.log_message(f"Stream error: {e}", 'ERROR')
    
    return app.response_class(event_generator(), mimetype='text/event-stream')


@app.route('/api/process-image', methods=['POST'])
def process_image():
    """Process single image"""
    utils.log_message(f"Image upload endpoint called. Files: {request.files.keys()}", 'INFO')
    
    if 'image' not in request.files:
        utils.log_message("No image file in request", 'WARNING')
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        utils.log_message("Empty filename", 'WARNING')
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        utils.log_message(f"Invalid file extension: {file.filename}", 'WARNING')
        return jsonify({'error': f'Invalid image format. Allowed: {ALLOWED_IMAGE_EXTENSIONS}'}), 400
    
    try:
        # Save uploaded file
        filename = f"img_{int(datetime.now().timestamp() * 1000)}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        utils.log_message(f"Image saved to {filepath}", 'INFO')
        
        # Verify file exists and has content
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({'error': 'File upload failed or is empty'}), 400
        
        # Process image
        result = recognizer.recognize_plate(filepath)
        detections = result.get('plates', [])
        utils.log_message(f"Detected {len(detections)} plates", 'INFO')
        
        # Flatten detections
        flat_detections = []
        for plate_info in detections:
            recognition = plate_info.get('recognition', {})
            flat_detections.append({
                'bbox': plate_info.get('bbox', (0, 0, 0, 0)),
                'plate_number': recognition.get('plate_number', 'UNKNOWN'),
                'confidence': recognition.get('confidence', 0.0)
            })
        
        # Add to history
        for det in flat_detections:
            detection_history.append({
                'timestamp': datetime.now().isoformat(),
                'plate_number': det['plate_number'],
                'confidence': det['confidence']
            })
        if len(detection_history) > MAX_HISTORY:
            detection_history.pop(0)
        
        # Draw on image and save
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Failed to read image file'}), 400
        
        image_with_boxes = draw_detections(image, flat_detections)
        result_filename = f"result_{int(datetime.now().timestamp() * 1000)}.jpg"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        cv2.imwrite(result_path, image_with_boxes)
        utils.log_message(f"Result image saved to {result_path}", 'INFO')
        
        return jsonify({
            'status': 'success',
            'detections': flat_detections,
            'result_image': result_path,
            'detection_count': len(flat_detections)
        })
    
    except Exception as e:
        utils.log_message(f"Image processing error: {str(e)}", 'ERROR')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """Upload and process video file"""
    utils.log_message(f"Video upload endpoint called. Files: {request.files.keys()}", 'INFO')
    
    if 'video' not in request.files:
        utils.log_message("No video file in request", 'WARNING')
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        utils.log_message("Empty video filename", 'WARNING')
        return jsonify({'error': 'No video selected'}), 400
    
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        utils.log_message(f"Invalid video format: {file_ext}", 'WARNING')
        return jsonify({'error': f'Invalid video format. Allowed: {ALLOWED_VIDEO_EXTENSIONS}'}), 400
    
    try:
        # Save uploaded file
        filename = f"vid_{int(datetime.now().timestamp() * 1000)}.{file_ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        utils.log_message(f"Video saved to {filepath}", 'INFO')
        
        # Verify file exists and has content
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({'error': 'File upload failed or is empty'}), 400
        
        # Process video
        result_filename = f"result_{int(datetime.now().timestamp() * 1000)}.mp4"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        utils.log_message(f"Processing video: {filepath}", 'INFO')
        detections = video_recognizer.process_video(
            filepath,
            output_path=result_path,
            confidence_threshold=0.5
        )
        
        # Count unique plates
        plate_numbers = set()
        for det in detections:
            if isinstance(det, dict) and 'plate_number' in det:
                plate_numbers.add(det['plate_number'])
        
        utils.log_message(f"Video processing complete. Found {len(plate_numbers)} unique plates", 'INFO')
        
        return jsonify({
            'status': 'success',
            'result_video': result_path,
            'total_detections': len(detections),
            'unique_plates': list(plate_numbers),
            'unique_count': len(plate_numbers)
        })
    
    except Exception as e:
        utils.log_message(f"Video processing error: {str(e)}", 'ERROR')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def get_history():
    """Get detection history"""
    return jsonify(detection_history[-50:])  # Last 50 detections


@app.route('/api/stats')
def get_stats():
    """Get statistics"""
    if not detection_history:
        return jsonify({
            'total_detections': 0,
            'unique_plates': 0,
            'detection_rate': 0
        })
    
    unique_plates = set(d['plate_number'] for d in detection_history)
    
    return jsonify({
        'total_detections': len(detection_history),
        'unique_plates': list(unique_plates),
        'unique_count': len(unique_plates),
        'average_confidence': np.mean([d['confidence'] for d in detection_history if 'confidence' in d])
    })


@app.route('/results/<filename>')
def serve_result(filename):
    """Serve result files from results folder"""
    try:
        file_path = os.path.join(RESULTS_FOLDER, filename)
        # Security check: ensure file is in results folder
        if not os.path.abspath(file_path).startswith(os.path.abspath(RESULTS_FOLDER)):
            return jsonify({'error': 'Invalid file path'}), 403
        
        if not os.path.exists(file_path):
            utils.log_message(f"File not found: {file_path}", 'WARNING')
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, mimetype='image/jpeg')
    except Exception as e:
        utils.log_message(f"Error serving file {filename}: {str(e)}", 'ERROR')
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check"""
    return jsonify({'status': 'healthy'})


@app.route('/api/test-system', methods=['GET'])
def test_system():
    """Test if system components are working"""
    result = {
        'status': 'ok',
        'components': {}
    }
    
    try:
        # Check recognizer
        result['components']['recognizer'] = {
            'status': 'ok',
            'type': type(recognizer).__name__
        }
    except Exception as e:
        result['components']['recognizer'] = {'status': 'error', 'error': str(e)}
    
    try:
        # Check video recognizer
        result['components']['video_recognizer'] = {
            'status': 'ok',
            'type': type(video_recognizer).__name__
        }
    except Exception as e:
        result['components']['video_recognizer'] = {'status': 'error', 'error': str(e)}
    
    # Check folders
    result['folders'] = {
        'uploads': os.path.exists(UPLOAD_FOLDER),
        'results': os.path.exists(RESULTS_FOLDER)
    }
    
    # Check write permissions
    try:
        test_file = os.path.join(UPLOAD_FOLDER, '.test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        result['write_permission'] = 'ok'
    except Exception as e:
        result['write_permission'] = f'error: {str(e)}'
    
    return jsonify(result)


if __name__ == '__main__':
    print("=" * 50)
    print("License Plate Recognition Web Interface")
    print("=" * 50)
    print("Open browser at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

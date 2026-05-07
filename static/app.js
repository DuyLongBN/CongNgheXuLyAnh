/**
 * License Plate Recognition - Web Interface JavaScript
 * Giao diện web cho nhận dạng biển số xe
 */

// Global variables
let eventSource = null;
let frameCount = 0;
let lastFrameTime = Date.now();
let fpsCounter = 0;
let currentDetections = [];
let plateCache = new Set();

// Constants
const API_BASE = '/api';
const NOTIFICATION_DURATION = 3000;

// ============================================
// Utility Functions
// ============================================

function showNotification(message, type = 'success') {
    const modal = document.getElementById('notificationModal');
    modal.textContent = message;
    modal.className = `notification-modal ${type}`;
    modal.classList.remove('hidden');

    setTimeout(() => {
        modal.classList.add('hidden');
    }, NOTIFICATION_DURATION);
}

function showLoading(show = true) {
    const spinner = document.getElementById('loadingSpinner');
    if (show) {
        spinner.classList.remove('hidden');
    } else {
        spinner.classList.add('hidden');
    }
}

function updateStatusIndicator(active) {
    const dot = document.querySelector('.status-dot');
    const text = document.getElementById('status-text');
    const status = document.getElementById('systemStatus');

    if (active) {
        dot.classList.add('active');
        text.textContent = 'Live';
        status.textContent = '🟢 Đang phát trực tiếp...';
    } else {
        dot.classList.remove('active');
        text.textContent = 'Offline';
        status.textContent = '🔴 Offline - Nhấn [Bắt Đầu] để khởi động';
    }
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('vi-VN');
}

function formatConfidence(conf) {
    return (conf * 100).toFixed(1) + '%';
}

// ============================================
// FPS Counter
// ============================================

function updateFPS() {
    const now = Date.now();
    const delta = (now - lastFrameTime) / 1000;
    
    if (delta >= 1) {
        document.getElementById('fpsCounter').textContent = `FPS: ${fpsCounter.toFixed(0)}`;
        fpsCounter = 0;
        lastFrameTime = now;
    } else {
        fpsCounter++;
    }
}

// ============================================
// Webcam Functions
// ============================================

function startWebcam() {
    showLoading(true);

    fetch(`${API_BASE}/start-webcam`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'started' || data.status === 'already_running') {
            // Show video frame, hide image result
            const videoFrame = document.getElementById('videoCanvas');
            videoFrame.parentElement.style.display = 'flex';
            videoFrame.classList.add('active');
            document.getElementById('imageResultContainer').classList.add('hidden');
            
            startStreamListener();
            updateStatusIndicator(true);
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            showNotification('✓ Webcam đã bắt đầu', 'success');
        }
        showLoading(false);
    })
    .catch(error => {
        console.error('Error starting webcam:', error);
        showNotification('✗ Lỗi bắt đầu webcam', 'error');
        showLoading(false);
    });
}

function stopWebcam() {
    fetch(`${API_BASE}/stop-webcam`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.json())
    .then(data => {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        updateStatusIndicator(false);
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        showNotification('✓ Webcam đã dừng', 'success');
    })
    .catch(error => {
        console.error('Error stopping webcam:', error);
        showNotification('✗ Lỗi dừng webcam', 'error');
    });
}

function startStreamListener() {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource(`${API_BASE}/stream`);

    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'ping') {
                return; // Keep-alive ping
            }

            if (data.frame) {
                // Display frame
                const canvas = document.getElementById('videoCanvas');
                const img = document.createElement('img');
                img.src = 'data:image/jpeg;base64,' + data.frame;
                img.onload = function() {
                    canvas.innerHTML = '';
                    canvas.appendChild(img);
                    updateFPS();
                };

                // Update timestamp
                document.getElementById('timestamp').textContent = formatTime(data.timestamp);

                // Update detections
                if (data.detections && data.detections.length > 0) {
                    currentDetections = data.detections;
                    updateResultsList(data.detections);
                }
            }
        } catch (error) {
            console.error('Error parsing stream data:', error);
        }
    };

    eventSource.onerror = function(event) {
        console.error('Stream error:', event);
        if (eventSource) {
            eventSource.close();
        }
    };
}

// ============================================
// Image Processing
// ============================================

function processImage(event) {
    const file = event.target.files[0];
    if (!file) {
        showNotification('✗ Vui lòng chọn ảnh', 'error');
        return;
    }

    console.log('Processing image:', file.name, file.size);
    showLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    fetch(`${API_BASE}/process-image`, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Response data:', data);
        if (data.status === 'success') {
            displayImageResult(data);
            updateResultsList(data.detections);
            showNotification(`✓ Phát hiện ${data.detection_count} biển số`, 'success');
        } else if (data.error) {
            showNotification(`✗ Lỗi: ${data.error}`, 'error');
        } else {
            showNotification('✗ Lỗi không xác định', 'error');
        }
        showLoading(false);
    })
    .catch(error => {
        console.error('Error processing image:', error);
        showNotification(`✗ Lỗi xử lý ảnh: ${error.message}`, 'error');
        showLoading(false);
    });

    // Reset input
    event.target.value = '';
}

function displayImageResult(data) {
    const container = document.getElementById('imageResultContainer');
    const img = document.getElementById('resultImage');
    const videoFrame = document.getElementById('videoCanvas');
    
    img.src = data.result_image;
    container.classList.remove('hidden');

    // Hide video frame container completely
    videoFrame.parentElement.style.display = 'none';
}

// ============================================
// Video Upload
// ============================================

function uploadVideo(event) {
    const file = event.target.files[0];
    if (!file) return;

    showLoading(true);
    const formData = new FormData();
    formData.append('video', file);

    const xhr = new XMLHttpRequest();

    // Progress tracking
    xhr.upload.addEventListener('progress', function(e) {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            const progressBar = document.getElementById('uploadProgressBar');
            progressBar.style.width = percentComplete + '%';
            document.getElementById('uploadProgress').classList.remove('hidden');
        }
    });

    xhr.addEventListener('loadstart', function() {
        document.getElementById('uploadProgress').classList.remove('hidden');
    });

    xhr.addEventListener('load', function() {
        console.log('Video upload complete. Status:', xhr.status);
        console.log('Response:', xhr.responseText);
        
        try {
            const data = JSON.parse(xhr.responseText);
            
            if (xhr.status === 200 && data.status === 'success') {
                showNotification(
                    `✓ Video xử lý xong: ${data.unique_count} biển số riêng`,
                    'success'
                );
                
                // Display results
                const resultsList = document.getElementById('resultsList');
                resultsList.innerHTML = `
                    <div class="detection-item">
                        <div style="font-weight: bold; color: var(--secondary-color);">Kết quả Video</div>
                        <div class="detection-confidence">
                            Tổng phát hiện: ${data.total_detections}
                        </div>
                        <div class="detection-confidence">
                            Biển số riêng: ${data.unique_count}
                        </div>
                        <div class="detection-confidence">
                            Video: <a href="${data.result_video}" target="_blank" style="color: var(--secondary-color);">Xem kết quả</a>
                        </div>
                    </div>
                `;
            } else if (data.error) {
                showNotification(`✗ Lỗi: ${data.error}`, 'error');
            } else {
                showNotification(`✗ Lỗi tải lên (Status: ${xhr.status})`, 'error');
            }
        } catch (parseError) {
            console.error('Error parsing response:', parseError);
            showNotification(`✗ Lỗi phân tích phản hồi: ${parseError.message}`, 'error');
        }
        
        showLoading(false);
        document.getElementById('uploadProgress').classList.add('hidden');
    });

    xhr.addEventListener('error', function() {
        console.error('XHR Error:', xhr.status, xhr.statusText);
        showNotification('✗ Lỗi kết nối', 'error');
        showLoading(false);
        document.getElementById('uploadProgress').classList.add('hidden');
    });

    xhr.addEventListener('abort', function() {
        console.warn('XHR Aborted');
        showNotification('✗ Đã hủy tải lên', 'warning');
        showLoading(false);
        document.getElementById('uploadProgress').classList.add('hidden');
    });

    xhr.open('POST', `${API_BASE}/upload-video`, true);
    xhr.send(formData);

    // Reset input
    event.target.value = '';
}

// ============================================
// Display Functions
// ============================================

function updateResultsList(detections) {
    const resultsList = document.getElementById('resultsList');
    
    if (!detections || detections.length === 0) {
        resultsList.innerHTML = '<div class="empty-state">Chưa có kết quả</div>';
        return;
    }

    let html = '';
    detections.forEach((detection, index) => {
        const plateNumber = detection.plate_number || 'UNKNOWN';
        const confidence = detection.confidence || 0;
        
        // Add to cache
        plateCache.add(plateNumber);

        html += `
            <div class="detection-item">
                <div class="plate-number">🚗 ${plateNumber}</div>
                <div class="detection-confidence">
                    Độ tin cậy: ${formatConfidence(confidence)}
                </div>
                <div class="detection-time">
                    ${formatTime(new Date())}
                </div>
            </div>
        `;
    });

    resultsList.innerHTML = html;
}

function updateStats() {
    fetch(`${API_BASE}/stats`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('totalDetections').textContent = data.total_detections;
            document.getElementById('uniquePlates').textContent = data.unique_count;
            
            const avgConf = data.average_confidence ? formatConfidence(data.average_confidence) : '0%';
            document.getElementById('avgConfidence').textContent = avgConf;

            // Update plate history
            updatePlateHistory(data.unique_plates);
        })
        .catch(error => console.error('Error fetching stats:', error));
}

function updatePlateHistory(plates) {
    const history = document.getElementById('plateHistory');
    
    if (!plates || plates.length === 0) {
        history.innerHTML = '<div class="empty-state">Chưa có lịch sử</div>';
        return;
    }

    let html = '';
    plates.slice(-10).reverse().forEach(plate => {
        html += `
            <div class="detection-item">
                <div class="plate-number">📋 ${plate}</div>
            </div>
        `;
    });

    history.innerHTML = html;
}

// ============================================
// Test Functions
// ============================================

function testSystem() {
    showLoading(true);
    console.log('Testing system...');
    
    fetch(`${API_BASE}/test-system`)
        .then(response => {
            console.log('Test response status:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('Test result:', data);
            
            let message = '🧪 Kiểm Tra Hệ Thống:\\n\\n';
            message += `Status: ${data.status}\\n`;
            message += `Recognizer: ${data.components.recognizer.status}\\n`;
            message += `Video Recognizer: ${data.components.video_recognizer.status}\\n`;
            message += `Uploads folder: ${data.folders.uploads ? '✓' : '✗'}\\n`;
            message += `Results folder: ${data.folders.results ? '✓' : '✗'}\\n`;
            message += `Write permission: ${data.write_permission}\\n`;
            
            alert(message);
            showNotification('✓ Hệ thống hoạt động bình thường', 'success');
            showLoading(false);
        })
        .catch(error => {
            console.error('Test error:', error);
            showNotification(`✗ Lỗi test: ${error.message}`, 'error');
            showLoading(false);
        });
}

// ============================================

setInterval(updateStats, 2000); // Update stats every 2 seconds

// ============================================
// Page Initialization
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Page initialized');
    updateStatusIndicator(false);
    updateStats();

    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
        if (eventSource) {
            eventSource.close();
        }
        fetch(`${API_BASE}/stop-webcam`, { method: 'POST' });
    });
});

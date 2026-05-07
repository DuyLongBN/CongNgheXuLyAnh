"""
Deep learning models for license plate recognition
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import config


class PlateDetectionModel:
    """YOLO-based plate detection model"""
    
    @staticmethod
    def build_yolo_model(input_shape=(416, 416, 3), num_anchors=3):
        """
        Build YOLO model for plate detection
        Args:
            input_shape: Input image shape
            num_anchors: Number of anchor boxes
        Returns:
            Compiled model
        """
        model = models.Sequential([
            # Backbone - Darknet53
            # Initial convolution blocks
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Detection head
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(1024, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            
            # Output layer (coordinates + confidence + class probabilities)
            layers.Conv2D(num_anchors * 6, (1, 1), padding='same'),
            layers.Reshape((-1, 6))  # [x, y, w, h, confidence, class]
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        return model


class CharacterRecognitionCNN:
    """CNN model for character recognition"""
    
    @staticmethod
    def build_model(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                   num_classes=config.NUM_CLASSES):
        """
        Build CNN for character recognition
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of character classes
        Returns:
            Compiled model
        """
        model = models.Sequential([
            # Normalization layer
            layers.Lambda(lambda x: x / 255.0, input_shape=input_shape),
            
            # Block 1
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class LSTMCharacterRecognizer:
    """LSTM-based sequence-to-sequence model for plate recognition"""
    
    @staticmethod
    def build_model(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                   num_classes=config.NUM_CLASSES,
                   max_length=config.MAX_PLATE_LENGTH):
        """
        Build LSTM model for plate number recognition
        Args:
            input_shape: Input image shape
            num_classes: Number of character classes
            max_length: Maximum plate length
        Returns:
            Compiled model
        """
        inputs = layers.Input(shape=input_shape)
        
        # CNN feature extractor
        x = layers.Lambda(lambda img: img / 255.0)(inputs)
        
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Reshape for LSTM (sequence of features)
        # Compute dimensions statically: after 3x MaxPooling2D(2,2), spatial dims are divided by 8
        reshape_h = input_shape[0] // 8
        reshape_w = input_shape[1] // 8
        reshape_channels = 128
        x = layers.Reshape((reshape_h, reshape_w * reshape_channels))(x)
        
        # Bidirectional LSTM
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        
        # Dense layer for each time step
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Take the last output (entire plate)
        outputs = layers.Lambda(lambda x: x[:, -1, :])(outputs)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class PlateRecognitionEnsemble:
    """Ensemble model combining detection and recognition"""
    
    def __init__(self):
        self.detector = PlateDetectionModel.build_yolo_model()
        self.recognizer = CharacterRecognitionCNN.build_model()
    
    def detect_and_recognize(self, image):
        """
        Detect plate and recognize characters
        Args:
            image: Input image
        Returns:
            Detection and recognition results
        """
        # First detect plate
        detections = self.detector.predict(image)
        
        # Then recognize characters in detected plate
        recognitions = self.recognizer.predict(detections)
        
        return {
            'detections': detections,
            'recognitions': recognitions
        }

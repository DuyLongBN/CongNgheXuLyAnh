"""
Training script for license plate recognition model
"""
import os
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import config
from src import utils, model


class DatasetBuilder:
    """Build training dataset from images"""
    
    def __init__(self, data_path=config.DATA_PROCESSED_PATH):
        self.data_path = data_path
        self.char_to_idx = {char: idx for idx, char in enumerate(config.ALLOWED_CHARS)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
    
    def load_and_preprocess_image(self, image_path, target_size=(config.IMG_WIDTH, config.IMG_HEIGHT)):
        """Load and preprocess image"""
        img = utils.load_image(str(image_path))
        img = cv2.resize(img, target_size)
        img = utils.normalize_image(img)
        return img
    
    def create_dataset(self, image_folder, labels_file):
        """
        Create dataset from folder of images and labels
        Args:
            image_folder: Folder containing image files
            labels_file: Text file with labels (one per line, in same order as images)
        Returns:
            X, y arrays
        """
        images = []
        labels = []
        
        # Get image files
        image_files = sorted(utils.get_image_files(image_folder))
        
        # Read labels
        with open(labels_file, 'r') as f:
            label_list = [line.strip() for line in f.readlines()]
        
        for img_path, label in zip(image_files, label_list):
            try:
                img = self.load_and_preprocess_image(img_path)
                images.append(img)
                labels.append(label)
            except Exception as e:
                utils.log_message(f"Error loading {img_path}: {e}", 'WARNING')
        
        X = np.array(images)
        y = self._encode_labels(labels)
        
        return X, y
    
    def _encode_labels(self, labels):
        """Encode string labels to one-hot vectors"""
        max_len = config.MAX_PLATE_LENGTH
        num_classes = config.NUM_CLASSES
        
        encoded = np.zeros((len(labels), num_classes))
        
        for idx, label in enumerate(labels):
            # Use first character only for now (single character classification)
            if len(label) > 0:
                char = label[0].upper()
                if char in self.char_to_idx:
                    encoded[idx, self.char_to_idx[char]] = 1
        
        return encoded
    
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic training data"""
        X = []
        y = []
        
        for _ in range(num_samples):
            # Generate random noise image
            img = np.random.randint(0, 256, (config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8)
            
            # Add random character
            char_idx = np.random.randint(0, config.NUM_CLASSES)
            char = config.ALLOWED_CHARS[char_idx]
            
            X.append(utils.normalize_image(img))
            
            # One-hot encode
            label = np.zeros(config.NUM_CLASSES)
            label[char_idx] = 1
            y.append(label)
        
        return np.array(X), np.array(y)


class ModelTrainer:
    """Train recognition models"""
    
    def __init__(self, model_name='cnn'):
        self.model_name = model_name
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self):
        """Build model based on name"""
        if self.model_name == 'cnn':
            return model.CharacterRecognitionCNN.build_model()
        elif self.model_name == 'lstm':
            return model.LSTMCharacterRecognizer.build_model()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=config.EPOCHS):
        """
        Train the model
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
        Returns:
            Training history
        """
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Split data if validation not provided
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=config.VALIDATION_SPLIT,
                random_state=42
            )
        
        # Train model
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
            validation_data=(X_val, y_val),
            epochs=epochs,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                )
            ]
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        loss, accuracy = self.model.evaluate(X_test, y_test)
        utils.log_message(f"Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, accuracy
    
    def save_model(self, save_path=None):
        """Save trained model"""
        if save_path is None:
            save_path = os.path.join(config.MODELS_PATH, f'{self.model_name}_model.h5')
        
        self.model.save(save_path)
        utils.log_message(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """Load trained model"""
        self.model = tf.keras.models.load_model(model_path)
        utils.log_message(f"Model loaded from {model_path}")


def main():
    """Main training script"""
    utils.create_directories()
    
    utils.log_message("Starting license plate character recognition training...")
    
    # Create synthetic dataset for demonstration
    utils.log_message("Generating synthetic dataset...")
    dataset_builder = DatasetBuilder()
    X, y = dataset_builder.generate_synthetic_data(num_samples=5000)
    
    # Split into train, validation, test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    utils.log_message(f"Dataset sizes - Train: {X_train.shape[0]}, "
                     f"Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train CNN model
    utils.log_message("Training CNN model...")
    trainer = ModelTrainer(model_name='cnn')
    trainer.train(X_train, y_train, X_val, y_val, epochs=config.EPOCHS)
    
    # Evaluate
    utils.log_message("Evaluating model...")
    trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    utils.log_message("Training completed!")


if __name__ == '__main__':
    main()

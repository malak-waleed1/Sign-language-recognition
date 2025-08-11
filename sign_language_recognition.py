import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import json
import threading
import time
import platform
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import random

class HandGestureDetector:
    """Detect and track hand gestures using MediaPipe"""
    
    def __init__(self):
        """Initialize MediaPipe hand detection"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Hand gesture detector initialized")
    
    def detect_hands(self, image):
        """Detect hands in the image and return landmarks"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Get the first hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks), hand_landmarks
        
        return None, None
    
    def draw_hand_landmarks(self, image, hand_landmarks):
        """Draw hand landmarks on the image"""
        if hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        return image
    
    def extract_features(self, landmarks):
        """Extract features from hand landmarks for classification"""
        if landmarks is None:
            return None
        
        features = []
        
        # Normalize landmarks relative to wrist (landmark 0)
        wrist = landmarks[0]
        normalized_landmarks = landmarks - wrist
        
        # Flatten the landmarks
        features = normalized_landmarks.flatten()
        
        # Add additional features
        # Distance between key points
        key_points = [4, 8, 12, 16, 20]  # Fingertips
        for i in range(len(key_points)):
            for j in range(i + 1, len(key_points)):
                dist = np.linalg.norm(landmarks[key_points[i]] - landmarks[key_points[j]])
                features = np.append(features, dist)
        
        # Angles between fingers
        for i in range(len(key_points) - 1):
            for j in range(i + 1, len(key_points)):
                for k in range(j + 1, len(key_points)):
                    v1 = landmarks[key_points[i]] - landmarks[key_points[j]]
                    v2 = landmarks[key_points[k]] - landmarks[key_points[j]]
                    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                    features = np.append(features, angle)
        
        return features

class SignLanguageClassifier:
    """Machine learning classifier for sign language recognition"""
    
    def __init__(self):
        """Initialize the classifier"""
        self.model = None
        self.scaler = StandardScaler()
        self.classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.data_dir = "sign_language_data"
        self.model_file = os.path.join(self.data_dir, "sign_language_model.joblib")
        self.scaler_file = os.path.join(self.data_dir, "scaler.joblib")
        self.training_data_file = os.path.join(self.data_dir, "training_data.pkl")
        
        self.create_directories()
        self.load_model()
    
    def create_directories(self):
        """Create necessary directories"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(os.path.join(self.data_dir, "images")):
            os.makedirs(os.path.join(self.data_dir, "images"))
    
    def load_model(self):
        """Load trained model and scaler"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
                self.model = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                self.logger.info("Model and scaler loaded successfully")
            else:
                self.logger.info("No trained model found. Please train the model first.")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
    
    def save_model(self):
        """Save trained model and scaler"""
        try:
            joblib.dump(self.model, self.model_file)
            joblib.dump(self.scaler, self.scaler_file)
            self.logger.info("Model and scaler saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def train_model(self, X, y):
        """Train the machine learning model"""
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate the model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
            self.logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            # Save the model
            self.save_model()
            
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return None
    
    def predict(self, features):
        """Predict sign language letter from features"""
        if self.model is None:
            return None, 0.0
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return None, 0.0

class DataCollector:
    """Collect training data for sign language recognition"""
    
    def __init__(self):
        """Initialize data collector"""
        self.hand_detector = HandGestureDetector()
        self.data_dir = "sign_language_data"
        self.training_data = []
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def collect_data(self, letter, num_samples=50):
        """Collect training data for a specific letter"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.logger.error("Could not open camera")
            return False
        
        self.logger.info(f"Collecting data for letter '{letter}'. Press 'q' to stop.")
        
        samples_collected = 0
        
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            landmarks, hand_landmarks = self.hand_detector.detect_hands(frame)
            
            # Draw landmarks
            if hand_landmarks:
                frame = self.hand_detector.draw_hand_landmarks(frame, hand_landmarks)
            
            # Add text to frame
            cv2.putText(frame, f"Collecting '{letter}' - {samples_collected}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to stop", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Collect sample when hand is detected
            if landmarks is not None:
                features = self.hand_detector.extract_features(landmarks)
                if features is not None:
                    self.training_data.append({
                        'letter': letter,
                        'features': features,
                        'landmarks': landmarks
                    })
                    samples_collected += 1
                    self.logger.info(f"Collected sample {samples_collected} for letter '{letter}'")
                    time.sleep(0.1)  # Small delay between samples
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.logger.info(f"Collected {samples_collected} samples for letter '{letter}'")
        return True
    
    def save_training_data(self):
        """Save collected training data"""
        try:
            data_file = os.path.join(self.data_dir, "training_data.pkl")
            with open(data_file, 'wb') as f:
                pickle.dump(self.training_data, f)
            self.logger.info(f"Training data saved with {len(self.training_data)} samples")
        except Exception as e:
            self.logger.error(f"Error saving training data: {e}")
    
    def load_training_data(self):
        """Load existing training data"""
        try:
            data_file = os.path.join(self.data_dir, "training_data.pkl")
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    self.training_data = pickle.load(f)
                self.logger.info(f"Loaded {len(self.training_data)} training samples")
            else:
                self.logger.info("No existing training data found")
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")

class SyntheticDataGenerator:
    """Generate synthetic training data for sign language recognition"""
    
    def __init__(self):
        """Initialize synthetic data generator"""
        self.logger = logging.getLogger(__name__)
    
    def generate_synthetic_data(self, num_samples_per_letter=100):
        """Generate synthetic training data for all letters"""
        training_data = []
        
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.logger.info(f"Generating synthetic data for letter '{letter}'")
            
            for i in range(num_samples_per_letter):
                # Generate synthetic landmarks based on letter characteristics
                landmarks = self.generate_letter_landmarks(letter)
                features = self.extract_features_from_landmarks(landmarks)
                
                training_data.append({
                    'letter': letter,
                    'features': features,
                    'landmarks': landmarks
                })
        
        self.logger.info(f"Generated {len(training_data)} synthetic samples")
        return training_data
    
    def generate_letter_landmarks(self, letter):
        """Generate synthetic landmarks for a specific letter"""
        # Base hand position (all fingers extended)
        base_landmarks = np.array([
            [0.5, 0.5, 0.0],  # Wrist
            [0.5, 0.4, 0.0],  # Thumb base
            [0.45, 0.35, 0.0],  # Thumb tip
            [0.6, 0.3, 0.0],  # Index base
            [0.6, 0.2, 0.0],  # Index tip
            [0.7, 0.3, 0.0],  # Middle base
            [0.7, 0.2, 0.0],  # Middle tip
            [0.8, 0.3, 0.0],  # Ring base
            [0.8, 0.2, 0.0],  # Ring tip
            [0.9, 0.3, 0.0],  # Pinky base
            [0.9, 0.2, 0.0],  # Pinky tip
        ])
        
        # Modify landmarks based on letter
        if letter == 'A':
            # Thumb extended, other fingers closed
            base_landmarks[2:11:2] += np.random.normal(0, 0.02, (5, 3))
        elif letter == 'B':
            # All fingers extended
            base_landmarks[2:11:2] += np.random.normal(0, 0.02, (5, 3))
        elif letter == 'C':
            # Curved hand
            base_landmarks[2:11:2] += np.array([0.1, 0.05, 0.0]) + np.random.normal(0, 0.02, (5, 3))
        # Add more letter-specific modifications...
        
        # Add noise for realism
        landmarks = base_landmarks + np.random.normal(0, 0.01, base_landmarks.shape)
        
        return landmarks
    
    def extract_features_from_landmarks(self, landmarks):
        """Extract features from synthetic landmarks"""
        # Similar to HandGestureDetector.extract_features
        features = []
        
        # Normalize landmarks relative to wrist
        wrist = landmarks[0]
        normalized_landmarks = landmarks - wrist
        
        # Flatten the landmarks
        features = normalized_landmarks.flatten()
        
        # Add additional features
        key_points = [2, 4, 6, 8, 10]  # Fingertips
        for i in range(len(key_points)):
            for j in range(i + 1, len(key_points)):
                dist = np.linalg.norm(landmarks[key_points[i]] - landmarks[key_points[j]])
                features = np.append(features, dist)
        
        return features

class VideoCapture:
    """Handle video capture for real-time recognition"""
    
    def __init__(self):
        self.cap = None
        self.is_running = False
    
    def start_capture(self):
        """Start video capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        self.is_running = True
    
    def stop_capture(self):
        """Stop video capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def read_frame(self):
        """Read frame from camera"""
        if not self.is_running or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def __del__(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()

class SignLanguageRecognitionGUI:
    """GUI for the sign language recognition system"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sign Language Recognition System")
        self.root.geometry("1200x700")
        
        # Initialize components
        self.hand_detector = HandGestureDetector()
        self.classifier = SignLanguageClassifier()
        self.video_capture = VideoCapture()
        self.is_running = False
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Video frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Live Camera Feed")
        self.video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.video_label = ttk.Label(self.video_frame, text="Click 'Start Recognition' to begin")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side="right", fill="y")
        
        # Recognition controls
        recog_frame = ttk.LabelFrame(control_frame, text="Recognition Controls")
        recog_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(recog_frame, text="Start Recognition", 
                  command=self.start_recognition).pack(fill="x", padx=5, pady=5)
        ttk.Button(recog_frame, text="Stop Recognition", 
                  command=self.stop_recognition).pack(fill="x", padx=5, pady=5)
        
        # Training controls
        train_frame = ttk.LabelFrame(control_frame, text="Training Controls")
        train_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(train_frame, text="Collect Training Data", 
                  command=self.collect_training_data).pack(fill="x", padx=5, pady=5)
        ttk.Button(train_frame, text="Generate Synthetic Data", 
                  command=self.generate_synthetic_data).pack(fill="x", padx=5, pady=5)
        ttk.Button(train_frame, text="Train Model", 
                  command=self.train_model).pack(fill="x", padx=5, pady=5)
        
        # Recognition results
        results_frame = ttk.LabelFrame(control_frame, text="Recognition Results")
        results_frame.pack(fill="x", pady=(0, 10))
        
        self.result_label = ttk.Label(results_frame, text="No recognition yet")
        self.result_label.pack(padx=5, pady=5)
        
        self.confidence_label = ttk.Label(results_frame, text="")
        self.confidence_label.pack(padx=5, pady=5)
        
        # Status
        status_frame = ttk.LabelFrame(control_frame, text="System Status")
        status_frame.pack(fill="x")
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(padx=5, pady=5)
    
    def start_recognition(self):
        """Start the sign language recognition process"""
        if self.is_running:
            return
        
        try:
            self.video_capture.start_capture()
            self.is_running = True
            self.status_label.config(text="Recognition Active")
            
            # Start recognition in a separate thread
            self.recognition_thread = threading.Thread(target=self.recognition_loop)
            self.recognition_thread.daemon = True
            self.recognition_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not start camera: {str(e)}")
    
    def stop_recognition(self):
        """Stop the sign language recognition process"""
        self.is_running = False
        self.video_capture.stop_capture()
        self.status_label.config(text="Stopped")
        self.video_label.config(text="Click 'Start Recognition' to begin")
    
    def recognition_loop(self):
        """Main recognition loop"""
        while self.is_running:
            frame = self.video_capture.read_frame()
            if frame is None:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            landmarks, hand_landmarks = self.hand_detector.detect_hands(frame)
            
            # Draw landmarks
            if hand_landmarks:
                frame = self.hand_detector.draw_hand_landmarks(frame, hand_landmarks)
            
            # Recognize sign language
            if landmarks is not None:
                features = self.hand_detector.extract_features(landmarks)
                if features is not None:
                    prediction, confidence = self.classifier.predict(features)
                    
                    if prediction:
                        # Draw prediction on frame
                        cv2.putText(frame, f"Letter: {prediction}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Update GUI
                        self.root.after(0, lambda: self.update_results(prediction, confidence))
            
            # Convert frame to PhotoImage for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update video label
            self.video_label.config(image=frame_tk)
            self.video_label.image = frame_tk
            
            # Small delay
            time.sleep(0.03)
    
    def update_results(self, prediction, confidence):
        """Update recognition results"""
        self.result_label.config(text=f"Recognized: {prediction}")
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
    
    def collect_training_data(self):
        """Open data collection dialog"""
        dialog = DataCollectionDialog(self.root)
        self.root.wait_window(dialog)
    
    def generate_synthetic_data(self):
        """Generate synthetic training data"""
        try:
            generator = SyntheticDataGenerator()
            training_data = generator.generate_synthetic_data()
            
            # Save training data
            data_collector = DataCollector()
            data_collector.training_data = training_data
            data_collector.save_training_data()
            
            messagebox.showinfo("Success", f"Generated {len(training_data)} synthetic samples")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating synthetic data: {str(e)}")
    
    def train_model(self):
        """Train the machine learning model"""
        try:
            # Load training data
            data_collector = DataCollector()
            data_collector.load_training_data()
            
            if not data_collector.training_data:
                messagebox.showerror("Error", "No training data available. Please collect or generate data first.")
                return
            
            # Prepare data
            X = np.array([sample['features'] for sample in data_collector.training_data])
            y = np.array([sample['letter'] for sample in data_collector.training_data])
            
            # Train model
            accuracy = self.classifier.train_model(X, y)
            
            if accuracy:
                messagebox.showinfo("Success", f"Model trained successfully. Accuracy: {accuracy:.4f}")
            else:
                messagebox.showerror("Error", "Failed to train model")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()
    
    def __del__(self):
        """Cleanup"""
        if self.video_capture:
            self.video_capture.stop_capture()

class DataCollectionDialog(tk.Toplevel):
    """Dialog for collecting training data"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Data Collection")
        self.geometry("400x300")
        
        self.data_collector = DataCollector()
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the data collection GUI"""
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Sign Language Data Collection", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Letter selection
        letter_frame = ttk.LabelFrame(main_frame, text="Select Letter")
        letter_frame.pack(fill="x", pady=(0, 10))
        
        self.letter_var = tk.StringVar(value="A")
        letter_combo = ttk.Combobox(letter_frame, textvariable=self.letter_var, 
                                   values=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        letter_combo.pack(fill="x", padx=5, pady=5)
        
        # Sample count
        count_frame = ttk.LabelFrame(main_frame, text="Number of Samples")
        count_frame.pack(fill="x", pady=(0, 10))
        
        self.count_var = tk.StringVar(value="50")
        count_entry = ttk.Entry(count_frame, textvariable=self.count_var)
        count_entry.pack(fill="x", padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(button_frame, text="Start Collection", 
                  command=self.start_collection).pack(fill="x", pady=2)
        ttk.Button(button_frame, text="Cancel", 
                  command=self.destroy).pack(fill="x", pady=2)
        
        # Instructions
        instructions = """
        Instructions:
        1. Select the letter to collect data for
        2. Set the number of samples to collect
        3. Click 'Start Collection'
        4. Show the sign language gesture to the camera
        5. Press 'q' to stop collection
        """
        
        instruction_label = ttk.Label(main_frame, text=instructions, justify="left")
        instruction_label.pack(fill="x", pady=(10, 0))
    
    def start_collection(self):
        """Start data collection"""
        try:
            letter = self.letter_var.get()
            num_samples = int(self.count_var.get())
            
            if not letter or num_samples <= 0:
                messagebox.showerror("Error", "Please enter valid letter and sample count")
                return
            
            self.withdraw()  # Hide dialog during collection
            
            # Collect data
            success = self.data_collector.collect_data(letter, num_samples)
            
            if success:
                messagebox.showinfo("Success", f"Collected {num_samples} samples for letter '{letter}'")
            
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during data collection: {str(e)}")
            self.deiconify()  # Show dialog again

def main():
    """Main function to run the sign language recognition system"""
    print("Starting Sign Language Recognition System...")
    print("Features:")
    print("- Real-time hand detection using MediaPipe")
    print("- Machine learning classification (Random Forest)")
    print("- Training data collection and synthetic data generation")
    print("- GUI interface for easy interaction")
    print("- Support for all alphabet letters (A-Z)")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sign_language_recognition.log'),
            logging.StreamHandler()
        ]
    )
    
    # Start GUI
    app = SignLanguageRecognitionGUI()
    app.run()

if __name__ == "__main__":
    main()
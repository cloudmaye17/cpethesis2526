#!/usr/bin/env python3
"""
Complete Face Recognition Attendance System with Accuracy Factors
Microcity College of Business and Technology
"""

import cv2
import numpy as np
import time
import os
import sqlite3
import pickle
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("FACE RECOGNITION ATTENDANCE SYSTEM")
print("Microcity College of Business and Technology")
print("=" * 70)
print("Starting system...")

# Create necessary folders
os.makedirs("logs", exist_ok=True)
os.makedirs("database", exist_ok=True)


class AccuracyFactors:
    """Tracks and manages factors that affect system accuracy"""

    def __init__(self):
        self.factors = {
            'lighting_quality': 1.0,
            'face_size': 1.0,
            'face_angle': 1.0,
            'motion_blur': 1.0,
            'occlusion': 1.0,
            'distance': 1.0
        }

    def analyze_frame(self, frame, face_bbox):
        """Analyze accuracy factors for current frame and face"""
        x, y, w, h = face_bbox

        # 1. Lighting Quality
        face_roi = frame[y:y + h, x:x + w]
        if face_roi.size > 0:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            # Optimal brightness: 100-150
            if 100 <= brightness <= 150:
                self.factors['lighting_quality'] = 1.0
            elif 50 <= brightness < 100 or 150 < brightness <= 200:
                self.factors['lighting_quality'] = 0.7
            else:
                self.factors['lighting_quality'] = 0.3

        # 2. Face Size
        ideal_size = 200
        face_area = w * h
        ideal_area = ideal_size * ideal_size
        size_ratio = min(1.0, face_area / ideal_area)
        self.factors['face_size'] = size_ratio

        # 3. Distance (inferred from size)
        self.factors['distance'] = size_ratio

        return self.factors

    def get_overall_accuracy_score(self):
        """Calculate overall accuracy score from all factors (0-100%)"""
        weights = {
            'lighting_quality': 0.25,
            'face_size': 0.20,
            'face_angle': 0.15,
            'motion_blur': 0.15,
            'occlusion': 0.15,
            'distance': 0.10
        }

        total_score = 0
        for factor, weight in weights.items():
            total_score += self.factors[factor] * weight

        return min(100, total_score * 100)


class DatabaseManager:
    """Manages all database operations"""

    def __init__(self, db_path='database/faces.db'):
        self.db_path = db_path
        self.student_info = {}
        self.load_all_students()

    def load_all_students(self):
        """Load all active student information from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT person_id, student_id, first_name, last_name, 
                       course, year_level, section, photo_count
                FROM persons
                WHERE status = 'active'
            ''')

            for row in cursor.fetchall():
                person_id = row[0]
                self.student_info[person_id] = {
                    'student_id': row[1],
                    'first_name': row[2],
                    'last_name': row[3],
                    'full_name': f"{row[2]} {row[3]}",
                    'course': row[4],
                    'year_level': row[5],
                    'section': row[6],
                    'photo_count': row[7]
                }

            conn.close()
            print(f"✓ Loaded {len(self.student_info)} students from database")

        except Exception as e:
            print(f"✗ Warning: Could not load students: {e}")

    def log_attendance(self, person_id, confidence, accuracy_score, factors):
        """Log attendance record with accuracy factors"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Store factors as string
            factors_json = str(factors)

            cursor.execute('''
                INSERT INTO attendance_logs 
                (person_id, confidence, camera_id, location, status, accuracy_score, factors)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                person_id,
                confidence,
                'Camera_1',
                'Classroom',
                'present',
                accuracy_score,
                factors_json
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"✗ Error logging attendance: {e}")
            return False

    def get_student_info(self, person_id):
        """Get student information by person_id"""
        return self.student_info.get(person_id, None)


class FaceDetector:
    """Face detector using Haar Cascade"""

    def __init__(self):
        print("Loading face detector...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        if self.face_cascade.empty():
            print("✗ ERROR: Could not load Haar Cascade!")
            raise RuntimeError("No face detector available")

        print("✓ Face detector ready")

    def detect_faces(self, frame):
        """Detect faces in frame - returns list of (x,y,w,h)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            maxSize=(400, 400)
        )

        return faces


class FaceRecognizer:
    """Face recognizer using LBPH with improved preprocessing"""

    def __init__(self, model_path='database/trained_model.yml'):
        self.model_path = model_path
        self.recognizer = None
        self.name_mapping = {}
        self.load_model()

        # Recognition thresholds
        self.confidence_threshold = 55  # LBPH distance threshold
        self.min_confidence_percent = 45  # Minimum confidence to accept
        self.debug_mode = False

    def load_model(self):
        """Load trained model and name mapping"""
        if not os.path.exists(self.model_path):
            print("⚠ WARNING: No trained model found!")
            print("  Please run train_model.py first")
            print("  Recognition will be DISABLED until model is trained")
            return False

        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1,
                neighbors=8,
                grid_x=8,
                grid_y=8
            )
            self.recognizer.read(self.model_path)

            # Load name mapping
            mapping_path = 'database/name_mapping.pkl'
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    self.name_mapping = pickle.load(f)
                print(f"✓ Loaded {len(self.name_mapping)} names")

            print("✓ Face recognition model loaded")
            return True

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False

    def preprocess_face(self, face_img):
        """Preprocess face image - MUST match training exactly"""
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        # Resize to 200x200 (MUST match training)
        gray = cv2.resize(gray, (200, 200))

        # Equalize histogram (MUST match training)
        gray = cv2.equalizeHist(gray)

        return gray

    def recognize(self, face_roi):
        """Recognize a face - returns (person_id, confidence_percent, distance)"""
        if self.recognizer is None:
            return None, 0, 0

        try:
            # Preprocess exactly like training
            processed_face = self.preprocess_face(face_roi)

            # Predict
            person_id, distance = self.recognizer.predict(processed_face)

            # Convert distance to confidence percentage
            confidence_percent = max(0, 100 - distance)

            if self.debug_mode:
                name = self.name_mapping.get(person_id, f"ID_{person_id}")
                print(f"  Recognition: ID={person_id}, Name={name}, "
                      f"Dist={distance:.1f}, Conf={confidence_percent:.1f}%")

            # Apply thresholds
            if distance > self.confidence_threshold:
                if self.debug_mode:
                    print(f"    -> REJECTED: Distance {distance:.1f} > {self.confidence_threshold}")
                return None, confidence_percent, distance

            if confidence_percent < self.min_confidence_percent:
                if self.debug_mode:
                    print(f"    -> REJECTED: Confidence {confidence_percent:.1f}% < {self.min_confidence_percent}%")
                return None, confidence_percent, distance

            return person_id, confidence_percent, distance

        except Exception as e:
            if self.debug_mode:
                print(f"  Recognition error: {e}")
            return None, 0, 0


def draw_accuracy_info(frame, x, y, w, h, factors, accuracy_score):
    """Draw accuracy factors information on frame"""
    bar_x = x + w + 10
    bar_y = y
    bar_width = 100
    bar_height = 15

    factors_list = [
        ('Light', factors['lighting_quality']),
        ('Size', factors['face_size']),
        ('Dist', factors['distance'])
    ]

    for i, (name, value) in enumerate(factors_list):
        bar_y_pos = bar_y + i * 20

        # Draw bar background
        cv2.rectangle(frame, (bar_x, bar_y_pos),
                      (bar_x + bar_width, bar_y_pos + bar_height),
                      (50, 50, 50), -1)

        # Draw bar value
        bar_fill = int(value * bar_width)
        color = (0, 255, 0) if value > 0.7 else (0, 165, 255) if value > 0.4 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y_pos),
                      (bar_x + bar_fill, bar_y_pos + bar_height),
                      color, -1)

        # Draw factor name and value
        cv2.putText(frame, f"{name}: {value:.1f}", (bar_x, bar_y_pos + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw overall accuracy score
    accuracy_color = (0, 255, 0) if accuracy_score > 70 else (0, 165, 255) if accuracy_score > 50 else (0, 0, 255)
    cv2.putText(frame, f"Accuracy: {accuracy_score:.0f}%",
                (bar_x, bar_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, accuracy_color, 1)


def check_requirements():
    """Check if system is properly set up"""
    errors = []

    if not os.path.exists('database/faces.db'):
        errors.append("Database not found - run setup_database.py")

    if not os.path.exists('database/trained_model.yml'):
        errors.append("Trained model not found - run train_model.py")

    if not os.path.exists('database/name_mapping.pkl'):
        errors.append("Name mapping not found - run train_model.py")

    if errors:
        print("\n✗ SETUP INCOMPLETE:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease complete setup before running main.py")
        return False

    return True


def main():
    """Main function - runs the complete attendance system"""

    # Check requirements first
    if not check_requirements():
        return

    print("\n" + "=" * 70)
    print("SYSTEM READY")
    print("=" * 70)
    print("\nControls:")
    print("  Q or ESC - Quit")
    print("  R - Reset statistics")
    print("  S - Save screenshot")
    print("  D - Toggle debug mode")
    print("  +/- - Adjust recognition threshold")
    print("=" * 70)

    # Initialize components
    try:
        detector = FaceDetector()
        recognizer = FaceRecognizer()
        db = DatabaseManager()
        accuracy_tracker = AccuracyFactors()
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        return

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("✗ ERROR: Cannot open camera")
            return

    # Set HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n✓ Camera started! Press 'Q' to quit\n")

    # Statistics
    fps = 0
    frame_count = 0
    last_time = time.time()

    stats = {
        'recognized': 0,
        'unknown': 0,
        'attendance_logged': 0,
        'total_faces': 0
    }

    # Attendance cooldown
    last_logged = {}
    log_cooldown = 5  # seconds

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                continue

            # Flip for mirror view
            frame = cv2.flip(frame, 1)

            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_time = current_time

            # Detect faces
            faces = detector.detect_faces(frame)
            stats['total_faces'] += len(faces)

            # Process each face
            for (x, y, w, h) in faces:
                # Validate face size
                if w > 400 or h > 400 or w < 50 or h < 50:
                    continue

                # Extract face region
                face_roi = frame[y:y + h, x:x + w]
                if face_roi.size == 0:
                    continue

                # Analyze accuracy factors
                factors = accuracy_tracker.analyze_frame(frame, (x, y, w, h))
                accuracy_score = accuracy_tracker.get_overall_accuracy_score()

                # Draw accuracy info
                if x + w + 120 < frame.shape[1]:  # Only if there's space
                    draw_accuracy_info(frame, x, y, w, h, factors, accuracy_score)

                # Try recognition if model is loaded
                person_id = None
                confidence = 0
                student_info = None

                if recognizer.recognizer is not None:
                    person_id, confidence, distance = recognizer.recognize(face_roi)

                    if person_id is not None:
                        student_info = db.get_student_info(person_id)

                # Determine color and status
                if person_id is not None and student_info is not None:
                    # RECOGNIZED - GREEN
                    color = (0, 255, 0)
                    status = "RECOGNIZED"
                    name = student_info['full_name'][:20]

                    # Log attendance with cooldown
                    if person_id not in last_logged or (current_time - last_logged[person_id]) > log_cooldown:
                        if db.log_attendance(person_id, confidence, accuracy_score, factors):
                            last_logged[person_id] = current_time
                            stats['attendance_logged'] += 1
                            stats['recognized'] += 1
                            print(f"✓ {name} - Conf: {confidence:.1f}% | Accuracy: {accuracy_score:.0f}%")
                else:
                    # UNKNOWN or low confidence
                    color = (0, 255, 255) if confidence > 30 else (0, 0, 255)
                    status = "UNKNOWN" if confidence <= 30 else f"LOW:{confidence:.0f}%"
                    name = "UNKNOWN"
                    stats['unknown'] += 1

                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Draw status bar
                cv2.rectangle(frame, (x, y - 25), (x + w, y), color, -1)
                cv2.putText(frame, status, (x + 5, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Draw name and confidence
                if person_id is not None:
                    cv2.putText(frame, name, (x + 5, y + h + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(frame, f"{confidence:.0f}%", (x + w - 45, y + h + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw statistics overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (250, 140), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            # Draw stats
            cv2.putText(frame, f"FPS: {fps}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Recognized: {stats['recognized']}", (20, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Unknown: {stats['unknown']}", (20, 104),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Attendance: {stats['attendance_logged']}", (20, 126),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show detector type and threshold
            cv2.putText(frame, "Haar Cascade", (frame.shape[1] - 150, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(frame, f"Threshold: {recognizer.confidence_threshold}",
                        (frame.shape[1] - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display frame
            cv2.imshow("Face Recognition - Microcity College", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('r'):
                stats = {k: 0 for k in stats}
                last_logged.clear()
                print("\n✓ Statistics reset")
            elif key == ord('s'):
                filename = f"logs/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Screenshot saved: {filename}")
            elif key == ord('d'):
                recognizer.debug_mode = not recognizer.debug_mode
                status = "ON" if recognizer.debug_mode else "OFF"
                print(f"✓ Debug mode: {status}")
            elif key == ord('+') or key == ord('='):
                recognizer.confidence_threshold = min(100, recognizer.confidence_threshold + 5)
                print(f"✓ Threshold increased to {recognizer.confidence_threshold}")
            elif key == ord('-') or key == ord('_'):
                recognizer.confidence_threshold = max(20, recognizer.confidence_threshold - 5)
                print(f"✓ Threshold decreased to {recognizer.confidence_threshold}")

    except KeyboardInterrupt:
        print("\n\n✓ Stopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Print summary
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(f"Detector: Haar Cascade")
        print(f"Total faces detected: {stats['total_faces']}")
        print(f"Recognized: {stats['recognized']}")
        print(f"Unknown: {stats['unknown']}")
        print(f"Attendance records logged: {stats['attendance_logged']}")
        print(f"Final threshold: {recognizer.confidence_threshold}")
        print("=" * 70)


if __name__ == "__main__":
    main()
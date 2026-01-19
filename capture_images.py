#!/usr/bin/env python3
"""
Capture Training Images for Students
Opens camera and captures multiple face images for training
"""

import cv2
import os
import sqlite3
import sys
from datetime import datetime


def check_database_exists():
    """Check if database exists"""
    if not os.path.exists('database/faces.db'):
        print("ERROR: Database not found!")
        print("Please run setup_database.py first")
        return False
    return True


def capture_images_for_student(person_id, first_name, last_name, num_images=20):
    """
    Capture training images for a specific student
    """

    # Create folder for this student's images
    folder_name = f"Person_{person_id}_{first_name}_{last_name}"
    folder_path = os.path.join("database", "training_images", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("✗ ERROR: Cannot open camera!")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    if face_cascade.empty():
        print("✗ ERROR: Could not load face detector!")
        cap.release()
        return

    # Counters
    count = 0
    saved_count = 0

    print(f"\n{'=' * 70}")
    print(f"CAPTURING IMAGES FOR: {first_name} {last_name}")
    print(f"{'=' * 70}")
    print(f"Target: {num_images} images")
    print("\nInstructions:")
    print("  - Position your face in the green rectangle")
    print("  - Look at the camera")
    print("  - Images will be captured automatically")
    print("  - Move your head slightly between captures")
    print("  - Press SPACE to manually capture")
    print("  - Press 'Q' to quit early")
    print(f"{'=' * 70}\n")

    try:
        while saved_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                continue

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(150, 150),
                maxSize=(400, 400)
            )

            # Create display frame
            display = frame.copy()

            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if len(faces) == 1 else (0, 165, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display, f"{saved_count}/{num_images}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Add instruction text
            cv2.putText(display, f"Capturing for: {first_name} {last_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Progress: {saved_count}/{num_images}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if len(faces) == 0:
                cv2.putText(display, "No face detected - move closer",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif len(faces) > 1:
                cv2.putText(display, "Multiple faces - only you should be visible",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.putText(display, "Press SPACE to capture | Q to quit",
                        (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            # Show frame
            cv2.imshow("Capture Training Images", display)

            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Auto-capture every 30 frames if exactly one face is detected
            count += 1
            if count % 30 == 0 and len(faces) == 1:
                x, y, w, h = faces[0]
                face_roi = frame[y:y + h, x:x + w]

                # Resize face to standard size
                face_roi = cv2.resize(face_roi, (200, 200))

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{first_name}_{last_name}_{saved_count + 1}_{timestamp}.jpg"
                filepath = os.path.join(folder_path, filename)

                cv2.imwrite(filepath, face_roi)
                saved_count += 1

                print(f"✓ Captured image {saved_count}/{num_images}")

                # Store in database
                try:
                    conn = sqlite3.connect('database/faces.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO faces (person_id, image_path)
                        VALUES (?, ?)
                    ''', (person_id, filepath))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"  Warning: Could not save to database: {e}")

            # Manual capture with spacebar
            elif key == ord(' ') and len(faces) == 1:
                x, y, w, h = faces[0]
                face_roi = frame[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (200, 200))

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{first_name}_{last_name}_{saved_count + 1}_{timestamp}.jpg"
                filepath = os.path.join(folder_path, filename)

                cv2.imwrite(filepath, face_roi)
                saved_count += 1
                print(f"✓ Manually captured image {saved_count}/{num_images}")

                try:
                    conn = sqlite3.connect('database/faces.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO faces (person_id, image_path)
                        VALUES (?, ?)
                    ''', (person_id, filepath))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"  Warning: Could not save to database: {e}")

            # Quit if user presses 'q'
            elif key == ord('q') or key == 27:
                print("\nCapture cancelled by user")
                break

    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during capture: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Update photo count in database
    try:
        conn = sqlite3.connect('database/faces.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE persons SET photo_count = ? WHERE person_id = ?
        ''', (saved_count, person_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not update photo count: {e}")

    print(f"\n{'=' * 70}")
    print(f"Capture complete! Saved {saved_count} images")
    print(f"Location: {folder_path}")
    print(f"{'=' * 70}\n")


def list_students():
    """Get list of all students from database"""
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT person_id, student_id, first_name, last_name, photo_count, status
        FROM persons
        WHERE status = 'active'
        ORDER BY person_id
    ''')

    students = cursor.fetchall()
    conn.close()

    return students


if __name__ == "__main__":
    if not check_database_exists():
        sys.exit(1)

    print("=" * 70)
    print("TRAINING IMAGE CAPTURE SYSTEM")
    print("=" * 70)

    students = list_students()

    if not students:
        print("\n✗ No students found in database!")
        print("   Please run add_student.py first")
        sys.exit(1)

    # Display all students
    print("\nRegistered Students:")
    print("-" * 70)
    for i, (pid, sid, fname, lname, count, status) in enumerate(students, 1):
        print(f"{i}. {fname} {lname} (ID: {sid}) - {count} photos")

    # Let user choose
    print("\nOptions:")
    print("0. Capture for ALL students")
    print("1-{}: Capture for specific student".format(len(students)))

    try:
        choice = int(input("\nSelect option: "))

        if choice == 0:
            # Capture for all students
            for pid, sid, fname, lname, count, status in students:
                print(f"\n\nProcessing: {fname} {lname}")
                input("Press Enter when ready...")
                capture_images_for_student(pid, fname, lname, num_images=20)
        elif 1 <= choice <= len(students):
            # Capture for selected student
            pid, sid, fname, lname, count, status = students[choice - 1]
            capture_images_for_student(pid, fname, lname, num_images=20)
        else:
            print("Invalid choice!")
    except ValueError:
        print("Invalid input!")
    except KeyboardInterrupt:
        print("\nCancelled by user")
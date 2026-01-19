#!/usr/bin/env python3
"""
Train Face Recognition Model
Loads training images from database and creates LBPH model
"""

import cv2
import numpy as np
import os
import sqlite3
import pickle
import sys
from datetime import datetime


def check_database_exists():
    """Check if database exists"""
    if not os.path.exists('database/faces.db'):
        print("ERROR: Database not found!")
        print("Please run setup_database.py first")
        return False
    return True


def train_model():
    """
    Main training function
    1. Loads all training images from database
    2. Preprocesses images (grayscale, resize, histogram equalization)
    3. Trains LBPH face recognition model
    4. Saves trained model to file
    """
    print("=" * 70)
    print("TRAINING FACE RECOGNITION MODEL")
    print("=" * 70)

    # Connect to database
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()

    # Get all face images for active students
    cursor.execute('''
        SELECT f.person_id, f.image_path, p.first_name, p.last_name
        FROM faces f
        JOIN persons p ON f.person_id = p.person_id
        WHERE p.status = 'active'
        ORDER BY f.person_id
    ''')

    data = cursor.fetchall()

    if not data:
        print("\n✗ No training data found!")
        print("   Please run capture_images.py first")
        conn.close()
        return False

    print(f"\nFound {len(data)} images in database")
    print("Loading and preprocessing images...")

    # Lists to store training data
    faces = []
    labels = []
    names = {}

    loaded_count = 0
    error_count = 0
    person_counts = {}

    # Load each face image
    for person_id, image_path, first_name, last_name in data:
        # Track person
        full_name = f"{first_name} {last_name}"
        names[person_id] = full_name
        person_counts[person_id] = person_counts.get(person_id, 0) + 1

        # Check if file exists
        if not os.path.exists(image_path):
            error_count += 1
            continue

        try:
            # Read image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"  ✗ Could not read: {image_path}")
                error_count += 1
                continue

            # Resize to standard size (MUST be consistent)
            img = cv2.resize(img, (200, 200))

            # Apply histogram equalization (improves contrast)
            img = cv2.equalizeHist(img)

            # Add to training data
            faces.append(img)
            labels.append(person_id)
            loaded_count += 1

        except Exception as e:
            print(f"  ✗ Error loading {image_path}: {e}")
            error_count += 1

    print(f"\n✓ Successfully loaded: {loaded_count} images")
    if error_count > 0:
        print(f"✗ Failed to load: {error_count} images")

    print(f"\nTraining data summary:")
    print("-" * 70)
    for person_id, count in sorted(person_counts.items()):
        print(f"  {names[person_id]}: {count} images")

    # Check if we have enough data
    if len(faces) < 5:
        print("\n✗ Not enough training data!")
        print("   Need at least 5 images total")
        conn.close()
        return False

    # Warn if any person has too few images
    for person_id, count in person_counts.items():
        if count < 10:
            print(f"⚠ Warning: {names[person_id]} has only {count} images (recommended: 10+)")

    # Create LBPH recognizer with optimized parameters
    print("\nTraining LBPH model...")
    print("Parameters:")
    print("  - Radius: 1")
    print("  - Neighbors: 8")
    print("  - Grid: 8x8")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    try:
        # Train the model
        recognizer.train(faces, np.array(labels))
        print("✓ Training complete!")

        # Save model
        os.makedirs('database', exist_ok=True)
        model_path = 'database/trained_model.yml'
        recognizer.write(model_path)
        print(f"✓ Model saved: {model_path}")

        # Save name mapping
        mapping_path = 'database/name_mapping.pkl'
        with open(mapping_path, 'wb') as f:
            pickle.dump(names, f)
        print(f"✓ Name mapping saved: {mapping_path}")

        # Log training session to database
        cursor.execute('''
            INSERT INTO training_history (total_persons, total_faces, model_path)
            VALUES (?, ?, ?)
        ''', (len(set(labels)), len(faces), model_path))
        conn.commit()

        print("\n" + "=" * 70)
        print("TRAINING SUCCESSFUL!")
        print("=" * 70)
        print(f"Total persons: {len(set(labels))}")
        print(f"Total images: {len(faces)}")
        print(f"Model ready for recognition!")
        print("\nNext step: Run main.py to start the attendance system")
        print("=" * 70)

        conn.close()
        return True

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        conn.close()
        return False


def verify_training_data():
    """Verify that training images exist and are readable"""
    print("\n" + "=" * 70)
    print("VERIFYING TRAINING DATA")
    print("=" * 70)

    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT p.person_id, p.first_name, p.last_name, COUNT(f.face_id) as count
        FROM persons p
        LEFT JOIN faces f ON p.person_id = f.person_id
        WHERE p.status = 'active'
        GROUP BY p.person_id
    ''')

    results = cursor.fetchall()

    print("\nStudent Image Counts:")
    print("-" * 70)
    total_images = 0
    for person_id, fname, lname, count in results:
        status = "✓" if count >= 10 else "⚠" if count >= 5 else "✗"
        print(f"{status} {fname} {lname}: {count} images")
        total_images += count

    print("-" * 70)
    print(f"Total: {total_images} images for {len(results)} students")

    if total_images < 5:
        print("\n✗ Not enough training data!")
        print("   Run capture_images.py to capture face images")
        return False

    conn.close()
    return True


if __name__ == "__main__":
    if not check_database_exists():
        sys.exit(1)

    # Verify training data first
    if not verify_training_data():
        sys.exit(1)

    # Ask for confirmation
    print("\nReady to train model with the above data.")
    response = input("Continue? (y/n): ").lower().strip()

    if response == 'y':
        success = train_model()
        sys.exit(0 if success else 1)
    else:
        print("Training cancelled")
        sys.exit(0)
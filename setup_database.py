#!/usr/bin/env python3
"""
Database Setup Script for Face Recognition Attendance System
Microcity College of Business and Technology
"""

import sqlite3
import os
from datetime import datetime


def create_database():
    """
    Create the SQLite database with all required tables
    This sets up the entire database structure for the attendance system
    """

    # Create database directory if it doesn't exist
    os.makedirs("database", exist_ok=True)
    os.makedirs("database/training_images", exist_ok=True)

    # Connect to SQLite database (creates file if doesn't exist)
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()

    # Create persons table - stores all student information
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            course TEXT,
            year_level TEXT,
            section TEXT,
            email TEXT,
            phone TEXT,
            date_enrolled TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active',
            photo_count INTEGER DEFAULT 0
        )
    ''')

    # Create faces table - stores paths to face images for training
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            face_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            face_encoding BLOB,
            image_path TEXT,
            capture_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            quality_score REAL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        )
    ''')

    # Create attendance logs table - records all attendance events
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            camera_id TEXT,
            location TEXT,
            status TEXT DEFAULT 'present',
            accuracy_score REAL,
            factors TEXT,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        )
    ''')

    # Create system settings table - stores configuration values
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create training history table - tracks model training sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_history (
            training_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_persons INTEGER,
            total_faces INTEGER,
            accuracy REAL,
            model_path TEXT
        )
    ''')

    conn.commit()
    print("✓ Database tables created successfully!")

    # Insert default system settings
    cursor.execute('''
        INSERT OR IGNORE INTO system_settings (setting_key, setting_value)
        VALUES ('recognition_threshold', '55')
    ''')

    cursor.execute('''
        INSERT OR IGNORE INTO system_settings (setting_key, setting_value)
        VALUES ('min_faces_per_person', '10')
    ''')

    conn.commit()
    conn.close()

    print("✓ Default settings configured!")


if __name__ == "__main__":
    print("=" * 70)
    print("FACE RECOGNITION DATABASE SETUP")
    print("Microcity College of Business and Technology")
    print("=" * 70)
    create_database()
    print("\nDatabase setup complete!")
    print("\nNext steps:")
    print("1. Run add_student.py to register students")
    print("2. Run capture_images.py to capture face images")
    print("3. Run train_model.py to train the recognition model")
    print("4. Run main.py to start the attendance system")
#!/usr/bin/env python3
"""
Add Student to Database
Registers new students in the face recognition system
"""

import sqlite3
import os
import sys
from datetime import datetime


def check_database_exists():
    """Check if database exists"""
    if not os.path.exists('database/faces.db'):
        print("ERROR: Database not found!")
        print("Please run setup_database.py first")
        return False
    return True


def add_student(student_info):
    """
    Add a new student to the database

    Args:
        student_info: Dictionary containing student details

    Returns:
        person_id: Database ID of newly added student
        folder_path: Path to student's image folder
    """

    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT INTO persons 
            (student_id, first_name, last_name, course, year_level, section, email, phone)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            student_info['student_id'],
            student_info['first_name'],
            student_info['last_name'],
            student_info['course'],
            student_info['year_level'],
            student_info['section'],
            student_info.get('email', ''),
            student_info.get('phone', '')
        ))

        conn.commit()
        person_id = cursor.lastrowid

        # Create folder for student's training images
        folder_name = f"Person_{person_id}_{student_info['first_name']}_{student_info['last_name']}"
        folder_path = os.path.join("database", "training_images", folder_name)
        os.makedirs(folder_path, exist_ok=True)

        print(f"\n✓ Student added successfully!")
        print(f"  Person ID: {person_id}")
        print(f"  Name: {student_info['first_name']} {student_info['last_name']}")
        print(f"  Student ID: {student_info['student_id']}")
        print(f"  Folder: {folder_path}")

        return person_id, folder_path

    except sqlite3.IntegrityError:
        print(f"\n✗ Error: Student ID '{student_info['student_id']}' already exists!")
        return None, None
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None, None
    finally:
        conn.close()


def add_group_members():
    """Add all group members to the database"""

    group_members = [
        {
            'student_id': '2210003M',
            'first_name': 'Eisley',
            'last_name': 'Atienza',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'eisley.atienza@microcity.edu.ph',
            'phone': '0967-676-6767'
        },
        {
            'student_id': '2210014M',
            'first_name': 'Jahanna',
            'last_name': 'Castro',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'jahnna.castro@microcity.edu.ph',
            'phone': '0938-674-6084'
        },
        {
            'student_id': '2210012M',
            'first_name': 'Diotanico',
            'last_name': 'Padilla',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'diotanico.padilla@microcity.edu.ph',
            'phone': '0938-722-4395'
        },
        {
            'student_id': '2210017M',
            'first_name': 'Betty Maye',
            'last_name': 'Eugenio',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'bettymayeeugenio0526@gmail.com',
            'phone': '0994-207-9840'
        },
        {
            'student_id': '2210016M',
            'first_name': 'Ace Joshua',
            'last_name': 'Abad',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'aceabad2300@gmail.com',
            'phone': '0919-386-7357'
        },
        {
            'student_id': '2021-006',
            'first_name': 'Richard Enock',
            'last_name': 'Catalan',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'enock.catalan@microcity.edu.ph',
            'phone': '0912-345-6789'
        },
        {
            'student_id': '2210010M',
            'first_name': 'Randel Krishna',
            'last_name': 'Bugay',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'randel.bugay@microcity.edu.ph',
            'phone': '0960-409-8354'
        },
        {
            'student_id': '2210013M',
            'first_name': 'Cyrrenz',
            'last_name': 'Paguirigan',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'cy.paguirigan@microcity.edu.ph',
            'phone': '0969-123-6967'
        },
    ]

    print("=" * 70)
    print("ADDING GROUP MEMBERS TO DATABASE")
    print("=" * 70)

    success_count = 0
    for member in group_members:
        result = add_student(member)
        if result[0] is not None:
            success_count += 1

    print("\n" + "=" * 70)
    print(f"Successfully added {success_count} out of {len(group_members)} students")
    print("=" * 70)
    print("\nNext step: Run capture_images.py to capture face images")


def add_single_student():
    """Interactive mode to add a single student"""
    print("=" * 70)
    print("ADD SINGLE STUDENT")
    print("=" * 70)

    student_info = {}

    print("\nEnter student information:")
    student_info['student_id'] = input("Student ID: ").strip()
    student_info['first_name'] = input("First Name: ").strip()
    student_info['last_name'] = input("Last Name: ").strip()
    student_info['course'] = input("Course: ").strip()
    student_info['year_level'] = input("Year Level (e.g., 4th Year): ").strip()
    student_info['section'] = input("Section: ").strip()
    student_info['email'] = input("Email (optional): ").strip()
    student_info['phone'] = input("Phone (optional): ").strip()

    add_student(student_info)


if __name__ == "__main__":
    if not check_database_exists():
        sys.exit(1)

    print("\n" + "=" * 70)
    print("STUDENT REGISTRATION SYSTEM")
    print("=" * 70)
    print("\nOptions:")
    print("1. Add all group members")
    print("2. Add single student")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == '1':
        add_group_members()
    elif choice == '2':
        add_single_student()
    else:
        print("Invalid choice!")
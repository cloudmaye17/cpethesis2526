#!/usr/bin/env python3
"""
Complete Database Viewer for Face Recognition System
View all tables, statistics, and data
Microcity College of Business and Technology
"""

import sqlite3
import os
import sys
from datetime import datetime

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Note: pandas not available, using basic formatting")


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"{title.center(80)}")
    print("=" * 80)


def check_database_exists():
    """Check if database file exists"""
    if not os.path.exists('database/faces.db'):
        print("\n✗ ERROR: Database file not found!")
        print("   Location: database/faces.db")
        print("   Please run setup_database.py first")
        return False
    return True


def format_table(cursor, headers):
    """Format query results as a table without pandas"""
    results = cursor.fetchall()

    if not results:
        return None

    # Calculate column widths
    col_widths = [len(str(h)) for h in headers]
    for row in results:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    # Print header
    header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in results:
        print(" | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)))

    return len(results)


def view_all_students():
    """Display all registered students"""
    print_header("REGISTERED STUDENTS")

    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()

    try:
        if PANDAS_AVAILABLE:
            df = pd.read_sql_query('''
                SELECT 
                    person_id as ID,
                    student_id as "Student ID",
                    first_name || ' ' || last_name as "Full Name",
                    course as Course,
                    year_level as "Year",
                    section as Section,
                    photo_count as "Photos",
                    status as Status
                FROM persons
                ORDER BY person_id
            ''', conn)

            if df.empty:
                print("\nNo students found")
            else:
                print(f"\nTotal Students: {len(df)}")
                print("\n" + df.to_string(index=False))
                print(f"\nActive: {len(df[df['Status'] == 'active'])}")
                print(f"Total Photos: {df['Photos'].sum()}")
        else:
            cursor.execute('''
                SELECT person_id, student_id, first_name || ' ' || last_name,
                       course, year_level, section, photo_count, status
                FROM persons ORDER BY person_id
            ''')

            count = format_table(cursor,
                                 ['ID', 'Student ID', 'Name', 'Course', 'Year', 'Section', 'Photos', 'Status'])

            if count:
                print(f"\nTotal Students: {count}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        conn.close()


def view_all_faces():
    """Display all captured face images"""
    print_header("TRAINING FACE IMAGES")

    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT 
                f.face_id,
                p.first_name || ' ' || p.last_name as name,
                f.image_path,
                f.capture_date
            FROM faces f
            JOIN persons p ON f.person_id = p.person_id
            ORDER BY p.person_id, f.face_id
            LIMIT 50
        ''')

        count = format_table(cursor, ['Face ID', 'Student', 'Path', 'Captured'])

        if count:
            cursor.execute('SELECT COUNT(*) FROM faces')
            total = cursor.fetchone()[0]
            print(f"\nShowing first {min(count, 50)} of {total} images")

            # Check for missing files
            cursor.execute('SELECT image_path FROM faces')
            missing = sum(1 for (path,) in cursor.fetchall() if not os.path.exists(path))
            if missing > 0:
                print(f"⚠ Warning: {missing} image files are missing!")
        else:
            print("\nNo face images found")
            print("   Run capture_images.py to capture training images")

    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        conn.close()


def view_attendance_logs():
    """Display attendance records"""
    print_header("ATTENDANCE LOGS")

    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT 
                a.log_id,
                p.first_name || ' ' || p.last_name as name,
                a.timestamp,
                ROUND(a.confidence, 1) as conf,
                a.status
            FROM attendance_logs a
            JOIN persons p ON a.person_id = p.person_id
            ORDER BY a.timestamp DESC
            LIMIT 30
        ''')

        count = format_table(cursor, ['Log ID', 'Student', 'Timestamp', 'Conf%', 'Status'])

        if count:
            cursor.execute('SELECT COUNT(*) FROM attendance_logs')
            total = cursor.fetchone()[0]
            print(f"\nShowing latest {min(count, 30)} of {total} records")

            # Today's count
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute(f"SELECT COUNT(*) FROM attendance_logs WHERE DATE(timestamp) = ?", (today,))
            today_count = cursor.fetchone()[0]
            print(f"Today's attendance: {today_count} records")
        else:
            print("\nNo attendance records found")

    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        conn.close()


def view_training_history():
    """Display model training history"""
    print_header("MODEL TRAINING HISTORY")

    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT training_id, timestamp, total_persons, total_faces, model_path
            FROM training_history
            ORDER BY timestamp DESC
        ''')

        count = format_table(cursor, ['ID', 'Timestamp', 'Persons', 'Faces', 'Model Path'])

        if count:
            print(f"\nTotal training sessions: {count}")

            # Check if model exists
            if os.path.exists('database/trained_model.yml'):
                size = os.path.getsize('database/trained_model.yml') / 1024
                print(f"✓ Model file exists ({size:.1f} KB)")
            else:
                print("✗ Model file not found!")
        else:
            print("\nNo training history found")
            print("   Run train_model.py to train the model")

    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        conn.close()


def view_database_statistics():
    """Display overall database statistics"""
    print_header("DATABASE STATISTICS")

    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()

    try:
        print("\nRecord Counts:")

        cursor.execute("SELECT COUNT(*) FROM persons")
        print(f"  Students: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM faces")
        print(f"  Face Images: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM attendance_logs")
        print(f"  Attendance Logs: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM training_history")
        print(f"  Training Sessions: {cursor.fetchone()[0]}")

        # Database file size
        if os.path.exists('database/faces.db'):
            db_size = os.path.getsize('database/faces.db') / (1024 * 1024)
            print(f"  Database Size: {db_size:.2f} MB")

        print("\nData Integrity:")

        # Students without images
        cursor.execute('''
            SELECT COUNT(*) FROM persons p
            LEFT JOIN faces f ON p.person_id = f.person_id
            WHERE f.face_id IS NULL AND p.status = 'active'
        ''')
        no_images = cursor.fetchone()[0]

        if no_images > 0:
            print(f"  ⚠ {no_images} student(s) have no training images")
        else:
            print(f"  ✓ All students have training images")

        # Orphaned faces
        cursor.execute('''
            SELECT COUNT(*) FROM faces f
            LEFT JOIN persons p ON f.person_id = p.person_id
            WHERE p.person_id IS NULL
        ''')
        orphaned = cursor.fetchone()[0]

        if orphaned > 0:
            print(f"  ⚠ {orphaned} orphaned face records")
        else:
            print(f"  ✓ No orphaned face records")

    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        conn.close()


def export_to_csv():
    """Export database tables to CSV files"""
    print_header("EXPORT DATABASE TO CSV")

    if not PANDAS_AVAILABLE:
        print("\n✗ pandas is required for CSV export")
        print("   Install with: pip install pandas")
        return

    conn = sqlite3.connect('database/faces.db')

    try:
        export_dir = "database/exports"
        os.makedirs(export_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        tables = {
            'students': "SELECT * FROM persons",
            'faces': "SELECT * FROM faces",
            'attendance': "SELECT * FROM attendance_logs",
            'training': "SELECT * FROM training_history"
        }

        for name, query in tables.items():
            df = pd.read_sql_query(query, conn)
            filename = f"{export_dir}/{name}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"✓ {name.capitalize()}: {filename} ({len(df)} rows)")

        print(f"\n✓ All exports saved to: {export_dir}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        conn.close()


def main_menu():
    """Main interactive menu"""

    if not check_database_exists():
        return

    while True:
        print("\n" + "=" * 80)
        print("FACE RECOGNITION DATABASE VIEWER".center(80))
        print("Microcity College of Business and Technology".center(80))
        print("=" * 80)
        print("\nMenu Options:")
        print("  1. View All Students")
        print("  2. View All Face Images")
        print("  3. View Attendance Logs")
        print("  4. View Training History")
        print("  5. View Database Statistics")
        print("  6. Export to CSV")
        print("  0. Exit")

        try:
            choice = input("\nEnter choice: ").strip()

            if choice == '1':
                view_all_students()
            elif choice == '2':
                view_all_faces()
            elif choice == '3':
                view_attendance_logs()
            elif choice == '4':
                view_training_history()
            elif choice == '5':
                view_database_statistics()
            elif choice == '6':
                export_to_csv()
            elif choice == '0':
                print("\nGoodbye!")
                break
            else:
                print("\n✗ Invalid choice")

            input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main_menu()
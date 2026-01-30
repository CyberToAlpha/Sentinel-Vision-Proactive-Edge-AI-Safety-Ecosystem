import sqlite3
import datetime
import os
import cv2

class SafetyLogger:
    def __init__(self, db_path="safety_logs.db", runs_folder="violation_snapshots"):
        self.db_path = db_path
        self.runs_folder = runs_folder
        
        if not os.path.exists(runs_folder):
            os.makedirs(runs_folder)
            
        self.init_db()
        
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                violation_type TEXT,
                snapshot_path TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def log_violation(self, violation_type, frame):
        """
        Logs a violation to the DB and saves a snapshot.
        violation_type: str (e.g., "NoHelmet", "NoVest", "DangerZone")
        frame: BGR image
        """
        # Save snapshot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{violation_type}_{timestamp}.jpg"
        filepath = os.path.join(self.runs_folder, filename)
        
        cv2.imwrite(filepath, frame)
        
        # Insert into DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO violations (timestamp, violation_type, snapshot_path) VALUES (?, ?, ?)",
                       (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), violation_type, filepath))
        conn.commit()
        conn.close()
        
    def get_recent_violations(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, violation_type, snapshot_path FROM violations ORDER BY id DESC LIMIT ?", (limit,))
        data = cursor.fetchall()
        conn.close()
        return data

    def get_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT violation_type, COUNT(*) FROM violations GROUP BY violation_type")
        data = cursor.fetchall()
        conn.close()
        return dict(data)

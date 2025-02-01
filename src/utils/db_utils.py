import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path: str = 'data/races_new.db'):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            raise Exception(f"Database connection error: {e}")

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_horse_history(self, horse_name: str) -> List[Dict]:
        """Get complete race history for a horse"""
        query = """
        SELECT r.date, r.venue, r.race_no, r.distance_track,
               rr.horse_no, rr.jockey, rr.weight, rr.performance_score,
               rr.last_6_races
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        JOIN horses h ON rr.horse_id = h.horse_id
        WHERE h.name = ?
        ORDER BY r.date DESC
        """
        
        try:
            self.cursor.execute(query, (horse_name,))
            columns = ['date', 'venue', 'race_no', 'distance', 'horse_no',
                      'jockey', 'weight', 'performance', 'last_6']
            return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error fetching horse history: {e}")
            return []

    def get_track_statistics(self, venue: str) -> Dict:
        """Get comprehensive statistics for a track"""
        query = """
        SELECT r.distance_track,
               COUNT(*) as total_races,
               AVG(CAST(rr.performance_score AS FLOAT)) as avg_performance
        FROM races r
        JOIN race_results rr ON r.race_id = rr.race_id
        WHERE r.venue = ?
        GROUP BY r.distance_track
        """
        
        try:
            self.cursor.execute(query, (venue,))
            return {row[0]: {'races': row[1], 'avg_performance': row[2]}
                    for row in self.cursor.fetchall()}
        except sqlite3.Error as e:
            print(f"Error fetching track statistics: {e}")
            return {}

    def get_jockey_performance(self, jockey: str) -> Dict:
        """Get jockey's performance statistics"""
        query = """
        SELECT COUNT(*) as total_rides,
               SUM(CASE WHEN CAST(rr.horse_no AS INTEGER) = 1 THEN 1 ELSE 0 END) as wins,
               AVG(CAST(rr.performance_score AS FLOAT)) as avg_performance
        FROM race_results rr
        WHERE rr.jockey = ?
        """
        
        try:
            self.cursor.execute(query, (jockey,))
            result = self.cursor.fetchone()
            return {
                'total_rides': result[0],
                'wins': result[1],
                'win_rate': (result[1] / result[0] * 100) if result[0] > 0 else 0,
                'avg_performance': result[2]
            }
        except sqlite3.Error as e:
            print(f"Error fetching jockey performance: {e}")
            return {}

    def get_recent_form(self, horse_name: str, last_n: int = 6) -> List[Dict]:
        """Get detailed recent form for a horse"""
        query = """
        SELECT r.date, r.venue, rr.horse_no, rr.performance_score
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        JOIN horses h ON rr.horse_id = h.horse_id
        WHERE h.name = ?
        ORDER BY r.date DESC
        LIMIT ?
        """
        
        try:
            self.cursor.execute(query, (horse_name, last_n))
            columns = ['date', 'venue', 'position', 'performance']
            return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error fetching recent form: {e}")
            return []

    def get_head_to_head(self, horse1: str, horse2: str) -> Dict:
        """Get head-to-head statistics between two horses"""
        query = """
        WITH horse_races AS (
            SELECT r.race_id, r.date, h.name, rr.horse_no
            FROM race_results rr
            JOIN races r ON rr.race_id = r.race_id
            JOIN horses h ON rr.horse_id = h.horse_id
            WHERE h.name IN (?, ?)
        )
        SELECT 
            hr1.date,
            hr1.horse_no as horse1_pos,
            hr2.horse_no as horse2_pos
        FROM horse_races hr1
        JOIN horse_races hr2 ON hr1.race_id = hr2.race_id
        WHERE hr1.name = ? AND hr2.name = ?
        ORDER BY hr1.date DESC
        """
        
        try:
            self.cursor.execute(query, (horse1, horse2, horse1, horse2))
            meetings = self.cursor.fetchall()
            
            stats = {
                'total_meetings': len(meetings),
                'wins_horse1': 0,
                'wins_horse2': 0,
                'better_finish_horse1': 0
            }
            
            for meeting in meetings:
                pos1 = int(meeting[1])
                pos2 = int(meeting[2])
                
                if pos1 == 1: stats['wins_horse1'] += 1
                if pos2 == 1: stats['wins_horse2'] += 1
                if pos1 < pos2: stats['better_finish_horse1'] += 1
            
            return stats
        except sqlite3.Error as e:
            print(f"Error fetching head-to-head statistics: {e}")
            return {'total_meetings': 0, 'wins_horse1': 0, 'wins_horse2': 0, 'better_finish_horse1': 0}
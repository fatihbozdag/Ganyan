import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3
from typing import List, Dict

def create_visualizations():
    """Create visualizations from the database"""
    conn = sqlite3.connect('horse_racing.db')
    
    # Load data
    races_df = pd.read_sql_query("SELECT * FROM races", conn)
    results_df = pd.read_sql_query("""
        SELECT r.*, rc.track, rc.distance, rc.track_condition
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
    """, conn)
    
    if len(races_df) == 0:
        print("No data found in database. Please run the scraper first.")
        conn.close()
        return
    
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn')
    
    # 1. Race Distribution by Track
    plt.figure(figsize=(12, 6))
    races_df['track'].value_counts().plot(kind='bar')
    plt.title('Number of Races by Track')
    plt.xlabel('Track')
    plt.ylabel('Number of Races')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/processed/races_by_track.png')
    
    if len(results_df) > 0:
        # 2. Finish Position Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results_df, x='finish_position', bins=20)
        plt.title('Distribution of Finish Positions')
        plt.xlabel('Finish Position')
        plt.ylabel('Count')
        plt.savefig('data/processed/finish_positions.png')
        
        # 3. Weight vs Finish Position
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x='weight', y='finish_position', alpha=0.5)
        plt.title('Weight vs Finish Position')
        plt.xlabel('Weight')
        plt.ylabel('Finish Position')
        plt.savefig('data/processed/weight_vs_position.png')
        
        # 4. Performance by Track Condition
        if 'finish_time' in results_df.columns and results_df['finish_time'].notna().any():
            plt.figure(figsize=(12, 6))
            avg_time_by_condition = results_df.groupby('track_condition')['finish_time'].mean()
            avg_time_by_condition.plot(kind='bar')
            plt.title('Average Finish Time by Track Condition')
            plt.xlabel('Track Condition')
            plt.ylabel('Average Finish Time')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('data/processed/track_condition_performance.png')
    
    conn.close()
    print("Visualizations have been saved to data/processed/")

if __name__ == "__main__":
    create_visualizations() 
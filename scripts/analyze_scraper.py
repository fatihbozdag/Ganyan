import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from scrapers.tjk_scraper import TJKScraper
import json
import pandas as pd
import sqlite3

def analyze_single_day(date_str):
    """Analyze the structure and content of a single day's race data"""
    scraper = TJKScraper()
    
    print(f"\n=== Analyzing races for {date_str} ===\n")
    
    # Get races for the day
    races = scraper.scrape_daily_races(date_str)
    
    print(f"Found {len(races)} races\n")
    
    if not races:
        print("No races found for this date")
        return
    
    # Analyze first race in detail
    print("=== Sample Race Analysis ===")
    sample_race = races[0]
    print("\nRace Info Structure:")
    for key, value in sample_race.items():
        if key != 'results':
            print(f"{key}: {value} ({type(value).__name__})")
    
    if 'results' in sample_race:
        print("\nResults Structure:")
        sample_result = sample_race['results'][0]
        for key, value in sample_result.items():
            print(f"{key}: {value} ({type(value).__name__})")
        
        # Create DataFrame for better analysis
        results_df = pd.DataFrame(sample_race['results'])
        print("\nResults Summary Statistics:")
        print(results_df.describe())
        
        # Check for missing values
        print("\nMissing Values Count:")
        print(results_df.isnull().sum())
        
        # Value distributions
        print("\nUnique Values per Column:")
        for column in results_df.columns:
            unique_vals = results_df[column].nunique()
            print(f"{column}: {unique_vals} unique values")
    
    # Save sample data for reference
    with open('data/raw/sample_race_data.json', 'w', encoding='utf-8') as f:
        json.dump(sample_race, f, ensure_ascii=False, indent=2)
    
    print("\n=== Data Quality Checks ===")
    check_data_quality(races)

def check_data_quality(races):
    """Perform basic data quality checks"""
    issues = []
    
    for race in races:
        # Check required fields
        required_fields = ['race_id', 'track', 'race_no', 'distance']
        for field in required_fields:
            if not race.get(field):
                issues.append(f"Missing required field: {field} in race {race.get('race_id', 'unknown')}")
        
        # Check results structure
        if 'results' in race:
            for idx, result in enumerate(race['results']):
                # Check for missing horse names
                if not result.get('horse_name'):
                    issues.append(f"Missing horse name in race {race.get('race_id')} result {idx}")
                
                # Check finish position validity
                pos = result.get('finish_position')
                if pos is not None and (not isinstance(pos, int) or pos < 1):
                    issues.append(f"Invalid finish position {pos} in race {race.get('race_id')}")
                
                # Check weight format
                weight = result.get('weight')
                if weight is not None and (not isinstance(weight, (int, float)) or weight < 20 or weight > 100):
                    issues.append(f"Suspicious weight value {weight} in race {race.get('race_id')}")
    
    if issues:
        print("\nFound following data quality issues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nNo major data quality issues found")

def analyze_database():
    """Analyze the entire database"""
    conn = sqlite3.connect('horse_racing.db')
    
    # Get overall statistics
    races_df = pd.read_sql_query("SELECT * FROM races", conn)
    results_df = pd.read_sql_query("SELECT * FROM results", conn)
    
    print("\n=== Database Analysis ===")
    print(f"\nTotal Races: {len(races_df)}")
    print(f"Total Results: {len(results_df)}")
    
    if len(races_df) > 0:
        print("\nRaces by Track:")
        print(races_df['track'].value_counts())
        
        print("\nAverage Horses per Race:", len(results_df) / len(races_df))
        
        if len(results_df) > 0:
            print("\nTop Jockeys by Wins:")
            wins_by_jockey = results_df[results_df['finish_position'] == 1]['jockey'].value_counts().head(10)
            print(wins_by_jockey)
    else:
        print("\nNo races found in database. Please run the scraper first.")
    
    conn.close()

def main():
    # Analyze today's races
    today = datetime.now().strftime('%d/%m/%Y')
    analyze_single_day(today)
    
    # Analyze entire database
    print("\nAnalyzing entire database...")
    analyze_database()

if __name__ == "__main__":
    main() 
import sqlite3
import os
import glob
import pandas as pd
from datetime import datetime

def create_db_schema(conn):
    """Create the database schema with proper indexes and constraints"""
    c = conn.cursor()
    
    # Drop existing tables if they exist
    c.execute('DROP TABLE IF EXISTS race_results')
    c.execute('DROP TABLE IF EXISTS races')
    c.execute('DROP TABLE IF EXISTS horses')
    
    # Create races table
    c.execute('''
        CREATE TABLE races (
            race_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            venue TEXT NOT NULL,
            race_no TEXT NOT NULL,
            race_type TEXT,
            horse_type TEXT,
            weight TEXT,
            distance_track TEXT,
            race_day TEXT,
            UNIQUE(date, venue, race_no)
        )
    ''')
    
    # Create horses table
    c.execute('''
        CREATE TABLE horses (
            horse_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age TEXT,
            origin TEXT,
            owner_trainer TEXT,
            UNIQUE(name, origin)
        )
    ''')
    
    # Create race_results table
    c.execute('''
        CREATE TABLE race_results (
            result_id TEXT PRIMARY KEY,
            race_id INTEGER,
            horse_id INTEGER,
            horse_no TEXT,
            jockey TEXT,
            weight TEXT,
            start_position TEXT,
            performance_score TEXT,
            last_6_races TEXT,
            score_1 TEXT,
            score_2 TEXT,
            score_3 TEXT,
            score_4 TEXT,
            score_5 TEXT,
            score_6 TEXT,
            FOREIGN KEY (race_id) REFERENCES races (race_id),
            FOREIGN KEY (horse_id) REFERENCES horses (horse_id)
        )
    ''')
    
    conn.commit()

def get_or_create_horse(cursor, name, age, origin, owner_trainer):
    """Get existing horse ID or create new horse entry"""
    cursor.execute('''
        SELECT horse_id FROM horses 
        WHERE name = ? AND origin = ?
    ''', (name, origin))
    
    result = cursor.fetchone()
    if result:
        return result[0]
    
    cursor.execute('''
        INSERT INTO horses (name, age, origin, owner_trainer)
        VALUES (?, ?, ?, ?)
    ''', (name, age, origin, owner_trainer))
    
    return cursor.lastrowid

def safe_float(value):
    """Safely convert value to float"""
    try:
        return float(value) if value and value.strip() != '' else None
    except (ValueError, TypeError):
        return None

def process_csv_file(file_path, conn):
    """Process a single CSV file and insert data into the database"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Basic validation
        required_columns = ['file_date', 'venue', 'race_no', 'race_type', 'horse_type', 'weight', 'distance_track',
                          'horse_name', 'age', 'origin', 'owner_trainer']
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns in {file_path}")
            return False
        
        c = conn.cursor()
        
        # Extract date from filename
        date_str = os.path.basename(file_path).split('-')[0]
        date = datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y-%m-%d')
        
        # Get first row values safely
        first_row = df.iloc[0] if not df.empty else None
        if first_row is None:
            print(f"Empty DataFrame for {file_path}")
            return False
            
        # Insert race and get race_id
        try:
            c.execute('''
                INSERT INTO races (date, venue, race_no, race_type, horse_type, weight, distance_track, race_day)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date,
                str(first_row['venue']),
                str(first_row['race_no']),
                str(first_row['race_type']),
                str(first_row['horse_type']),
                str(first_row['weight']),
                str(first_row['distance_track']),
                str(first_row['race_day']) if 'race_day' in df.columns else None
            ))
            race_id = c.lastrowid
            
            if not race_id:
                # If INSERT OR IGNORE skipped due to duplicate, get existing race_id
                c.execute('SELECT race_id FROM races WHERE date = ? AND venue = ? AND race_no = ?',
                         (date, str(first_row['venue']), str(first_row['race_no'])))
                result = c.fetchone()
                if result:
                    race_id = result[0]
                else:
                    print(f"Failed to get race_id for {file_path}")
                    return False
            
        except Exception as e:
            print(f"Error inserting race for {file_path}: {str(e)}")
            return False
        
        # Process each horse
        for idx, row in df.iterrows():
            try:
                # Convert row values to strings safely and clean them
                horse_name = str(row['horse_name']).strip() if pd.notna(row['horse_name']) else 'Unknown Horse'
                if horse_name == 'nan' or not horse_name:
                    horse_name = 'Unknown Horse'
                
                age = str(row['age']).strip() if pd.notna(row['age']) else ''
                if age == 'nan':
                    age = ''
                    
                origin = str(row['origin']).strip() if pd.notna(row['origin']) else ''
                if origin == 'nan':
                    origin = ''
                    
                owner_trainer = str(row['owner_trainer']).strip() if pd.notna(row['owner_trainer']) else ''
                if owner_trainer == 'nan':
                    owner_trainer = ''
                
                # Skip if we don't have a valid horse name
                if horse_name == 'Unknown Horse' and not origin:
                    print(f"Skipping invalid horse entry in {file_path}")
                    continue
                
                # Insert horse
                c.execute('''
                    INSERT OR IGNORE INTO horses (name, age, origin, owner_trainer)
                    VALUES (?, ?, ?, ?)
                ''', (horse_name, age, origin, owner_trainer))
                
                # Get horse_id
                c.execute('SELECT horse_id FROM horses WHERE name = ? AND origin = ?',
                         (horse_name, origin))
                result = c.fetchone()
                if not result:
                    print(f"Failed to get horse_id for {horse_name} in {file_path}")
                    continue
                
                horse_id = result[0]
                
                # Generate unique result_id by combining race_id and horse_no
                horse_no = str(row['horse_no']).strip() if pd.notna(row['horse_no']) else str(idx + 1)
                if horse_no == 'nan':
                    horse_no = str(idx + 1)
                result_id = f"{race_id}_{horse_no}"
                
                # Clean and prepare race result data
                jockey = str(row.get('jockey', '')).strip()
                jockey = None if jockey in ('nan', '') else jockey
                
                weight = str(row.get('weight', '')).strip()
                weight = None if weight in ('nan', '') else weight
                
                start_pos = str(row.get('starting_position', '')).strip()
                start_pos = None if start_pos in ('nan', '') else start_pos
                
                perf_score = str(row.get('performance_score', '')).strip()
                perf_score = None if perf_score in ('nan', '') else perf_score
                
                last_6 = str(row.get('last_6_races', '')).strip()
                last_6 = None if last_6 in ('nan', '') else last_6
                
                # Get score fields
                scores = []
                for score_type in ['kgs_score', 's20_score', 'eid_score', 'gny_score', 'agf_score']:
                    score = str(row.get(score_type, '')).strip()
                    scores.append(None if score in ('nan', '') else score)
                
                # Insert race result
                c.execute('''
                    INSERT OR IGNORE INTO race_results (
                        result_id, race_id, horse_id, horse_no, jockey, weight,
                        start_position, performance_score, last_6_races,
                        score_1, score_2, score_3, score_4, score_5, score_6
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_id,
                    race_id,
                    horse_id,
                    horse_no,
                    jockey,
                    weight,
                    start_pos,
                    perf_score,
                    last_6,
                    *scores,
                    None  # score_6 is not in our mapping
                ))
                
            except Exception as e:
                print(f"Error processing horse {horse_name} in {file_path}: {str(e)}")
                continue
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        conn.rollback()
        return False

def main():
    """Main function to process all files and create the database"""
    print("Creating new database from processed CSV files...")
    
    # Create a single database connection for the entire process
    conn = sqlite3.connect('races_new.db')
    
    # Create schema
    create_db_schema(conn)
    
    processed_dir = 'data/processed'
    total_files = 0
    successful_files = 0
    failed_files = 0
    
    for file in os.listdir(processed_dir):
        if file.endswith('.csv'):
            total_files += 1
            file_path = os.path.join(processed_dir, file)
            if process_csv_file(file_path, conn):
                successful_files += 1
            else:
                failed_files += 1
    
    print("\nDatabase creation completed!")
    print(f"Total files processed: {total_files}")
    print(f"Successfully processed: {successful_files}")
    print(f"Failed: {failed_files}")
    
    # Print database statistics
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM races')
    total_races = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM horses')
    total_horses = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM race_results')
    total_results = c.fetchone()[0]
    
    print("\nDatabase Statistics:")
    print(f"Total Races: {total_races}")
    print(f"Unique Horses: {total_horses}")
    print(f"Race Results: {total_results}")
    
    # Close the connection at the very end
    conn.close()

if __name__ == "__main__":
    main() 
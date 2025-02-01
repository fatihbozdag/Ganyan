import os
import pandas as pd
import re
from datetime import datetime

def is_date_line(line):
    """Check if a line contains a date in either format"""
    # Turkish format: DD.MM.YYYY
    turkish_pattern = r'\d{2}\.\d{2}\.\d{4}'
    # International format: DD/MM/YYYY
    international_pattern = r'\d{2}/\d{2}/\d{4}'
    
    return bool(re.search(turkish_pattern, line) or re.search(international_pattern, line))

def parse_header(line):
    """Parse header line to extract track and date information."""
    try:
        if ';' in line:
            parts = line.strip().split(';')
            track = parts[0].strip()
            date_str = parts[2].strip()
            # Convert date from DD/MM/YYYY to YYYY-MM-DD
            day, month, year = date_str.split('/')
            date = f"{year}-{month}-{day}"
            return track, date
        return None, None
    except Exception as e:
        print(f"Error parsing header: {e}")
        return None, None

def parse_race_info(line):
    """Parse race information from a line."""
    try:
        if ';' in line:
            parts = line.strip().split(';')
            race_info = parts[0].strip()
            if 'Kosu' in race_info:
                race_num = race_info.split('.')[0].strip()
                race_time = race_info.split(':')[1].strip()
                
                # Extract distance and surface
                distance = None
                surface = None
                for part in parts:
                    if 'm;' in part:
                        distance_part = part.strip()
                        distance = distance_part.split('m;')[0].strip()
                    if any(s in part.lower() for s in ['kum', 'çim']):
                        surface = part.strip()
                        if ';' in surface:
                            surface = surface.split(';')[0].strip()
                
                return {
                    'race_number': race_num,
                    'race_time': race_time,
                    'distance': distance,
                    'surface': surface
                }
        return None
    except Exception as e:
        print(f"Error parsing race info: {e}")
        return None

def parse_horse_entry(line):
    """Parse horse entry information from a line."""
    try:
        if ';' in line:
            parts = [p.strip() for p in line.split(';') if p.strip()]
            if len(parts) >= 10 and not any(header in line for header in ['At No', 'İkramiye']):
                horse_no = parts[0]
                horse_name = parts[1]
                age = parts[2]
                sire = parts[3]
                dam = parts[4]
                weight = parts[5]
                jockey = parts[6]
                owner = parts[7]
                trainer = parts[8]
                
                # Initialize optional fields
                start_pos = None
                hp_score = None
                last_6 = None
                kgs_score = None
                s20_score = None
                finish_time = None
                odds = None
                
                # Extract additional fields if available
                for i in range(9, len(parts)):
                    part = parts[i].strip()
                    if part.startswith('St'):
                        start_pos = part.replace('St', '').strip()
                    elif part.startswith('H'):
                        hp_score = part.replace('H', '').strip()
                    elif ':' in part and len(part.split(':')) == 2:
                        finish_time = part.strip()
                    elif ',' in part and 'TL' not in part:
                        odds = part.strip()
                
                return {
                    'horse_no': horse_no,
                    'horse_name': horse_name,
                    'age': age,
                    'sire': sire,
                    'dam': dam,
                    'weight': weight,
                    'jockey': jockey,
                    'owner': owner,
                    'trainer': trainer,
                    'start_pos': start_pos,
                    'hp_score': hp_score,
                    'last_6': last_6,
                    'kgs_score': kgs_score,
                    's20_score': s20_score,
                    'finish_time': finish_time,
                    'odds': odds
                }
        return None
    except Exception as e:
        print(f"Error parsing horse entry: {e}")
        return None

def standardize_race_file(input_file, output_dir='data/processed'):
    """Convert race file to standardized CSV format."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize variables
        track = None
        date = None
        current_race_info = None
        entries = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Try to parse header
                if track is None and date is None:
                    track, date = parse_header(line)
                    continue
                
                # Try to parse race info
                race_info = parse_race_info(line)
                if race_info:
                    current_race_info = race_info
                    continue
                
                # Try to parse horse entry
                entry = parse_horse_entry(line)
                if entry and current_race_info:
                    entry['track'] = track
                    entry['date'] = date
                    entry['race_number'] = current_race_info['race_number']
                    entry['race_time'] = current_race_info['race_time']
                    entry['distance'] = current_race_info['distance']
                    entry['surface'] = current_race_info['surface']
                    entries.append(entry)
        
        if not entries:
            print(f"No valid entries found in {input_file}")
            return False
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(entries)
        
        # Rename columns to match database requirements
        column_mapping = {
            'horse_no': 'horse_no',
            'horse_name': 'horse_name',
            'age': 'age',
            'sire': 'origin',  # Combine sire and dam
            'weight': 'weight',
            'jockey': 'jockey',
            'owner': 'owner_trainer',  # Combine owner and trainer
            'track': 'venue',
            'date': 'file_date',
            'race_number': 'race_no',
            'distance': 'distance_track',
            'surface': 'race_type'
        }
        
        # Select and rename columns
        df = df[list(column_mapping.keys())]
        df = df.rename(columns=column_mapping)
        
        # Combine sire and dam into origin if both exist
        if 'dam' in df.columns:
            df['origin'] = df.apply(lambda x: f"{x['sire']} - {x['dam']}" if pd.notna(x['dam']) else x['sire'], axis=1)
            df = df.drop('dam', axis=1)
        
        # Combine owner and trainer if both exist
        if 'trainer' in df.columns:
            df['owner_trainer'] = df.apply(lambda x: f"{x['owner']} / {x['trainer']}" if pd.notna(x['trainer']) else x['owner'], axis=1)
            df = df.drop('trainer', axis=1)
        
        # Add missing required columns with default values
        df['horse_type'] = df['age'].apply(lambda x: 'English' if any(char.isascii() for char in str(x)) else 'Arab')
        df['race_day'] = pd.to_datetime(df['file_date']).dt.strftime('%A')
        
        # Save to CSV
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        df.to_csv(output_file, index=False)
        print(f"Successfully processed: {input_file}")
        return True
        
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return False

def process_all_files(input_dir='data/raw', output_dir='data/processed'):
    """Process all CSV files in the input directory"""
    success_count = 0
    error_count = 0
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                input_file = os.path.join(root, file)
                try:
                    if standardize_race_file(input_file, output_dir):
                        success_count += 1
                    else:
                        print(f"No valid entries found in: {input_file}")
                        error_count += 1
                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")
                    error_count += 1
    
    print(f"\nProcessing complete. Successfully processed {success_count} files. Errors in {error_count} files.")

if __name__ == "__main__":
    process_all_files() 
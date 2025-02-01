import os
import pandas as pd
import re
from datetime import datetime

def parse_header(line):
    """Parse the track and date information from the header line."""
    parts = line.strip().split(';')
    track = parts[0]
    date_str = parts[2]
    date = datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
    return track, date

def parse_race_info(line):
    """Parse race information from the race header line."""
    parts = line.split(';')
    race_info = parts[0].split(' : ')
    race_num = int(race_info[0].split('.')[0])
    
    # Extract time
    time_match = re.search(r'(\d{2}:\d{2})', race_info[1])
    race_time = time_match.group(1) if time_match else None
    
    # Parse conditions
    conditions = parts[1] if len(parts) > 1 else ''
    
    # Parse distance and surface
    distance_surface = parts[4] if len(parts) > 4 else ''
    distance_match = re.search(r'(\d+)m', distance_surface)
    distance = int(distance_match.group(1)) if distance_match else None
    surface = 'Çim' if 'Çim' in distance_surface else 'Kum'
    
    return {
        'race_number': race_num,
        'race_time': race_time,
        'conditions': conditions.strip(),
        'distance': distance,
        'surface': surface
    }

def parse_horse_entry(line):
    """Parse horse entry information."""
    if not line or ';' not in line:
        return None
        
    parts = line.split(';')
    if len(parts) < 10:
        return None
        
    try:
        return {
            'horse_number': int(parts[0]),
            'horse_name': parts[1].strip(),
            'age': parts[2].strip(),
            'sire': parts[3].strip(),
            'dam': parts[4].strip(),
            'weight': parts[5].strip(),
            'jockey': parts[6].strip(),
            'owner': parts[7].strip(),
            'trainer': parts[8].strip(),
            'start_pos': parts[9].strip(),
            'finish_time': parts[12].strip() if len(parts) > 12 else None,
            'odds': parts[13].strip() if len(parts) > 13 else None
        }
    except (ValueError, IndexError):
        return None

def standardize_race_file(input_file, output_dir='data/processed'):
    """Convert a race file to standardized CSV format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    
    races = []
    current_race = None
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Parse header
    track, date = parse_header(lines[0])
    
    # Process each line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a race header
        if '. Kosu :' in line:
            if current_race:
                races.extend(current_race)
            race_info = parse_race_info(line)
            current_race = []
            continue
            
        # Skip prize money and column header lines
        if 'İkramiye' in line or 'At No;At İsmi;' in line:
            continue
            
        # Parse horse entry
        entry = parse_horse_entry(line)
        if entry:
            entry.update({
                'track': track,
                'date': date,
                'race_number': race_info['race_number'],
                'race_time': race_info['race_time'],
                'conditions': race_info['conditions'],
                'distance': race_info['distance'],
                'surface': race_info['surface']
            })
            current_race.append(entry)
    
    # Add last race
    if current_race:
        races.extend(current_race)
        
    if not races:
        print(f"No valid entries found in {input_file}")
        return False
        
    # Convert to DataFrame and save
    df = pd.DataFrame(races)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Successfully processed {input_file}")
    return True

def process_all_files(input_dir='data/raw', output_dir='data/processed'):
    """Process all CSV files in the input directory."""
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
                        error_count += 1
                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")
                    error_count += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {success_count} files")
    print(f"Errors: {error_count} files")

if __name__ == '__main__':
    process_all_files() 
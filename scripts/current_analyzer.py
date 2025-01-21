import pandas as pd
import numpy as np
from datetime import datetime
import os
from race_analyzer import RaceAnalyzer

def parse_historical_race(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Parse header information
        race_info = {}
        header_parts = lines[0].strip().split(';')
        race_info['track'] = header_parts[0]
        race_info['date'] = header_parts[2]
        
        # Find the line with horse data headers
        for i, line in enumerate(lines):
            if 'At No;At İsmi;Yaş;' in line:
                header_line = i
                break
        else:
            print(f"Could not find horse data headers in {file_path}")
            return None, None
            
        # Parse horse entries
        horses = []
        for line in lines[header_line + 1:]:
            if line.strip() and not line.startswith('['):  # Skip empty lines and result lines
                fields = line.strip().split(';')
                if len(fields) >= 10:  # Ensure we have enough fields
                    horse = {
                        'horse_number': fields[0],
                        'horse_name': fields[1],
                        'age': fields[2],
                        'sire': fields[3],
                        'dam': fields[4],
                        'weight': fields[5],
                        'jockey': fields[6],
                        'owner': fields[7],
                        'trainer': fields[8],
                        'finish_time': fields[12] if len(fields) > 12 else None,
                        'odds': fields[13] if len(fields) > 13 else None
                    }
                    horses.append(horse)
                
        return race_info, horses
        
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        return None, None

def load_historical_data(years=None):
    if years is None:
        years = ['2020', '2021', '2022', '2023', '2024']
    
    historical_data = []
    
    for year in years:
        year_dir = f'data/raw/{year}'
        if not os.path.exists(year_dir):
            continue
            
        for file_name in os.listdir(year_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(year_dir, file_name)
                race_info, horses = parse_historical_race(file_path)
                if race_info and horses:
                    historical_data.append({
                        'race_info': race_info,
                        'horses': horses
                    })
    
    print(f"Loaded {len(historical_data)} historical races")
    return historical_data

def analyze_current_race(current_race, historical_data):
    # Create analyzer instance
    analyzer = RaceAnalyzer(historical_data)
    
    # Extract race info and entries
    race_info = {
        'track': current_race['track'],
        'date': current_race['date'],
        'distance': current_race['distance'],
        'track_condition': current_race['track_condition']
    }
    
    race_entries = current_race['horses']
    
    # Analyze race
    predictions = analyzer.analyze_race(race_entries, race_info)
    return predictions

def main():
    # Load historical data
    print("\nLoading historical data...")
    historical_data = load_historical_data()
    
    if not historical_data:
        print("No historical data found")
        
    # Load and analyze current race
    print("\nAnalyzing current race...")
    current_race = {
        'track': 'İstanbul',
        'date': '22/01/2025',
        'distance': '1400m',
        'track_condition': 'Kum',
        'horses': [
            {
                'horse_number': '1',
                'horse_name': 'RABATTO',
                'age': '7y',
                'weight': '61',
                'jockey': 'İSMAİL YILDIRIM',
                'recent_form': '3 5 1 7 4 3',
                'handicap': '85',
                'best_time': '1.25.05'
            },
            {
                'horse_number': '2',
                'horse_name': 'RAGE OF THE WIND',
                'age': '6y',
                'weight': '62.5',
                'jockey': 'HALİL POLAT',
                'recent_form': '4 7 8 9 5 9',
                'handicap': '84',
                'best_time': '1.24.56'
            },
            {
                'horse_number': '3',
                'horse_name': 'BÜLENT',
                'age': '6y',
                'weight': '61.5',
                'jockey': 'MEHMET KAYA',
                'recent_form': '3 5 9 5 7 5',
                'handicap': '82',
                'best_time': '1.24.91'
            },
            {
                'horse_number': '4',
                'horse_name': 'İSEN BUGA',
                'age': '5y',
                'weight': '57.5',
                'jockey': 'ENES BOZDAĞ',
                'recent_form': '2 5 5 7 1 6',
                'handicap': '78',
                'best_time': '1.23.07'
            },
            {
                'horse_number': '5',
                'horse_name': 'NICE',
                'age': '5y',
                'weight': '59.5',
                'jockey': 'ERCAN ÇANKAYA',
                'recent_form': '5 3 3 7 4 6',
                'handicap': '78',
                'best_time': '1.25.08'
            },
            {
                'horse_number': '6',
                'horse_name': 'FEELING GOOD',
                'age': '4y',
                'weight': '57',
                'jockey': 'AYHAN KURŞUN',
                'recent_form': '0 4 4 4 7 1',
                'handicap': '75',
                'best_time': '1.24.24'
            },
            {
                'horse_number': '7',
                'horse_name': 'CRIMEAN WOLF',
                'age': '4y',
                'weight': '54',
                'jockey': 'UĞUR TEMUR',
                'recent_form': '6 8 9 8 8 7',
                'handicap': '69',
                'best_time': '1.24.78'
            },
            {
                'horse_number': '8',
                'horse_name': 'LONDON POWER',
                'age': '5y',
                'weight': '54.5',
                'jockey': 'BURAK AKÇAY',
                'recent_form': '5 6 6 1 3 3',
                'handicap': '68',
                'best_time': '1.24.05'
            },
            {
                'horse_number': '9',
                'horse_name': 'MELEYS',
                'age': '4y',
                'weight': '50.5',
                'jockey': 'TAMER BALİ',
                'recent_form': '4 6 3 8 8 9',
                'handicap': '62',
                'best_time': '1.26.13'
            },
            {
                'horse_number': '10',
                'horse_name': 'KING EREN',
                'age': '4y',
                'weight': '50.5',
                'jockey': 'CANER TEPE',
                'recent_form': '3 7 4 7 3 2',
                'handicap': '62',
                'best_time': '1.25.73'
            },
            {
                'horse_number': '11',
                'horse_name': 'KING FATİH',
                'age': '4y',
                'weight': '50',
                'jockey': 'İBRAHİM SEFA SÖĞÜT',
                'recent_form': '7 5 3 9 1 4',
                'handicap': '54',
                'best_time': '1.26.45'
            }
        ]
    }
    
    predictions = analyze_current_race(current_race, historical_data)
    
    # Display predictions
    print("\nPredictions:")
    for pred in predictions:
        print(f"{pred['horse_name']}: {pred['win_probability']:.2%}")
        if pred['factors']:
            print("Factors:")
            for factor in pred['factors']:
                print(f"  - {factor}")
        print()

if __name__ == "__main__":
    main() 
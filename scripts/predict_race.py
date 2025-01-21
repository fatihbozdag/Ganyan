import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import re

class RacePredictor:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.horse_history = {}
        self.jockey_history = {}
        self.track_history = {}
        self.surface_types = {'K': 'Kum', 'Ç': 'Çim', 'S': 'Sentetik'}
        self.load_historical_data()
        
    def load_historical_data(self):
        """Load and process historical race data"""
        for year in range(2021, 2026):
            year_dir = os.path.join(self.data_dir, str(year))
            if not os.path.exists(year_dir):
                continue
                
            for file in glob.glob(os.path.join(year_dir, '*.csv')):
                try:
                    self._process_race_file(file)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    def _process_race_file(self, file_path):
        """Process a single race file and update histories"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract track and date from first line
            header_parts = lines[0].strip().split(';')
            track = header_parts[0].strip()
            date = header_parts[2].strip() if len(header_parts) > 2 else None
            
            current_race = None
            race_horses = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(';')
                
                # Race header line
                if 'Kosu' in line:
                    if current_race and race_horses:
                        self._process_race_entries(current_race, race_horses)
                    race_horses = []
                    
                    # Parse race details
                    current_race = {
                        'track': track,
                        'date': date,
                        'time': parts[0].split(':')[1].strip() if ':' in parts[0] else '',
                        'distance': parts[4].strip() if len(parts) > 4 else '',
                        'surface': parts[5].strip() if len(parts) > 5 else 'kum'
                    }
                
                # Horse entry line
                elif current_race and 'At No' not in line and len(parts) >= 7:
                    try:
                        horse_entry = {
                            'number': parts[0].strip(),
                            'name': parts[1].strip(),
                            'age': parts[2].strip(),
                            'sire': parts[3].strip(),
                            'dam': parts[4].strip(),
                            'weight': parts[5].strip().split()[0] if parts[5].strip() else '0',
                            'jockey': parts[6].strip(),
                            'finish_pos': parts[0].strip() if not parts[0].strip().isalpha() else '',
                            'recent_form': parts[7].strip() if len(parts) > 7 else ''
                        }
                        race_horses.append(horse_entry)
                    except Exception as e:
                        print(f"Error parsing horse entry: {e}")
                        continue
            
            # Process last race
            if current_race and race_horses:
                self._process_race_entries(current_race, race_horses)
                
        except Exception as e:
            print(f"Error in _process_race_file: {e}")
    
    def _process_race_entries(self, race, horses):
        """Process entries for a single race"""
        for horse in horses:
            # Update horse history
            if horse['name'] not in self.horse_history:
                self.horse_history[horse['name']] = []
            
            self.horse_history[horse['name']].append({
                'date': race['date'],
                'track': race['track'],
                'distance': race['distance'],
                'surface': race['surface'],
                'finish_pos': horse['finish_pos'],
                'weight': horse['weight'],
                'jockey': horse['jockey']
            })
            
            # Update jockey history
            if horse['jockey'] not in self.jockey_history:
                self.jockey_history[horse['jockey']] = []
            
            self.jockey_history[horse['jockey']].append({
                'date': race['date'],
                'track': race['track'],
                'horse': horse['name'],
                'finish_pos': horse['finish_pos']
            })
    
    def _parse_recent_form(self, form_string):
        """Parse recent form string to extract surface types and positions"""
        if not form_string:
            return []
        
        # Regular expression to match surface type and position
        pattern = r'([KÇS])(\d+)'
        matches = re.findall(pattern, form_string)
        
        return [{'surface': self.surface_types.get(surface, 'Unknown'),
                'position': int(position)} 
                for surface, position in matches]

    def predict_race(self, race_entries):
        """Analyze chances for each horse in upcoming race"""
        predictions = []
        for horse in race_entries:
            score = 0
            factors = []
            
            # 1. Historical Performance
            horse_name = horse['name']
            if horse_name in self.horse_history:
                history = self.horse_history[horse_name]
                
                # Recent performance
                recent_races = sorted(history, key=lambda x: x['date'])[-5:]
                if recent_races:
                    # Track compatibility
                    track_matches = sum(1 for race in recent_races if race['track'] == horse['track'])
                    track_factor = track_matches * 2
                    score += track_factor
                    if track_factor > 0:
                        factors.append(f"Track experience: {track_factor}")
                    
                    # Distance compatibility
                    dist_matches = sum(1 for race in recent_races if race['distance'] == horse['distance'])
                    dist_factor = dist_matches * 1.5
                    score += dist_factor
                    if dist_factor > 0:
                        factors.append(f"Distance experience: {dist_factor}")
            
            # 2. Surface Preference Analysis
            if 'recent_form' in horse:
                form_data = self._parse_recent_form(horse['recent_form'])
                if form_data:
                    # Calculate surface-specific performance
                    surface_stats = {}
                    for entry in form_data:
                        surface = entry['surface']
                        if surface not in surface_stats:
                            surface_stats[surface] = {'races': 0, 'positions': []}
                        surface_stats[surface]['races'] += 1
                        surface_stats[surface]['positions'].append(entry['position'])
                    
                    # Calculate performance factor for current surface
                    current_surface = horse.get('surface', 'Kum')
                    if current_surface in surface_stats:
                        stats = surface_stats[current_surface]
                        avg_pos = sum(stats['positions']) / len(stats['positions'])
                        win_rate = sum(1 for pos in stats['positions'] if pos == 1) / len(stats['positions'])
                        
                        surface_factor = (win_rate * 5) + ((10 - avg_pos) / 2)
                        score += surface_factor
                        factors.append(f"Surface preference ({current_surface}): {surface_factor:.1f}")
            
            # 3. Jockey Performance
            jockey = horse['jockey']
            if jockey in self.jockey_history:
                jockey_races = self.jockey_history[jockey]
                recent_jockey = sorted(jockey_races, key=lambda x: x['date'])[-10:]
                if recent_jockey:
                    win_count = sum(1 for race in recent_jockey if race['finish_pos'] == '1')
                    win_rate = win_count / len(recent_jockey)
                    jockey_factor = win_rate * 10
                    score += jockey_factor
                    if jockey_factor > 0:
                        factors.append(f"Jockey performance: {jockey_factor:.1f}")
            
            # 4. Weight Analysis
            try:
                weight = float(horse['weight'].replace(',', '.'))
                weight_factor = (60 - weight) * 0.5
                score += weight_factor
                factors.append(f"Weight factor: {weight_factor:.1f}")
            except (ValueError, TypeError):
                pass
            
            predictions.append({
                'horse_name': horse_name,
                'score': max(0, score),
                'win_probability': 0,
                'factors': factors
            })
        
        # Normalize scores to probabilities
        total_score = sum(p['score'] for p in predictions)
        if total_score > 0:
            for p in predictions:
                p['win_probability'] = (p['score'] / total_score) * 100
        else:
            prob = 100.0 / len(predictions)
            for p in predictions:
                p['win_probability'] = prob
        
        return pd.DataFrame(predictions)

def analyze_race(race_data):
    """Analyze a race using historical data"""
    predictor = RacePredictor()
    
    # Convert DataFrame rows to list of dictionaries
    race_entries = []
    for _, horse in race_data.iterrows():
        entry = {
            'name': horse['horse_name'],
            'track': horse['track'],
            'distance': horse['distance'],
            'weight': str(horse['weight']),
            'jockey': horse['jockey'],
            'surface': horse.get('surface', 'kum'),
            'recent_form': horse.get('recent_form', '')
        }
        race_entries.append(entry)
    
    return predictor.predict_race(race_entries)

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict_race.py <race_data.csv>")
        return
        
    # Load race data
    try:
        race_data = pd.read_csv(sys.argv[1])
        
        # Make predictions
        predictions = analyze_race(race_data)
        
        # Display results
        print("\nRace Prediction Analysis")
        print("=======================")
        for _, pred in predictions.sort_values('win_probability', ascending=False).iterrows():
            print(f"\n{pred['horse_name']}")
            print(f"Win Probability: {pred['win_probability']:.1f}%")
            if pred['factors']:
                print("Factors:")
                for factor in pred['factors']:
                    print(f"  - {factor}")
    except Exception as e:
        print(f"Error analyzing race: {e}")

if __name__ == "__main__":
    main() 
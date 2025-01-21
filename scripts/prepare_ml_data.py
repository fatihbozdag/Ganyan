import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

class RaceDataPreprocessor:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.races_df = None
        self.results_df = None
        self.horse_history = {}  # Track horse performance history
        self.jockey_history = {}  # Track jockey performance history
        self.track_history = {}  # Track course-specific patterns
        
        # Initialize encoders
        self.track_encoder = LabelEncoder()
        self.condition_encoder = LabelEncoder()
        
    def parse_race_file(self, file_path):
        """Parse a single race file into structured data"""
        races = []
        results = []
        current_race = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            if not line.strip():
                continue
                
            fields = [f.strip() for f in line.split(';')]
            
            # Race header
            if 'Kosu' in line:
                race_no = int(fields[0].split('.')[0])
                race_time = fields[0].split(':')[-1].strip()
                
                # Extract date and track from filename
                date = file_path.split('-')[0].split('/')[-1]
                track = file_path.split('-')[1].split('.')[0]
                
                # Create race_id
                race_id = f"{date}_{track}_{race_no}"
                
                # Find the distance field by looking for 'm' suffix
                distance = None
                track_condition = None
                race_class = None
                horse_class = None
                weight_condition = None
                
                for field in fields[1:]:  # Skip the first field (race number)
                    if field.endswith('m'):
                        try:
                            distance = int(field.replace('m', ''))
                        except ValueError:
                            continue
                    elif field in ['Kum', 'Çim']:  # Track conditions
                        track_condition = field
                    elif 'kg' in field.lower():
                        weight_condition = field
                    elif any(x in field for x in ['Handikap', 'Maiden', 'ŞARTLI']):
                        race_class = field
                    elif any(x in field for x in ['Araplar', 'İngilizler', 'Yaşlı']):
                        horse_class = field
                
                current_race = {
                    'race_id': race_id,  # Add race_id to the race data
                    'date': date,
                    'track': track,
                    'race_no': race_no,
                    'race_time': race_time,
                    'race_class': race_class,
                    'horse_class': horse_class,
                    'weight_condition': weight_condition,
                    'distance': distance,
                    'track_condition': track_condition
                }
                races.append(current_race)
            
            # Horse result
            elif current_race and len(fields) >= 9 and fields[0].isdigit():
                try:
                    result = {
                        'race_id': current_race['race_id'],  # Use the race_id from current_race
                        'finish_position': int(fields[0]),
                        'horse_name': fields[1],
                        'age': int(fields[2].split('y')[0]) if 'y' in fields[2] else None,
                        'horse_type': fields[2][-1] if len(fields[2]) > 0 else None,  # a for Arabian, g for English
                        'sire': fields[3],
                        'dam': fields[4],
                        'weight': float(fields[5]) if fields[5].replace('.', '').isdigit() else None,
                        'jockey': fields[6],
                        'owner': fields[7],
                        'trainer': fields[8],
                        'finish_time': fields[12] if len(fields) > 12 else None,
                        'odds': float(fields[13].replace(',', '.')) if len(fields) > 13 and fields[13] else None,
                        'margin': fields[14] if len(fields) > 14 else None
                    }
                    results.append(result)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse result line in {file_path}: {e}")
                    continue
        
        return pd.DataFrame(races), pd.DataFrame(results)
    
    def load_all_data(self):
        """Load and combine all race data"""
        all_races = []
        all_results = []
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    races_df, results_df = self.parse_race_file(file_path)
                    all_races.append(races_df)
                    all_results.append(results_df)
        
        self.races_df = pd.concat(all_races, ignore_index=True)
        self.results_df = pd.concat(all_results, ignore_index=True)
    
    def calculate_horse_features(self):
        """Calculate historical performance features for each horse"""
        horse_features = {}
        
        for _, row in self.results_df.iterrows():
            horse = row['horse_name']
            if horse not in horse_features:
                horse_features[horse] = {
                    'races': 0,
                    'wins': 0,
                    'places': 0,  # Top 3 finishes
                    'avg_position': [],
                    'avg_odds': [],
                    'preferred_distance': [],
                    'preferred_track': [],
                    'best_time': None
                }
            
            stats = horse_features[horse]
            stats['races'] += 1
            stats['wins'] += 1 if row['finish_position'] == 1 else 0
            stats['places'] += 1 if row['finish_position'] <= 3 else 0
            stats['avg_position'].append(row['finish_position'])
            if row['odds']:
                stats['avg_odds'].append(row['odds'])
            
            # Get race details
            race = self.races_df[self.races_df['race_id'] == row['race_id']].iloc[0]
            stats['preferred_distance'].append(race['distance'])
            stats['preferred_track'].append(race['track'])
            
            if row['finish_time']:
                if not stats['best_time'] or row['finish_time'] < stats['best_time']:
                    stats['best_time'] = row['finish_time']
        
        return horse_features
    
    def calculate_jockey_features(self):
        """Calculate historical performance features for each jockey"""
        jockey_features = {}
        
        for _, row in self.results_df.iterrows():
            jockey = row['jockey']
            if jockey not in jockey_features:
                jockey_features[jockey] = {
                    'rides': 0,
                    'wins': 0,
                    'places': 0,
                    'track_wins': defaultdict(int),
                    'distance_wins': defaultdict(int),
                    'avg_odds': [],
                    'track_win_rates': defaultdict(list)
                }
            
            stats = jockey_features[jockey]
            stats['rides'] += 1
            stats['wins'] += 1 if row['finish_position'] == 1 else 0
            stats['places'] += 1 if row['finish_position'] <= 3 else 0
            if row['odds']:
                stats['avg_odds'].append(row['odds'])
            
            # Get race details for track-specific stats
            race = self.races_df[self.races_df['race_id'] == row['race_id']].iloc[0]
            if row['finish_position'] == 1:
                stats['track_wins'][race['track']] += 1
                stats['distance_wins'][race['distance']] += 1
            
            # Track win rate for each track
            stats['track_win_rates'][race['track']].append(1 if row['finish_position'] == 1 else 0)
        
        return jockey_features

    def calculate_trainer_features(self):
        """Calculate historical performance features for each trainer"""
        trainer_features = {}
        
        for _, row in self.results_df.iterrows():
            trainer = row['trainer']
            if trainer not in trainer_features:
                trainer_features[trainer] = {
                    'total_horses': 0,
                    'wins': 0,
                    'places': 0,
                    'horse_types': defaultdict(int),  # Count of Arabian vs English horses
                    'avg_performance': [],
                    'track_performance': defaultdict(list)
                }
            
            stats = trainer_features[trainer]
            stats['total_horses'] += 1
            stats['wins'] += 1 if row['finish_position'] == 1 else 0
            stats['places'] += 1 if row['finish_position'] <= 3 else 0
            stats['horse_types'][row['horse_type']] += 1
            stats['avg_performance'].append(row['finish_position'])
            
            # Track-specific performance
            race = self.races_df[self.races_df['race_id'] == row['race_id']].iloc[0]
            stats['track_performance'][race['track']].append(row['finish_position'])
        
        return trainer_features

    def calculate_track_features(self):
        """Calculate track-specific patterns and statistics"""
        track_features = {}
        
        for _, race in self.races_df.iterrows():
            track = race['track']
            if track not in track_features:
                track_features[track] = {
                    'total_races': 0,
                    'avg_field_size': [],
                    'surface_conditions': defaultdict(int),
                    'winning_posts': defaultdict(list),  # Track winning positions/times
                    'seasonal_patterns': defaultdict(list),  # Performance by season
                    'distance_patterns': defaultdict(list)  # Performance patterns by distance
                }
            
            stats = track_features[track]
            stats['total_races'] += 1
            
            # Get race results
            race_results = self.results_df[self.results_df['race_id'] == race['race_id']]
            stats['avg_field_size'].append(len(race_results))
            stats['surface_conditions'][race['track_condition']] += 1
            
            # Track winning times
            winner = race_results[race_results['finish_position'] == 1].iloc[0]
            if winner['finish_time']:
                stats['winning_posts'][race['distance']].append(winner['finish_time'])
            
            # Seasonal patterns (by month)
            month = datetime.strptime(race['date'], '%d.%m.%Y').month
            stats['seasonal_patterns'][month].append(len(race_results))
            
            # Distance patterns
            stats['distance_patterns'][race['distance']].extend(
                race_results['finish_position'].tolist()
            )
        
        return track_features

    def prepare_features(self):
        """Prepare features for machine learning"""
        # Calculate all historical statistics
        horse_features = self.calculate_horse_features()
        jockey_features = self.calculate_jockey_features()
        trainer_features = self.calculate_trainer_features()
        track_features = self.calculate_track_features()
        
        # Prepare feature matrix
        features = []
        for _, row in self.results_df.iterrows():
            race = self.races_df[self.races_df['race_id'] == row['race_id']].iloc[0]
            
            # Get all historical stats
            horse_stats = horse_features[row['horse_name']]
            jockey_stats = jockey_features[row['jockey']]
            trainer_stats = trainer_features[row['trainer']]
            track_stats = track_features[race['track']]
            
            # Calculate seasonal features
            race_month = datetime.strptime(race['date'], '%d.%m.%Y').month
            season = (race_month % 12 + 3) // 3  # Convert month to season (1-4)
            
            feature_dict = {
                # Basic race features
                'track': race['track'],
                'distance': race['distance'],
                'track_condition': race['track_condition'],
                'season': season,
                
                # Horse features
                'horse_age': row['age'],
                'horse_type': row['horse_type'],
                'weight': row['weight'],
                'odds': row['odds'],
                'horse_races': horse_stats['races'],
                'horse_wins': horse_stats['wins'],
                'horse_places': horse_stats['places'],
                'horse_win_rate': horse_stats['wins'] / horse_stats['races'] if horse_stats['races'] > 0 else 0,
                'horse_place_rate': horse_stats['places'] / horse_stats['races'] if horse_stats['races'] > 0 else 0,
                'horse_avg_position': np.mean(horse_stats['avg_position']),
                'horse_avg_odds': np.mean(horse_stats['avg_odds']) if horse_stats['avg_odds'] else None,
                
                # Jockey features
                'jockey_rides': jockey_stats['rides'],
                'jockey_wins': jockey_stats['wins'],
                'jockey_places': jockey_stats['places'],
                'jockey_win_rate': jockey_stats['wins'] / jockey_stats['rides'] if jockey_stats['rides'] > 0 else 0,
                'jockey_track_wins': jockey_stats['track_wins'].get(race['track'], 0),
                'jockey_distance_wins': jockey_stats['distance_wins'].get(race['distance'], 0),
                
                # Trainer features
                'trainer_horses': trainer_stats['total_horses'],
                'trainer_wins': trainer_stats['wins'],
                'trainer_win_rate': trainer_stats['wins'] / trainer_stats['total_horses'] if trainer_stats['total_horses'] > 0 else 0,
                'trainer_track_avg': np.mean(trainer_stats['track_performance'].get(race['track'], [0])),
                
                # Track features
                'track_avg_field_size': np.mean(track_stats['avg_field_size']),
                'track_season_avg_field': np.mean(track_stats['seasonal_patterns'].get(race_month, [0])),
                
                # Target variable
                'target': 1 if row['finish_position'] == 1 else 0
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)

def main():
    preprocessor = RaceDataPreprocessor()
    preprocessor.load_all_data()
    features_df = preprocessor.prepare_features()
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    features_df.to_csv('data/processed/ml_features.csv', index=False)
    print(f"Processed {len(features_df)} race entries")
    print("\nFeature statistics:")
    print(features_df.describe())

if __name__ == "__main__":
    main() 
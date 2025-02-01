import pandas as pd
import numpy as np
from scipy import stats
import os
import glob

class BayesianRaceAnalyzer:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.priors = {
            'weight_impact': stats.norm(0, 1),  # Normal distribution for weight impact
            'surface_performance': stats.beta(2, 2),  # Beta distribution for surface performance
            'distance_performance': stats.gamma(2, 2),  # Gamma distribution for distance ability
            'recent_form': stats.beta(2, 2),  # Beta distribution for form
            'handicap_effect': stats.norm(0, 1),
            'track_specific': stats.beta(2, 2)  # Added Istanbul-specific prior
        }
        self.historical_data = self.load_historical_data()
        
    def load_historical_data(self):
        """Load all historical race data focusing on Istanbul"""
        all_races = []
        for year in range(2021, 2026):  # Extended to 2021-2025
            year_dir = os.path.join(self.data_dir, str(year))
            if os.path.exists(year_dir):
                for file in glob.glob(os.path.join(year_dir, '*.csv')):
                    if 'İstanbul' in file:  # Focus on Istanbul races
                        try:
                            races = self.parse_race_file(file)
                            all_races.extend(races)
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
        print(f"Loaded {len(all_races)} historical races from Istanbul (2021-2025)")
        return all_races
    
    def parse_race_file(self, file_path):
        """Parse a single race file"""
        races = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_race = None
        for line in lines:
            if not line.strip():
                continue
                
            fields = [f.strip() for f in line.split(';')]
            
            if 'Kosu' in line:
                # Parse race header
                race_info = {
                    'track': 'İstanbul',
                    'distance': self.extract_distance(fields[4]) if len(fields) > 4 else None,
                    'surface': self.extract_surface(fields[5]) if len(fields) > 5 else None,
                    'class': fields[1] if len(fields) > 1 else None,
                    'date': self.extract_date(file_path)
                }
                current_race = race_info
                
            elif current_race and len(fields) > 10 and fields[0].isdigit():
                # Parse horse result
                result = {
                    'position': int(fields[0]),
                    'horse_name': fields[1],
                    'weight': self.extract_weight(fields[5]),
                    'finish_time': self.parse_time(fields[12]) if len(fields) > 12 else None,
                    'odds': float(fields[13].replace(',', '.')) if len(fields) > 13 and fields[13].strip() else None,
                    'jockey': fields[6] if len(fields) > 6 else None,
                    'trainer': fields[8] if len(fields) > 8 else None
                }
                current_race['result'] = result
                races.append(current_race.copy())
                
        return races
    
    @staticmethod
    def extract_date(file_path):
        """Extract date from filename"""
        try:
            date_str = os.path.basename(file_path).split('-')[0]
            return date_str
        except:
            return None
    
    @staticmethod
    def extract_distance(distance_str):
        """Extract distance in meters from string"""
        try:
            return int(''.join(filter(str.isdigit, distance_str)))
        except:
            return None
            
    @staticmethod
    def extract_surface(surface_str):
        """Extract surface type"""
        surface_map = {
            'Kum': 'dirt',
            'Çim': 'turf',
            'Sentetik': 'synthetic'
        }
        return surface_map.get(surface_str, 'unknown')
        
    @staticmethod
    def extract_weight(weight_str):
        """Extract weight in kg"""
        try:
            return float(weight_str.split()[0])
        except:
            return None
            
    @staticmethod
    def parse_time(time_str):
        """Parse finish time to seconds"""
        try:
            if ':' in time_str:
                minutes, seconds = time_str.split(':')
                return float(minutes) * 60 + float(seconds)
            return float(time_str)
        except:
            return None
    
    def calculate_surface_prior(self, horse_name, surface):
        """Calculate surface performance prior for Istanbul"""
        horse_races = [r for r in self.historical_data 
                      if r['result']['horse_name'] == horse_name 
                      and r['surface'] == surface]
        
        if not horse_races:
            return self.priors['surface_performance'].mean()
            
        positions = [r['result']['position'] for r in horse_races]
        win_rate = sum(1 for p in positions if p == 1) / len(positions)
        return win_rate
    
    def calculate_distance_prior(self, horse_name, distance):
        """Calculate distance aptitude prior for Istanbul"""
        horse_races = [r for r in self.historical_data 
                      if r['result']['horse_name'] == horse_name 
                      and abs(r['distance'] - distance) <= 200]
                      
        if not horse_races:
            return self.priors['distance_performance'].mean()
            
        positions = [r['result']['position'] for r in horse_races]
        performance = 1 - (sum(positions) / (len(positions) * 3))
        return max(0.1, performance)
    
    def calculate_weight_prior(self, horse_name, weight):
        """Calculate weight impact prior"""
        horse_races = [r for r in self.historical_data 
                      if r['result']['horse_name'] == horse_name]
                      
        if not horse_races:
            return self.priors['weight_impact'].mean()
            
        weights = [r['result']['weight'] for r in horse_races if r['result']['weight']]
        if not weights:
            return self.priors['weight_impact'].mean()
            
        avg_weight = sum(weights) / len(weights)
        return -0.1 * (weight - avg_weight)
    
    def calculate_form_prior(self, recent_form):
        """Calculate form prior from recent results"""
        if not recent_form:
            return self.priors['recent_form'].mean()
            
        positions = [int(pos) for pos in recent_form.split() if pos.isdigit()]
        if not positions:
            return self.priors['recent_form'].mean()
            
        weighted_positions = [pos * (1.2 ** i) for i, pos in enumerate(reversed(positions))]
        avg_position = sum(weighted_positions) / sum((1.2 ** i) for i in range(len(positions)))
        
        return max(0.1, 1 - (avg_position / 10))

    def calculate_track_specific_prior(self, horse_name):
        """Calculate Istanbul-specific performance prior"""
        horse_races = [r for r in self.historical_data 
                      if r['result']['horse_name'] == horse_name]
        
        if not horse_races:
            return self.priors['track_specific'].mean()
        
        positions = [r['result']['position'] for r in horse_races]
        performance = 1 - (sum(positions) / (len(positions) * 3))
        return max(0.1, performance)
    
    def predict_race(self, race_entries):
        """Make predictions using Bayesian analysis"""
        predictions = []
        
        for horse in race_entries:
            # Calculate priors
            surface_prior = self.calculate_surface_prior(horse['name'], horse['surface'])
            distance_prior = self.calculate_distance_prior(horse['name'], int(horse['distance']))
            weight_prior = self.calculate_weight_prior(horse['name'], float(horse['weight']))
            form_prior = self.calculate_form_prior(horse['recent_form'])
            track_prior = self.calculate_track_specific_prior(horse['name'])
            
            # Combine priors using Bayesian update with track-specific weight
            combined_score = (
                surface_prior * 0.25 +     # Surface aptitude
                distance_prior * 0.20 +     # Distance aptitude
                weight_prior * 0.15 +       # Weight impact
                form_prior * 0.25 +         # Recent form
                track_prior * 0.15          # Istanbul-specific performance
            )
            
            predictions.append({
                'horse_name': horse['name'],
                'win_probability': combined_score,
                'factors': {
                    'surface_performance': surface_prior,
                    'distance_aptitude': distance_prior,
                    'weight_impact': weight_prior,
                    'recent_form': form_prior,
                    'track_performance': track_prior
                }
            })
        
        # Normalize probabilities
        total_score = sum(p['win_probability'] for p in predictions)
        for p in predictions:
            p['win_probability'] = p['win_probability'] / total_score
            
        return sorted(predictions, key=lambda x: x['win_probability'], reverse=True)

def main():
    analyzer = BayesianRaceAnalyzer()
    
    # Race specifications
    race_entries = [
        {
            'name': 'UMUDUNU KAYBETME',
            'surface': 'synthetic',
            'distance': '2000',
            'weight': 62,
            'recent_form': '2 4 4 5 2 7',
            'jockey': 'MEHMET KAYA',
            'handicap': 39
        },
        {
            'name': 'BOĞUŞLU',
            'surface': 'synthetic',
            'distance': '2000',
            'weight': 60,
            'recent_form': '4 7 4 7 2 2',
            'jockey': 'AHMET ÇELİK',
            'handicap': 36
        },
        {
            'name': 'RÜZGARIMA KAPIL',
            'surface': 'synthetic',
            'distance': '2000',
            'weight': 59,
            'recent_form': '2 2 5 3 4 4',
            'jockey': 'ENES BOZDAĞ',
            'handicap': 38
        },
        {
            'name': 'AKALTON',
            'surface': 'synthetic',
            'distance': '2000',
            'weight': 53,
            'recent_form': '0 0 6 6 8 7',
            'jockey': 'MUSTAFA TEKPETER',
            'handicap': 14
        },
        {
            'name': 'KING PELE',
            'surface': 'synthetic',
            'distance': '2000',
            'weight': 58,
            'recent_form': '7 7 7 5',
            'jockey': 'AKIN SÖZEN',
            'handicap': 12
        },
        {
            'name': 'KIRMÜJDE',
            'surface': 'synthetic',
            'distance': '2000',
            'weight': 58,
            'recent_form': '8 0',
            'jockey': 'ERHAN AKTUĞ',
            'handicap': None
        },
        {
            'name': 'YAZASLANI',
            'surface': 'synthetic',
            'distance': '2000',
            'weight': 58,
            'recent_form': '4',
            'jockey': 'MERTCAN ÇELİK',
            'handicap': None
        },
        {
            'name': 'SONTURBO',
            'surface': 'synthetic',
            'distance': '2000',
            'weight': 53,
            'recent_form': '3 5 3 3 6 4',
            'jockey': 'EREN KADİRLER',
            'handicap': 29
        },
        {
            'name': 'TURBOLU',
            'surface': 'synthetic',
            'distance': '2000',
            'weight': 53,
            'recent_form': '7 8 7 5',
            'jockey': 'ZÜLFÜKAR KARABUL',
            'handicap': 15
        }
    ]
    
    # Make predictions
    predictions = analyzer.predict_race(race_entries)
    
    # Display results
    print("\nBayesian Race Analysis (Istanbul 2021-2025)")
    print("Race Details: Maiden/DHÖW, 4 Yaşlı Araplar, 2000m Sentetik")
    print("=====================================")
    for pred in predictions:
        print(f"\n{pred['horse_name']}")
        print(f"Win Probability: {pred['win_probability']:.1%}")
        print("Factor Breakdown:")
        for factor, value in pred['factors'].items():
            print(f"  - {factor}: {value:.3f}")
        print(f"Weight: {next((entry['weight'] for entry in race_entries if entry['name'] == pred['horse_name']))}")
        print(f"Jockey: {next((entry['jockey'] for entry in race_entries if entry['name'] == pred['horse_name']))}")
        if next((entry['handicap'] for entry in race_entries if entry['name'] == pred['horse_name'])) is not None:
            print(f"Handicap: {next((entry['handicap'] for entry in race_entries if entry['name'] == pred['horse_name']))}")

if __name__ == "__main__":
    main() 
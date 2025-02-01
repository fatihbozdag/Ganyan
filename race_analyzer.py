import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class RaceAnalyzer:
    def __init__(self, db_path='races_new.db'):
        """Initialize the race analyzer with database path"""
        self.db_path = db_path
        self.weights = {
            'speed': 0.30,        # Increased further for EİD importance
            'form': 0.25,         # Increased for recent form importance
            'class': 0.10,        # Reduced class factor weight
            'weight': 0.20,       # Maintained weight importance
            'track_fit': 0.15     # Slightly reduced
        }
        
        # Performance factors
        self.performance_weights = {
            'eid_factor': 1.2,    # Increased EİD impact
            'recent_form': 1.5,   # Increased recent form impact
            'weight_adj': 0.8,    # Reduced weight adjustment
            'kgs_factor': 1.1,    # Increased KGS impact
            's20_impact': 0.9     # Reduced S20 impact
        }
        
        # Load historical data
        print("\nLoading historical data...")
        self.historical_data = self.load_historical_data()
        
        # Initialize models
        self.scaler = StandardScaler()
        self.logistic_model = LogisticRegression(max_iter=1000)
        self.rf_model = RandomForestClassifier(n_estimators=100)
        self.nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

    def analyze_race(self, race_info, entries):
        """Analyze a race using historical data and current entries."""
        predictions = []
        total_base_points = 0
        
        # First pass - calculate base scores
        for entry in entries:
            prediction = {
                'horse_name': entry['name'],
                'base_score': 0,
                'speed_score': 0,
                'form_score': 0,
                'class_score': 0,
                'weight_score': 0,
                'track_score': 0
            }
            
            # Calculate individual factor scores
            prediction['speed_score'] = self.calculate_speed_score(entry) * self.weights['speed']
            prediction['form_score'] = self.calculate_form_score(entry) * self.weights['form']
            prediction['class_score'] = self.calculate_class_score(entry) * self.weights['class']
            prediction['weight_score'] = self.calculate_weight_score(entry) * self.weights['weight']
            prediction['track_score'] = self.calculate_track_score(entry) * self.weights['track_fit']
            
            # Calculate base score
            prediction['base_score'] = (
                prediction['speed_score'] +
                prediction['form_score'] +
                prediction['class_score'] +
                prediction['weight_score'] +
                prediction['track_score']
            )
            
            # Apply performance adjustments
            prediction['base_score'] *= self.calculate_performance_adjustment(entry)
            
            total_base_points += prediction['base_score']
            predictions.append(prediction)
        
        # Second pass - calculate total scores and probabilities
        for pred in predictions:
            # Calculate total score
            pred['total_score'] = pred['base_score']
            
            # Calculate win probability (avoid division by zero)
            if total_base_points > 0:
                pred['win_probability'] = round((pred['total_score'] / total_base_points) * 100, 1)
            else:
                pred['win_probability'] = 0
        
        # Sort by total score
        predictions.sort(key=lambda x: x['total_score'], reverse=True)
        return predictions

    def _extract_race_features(self, race_group):
        """Extract features from a race group for training"""
        try:
            features_list = []
            for _, horse in race_group.iterrows():
                features = [
                    horse['weight'],
                    horse.get('kgs', 0),
                    horse.get('s20', 0),
                    self._parse_recent_form(horse.get('recent_form', '')),
                    self._calculate_jockey_rating(horse['jockey']),
                    horse.get('handicap', 0)
                ]
                features_list.append(features)
            return features_list
        except Exception as e:
            print(f"Error extracting race features: {str(e)}")
            return None

    def _extract_horse_features(self, entry, history):
        """Extract features for a single horse"""
        try:
            return [
                entry.get('weight', 0),
                entry.get('kgs', 0),
                entry.get('s20', 0),
                self._parse_recent_form(entry.get('recent_form', '')),
                self._calculate_jockey_rating(entry.get('jockey', '')),
                entry.get('handicap', 0)
            ]
        except Exception as e:
            print(f"Error extracting horse features: {str(e)}")
            return None

    def _parse_recent_form(self, form_string):
        """Convert recent form string to numerical value"""
        if not form_string:
            return 0
            
        total = 0
        count = 0
        weight = 1.0
        
        for char in reversed(str(form_string)):
            if char.isdigit():
                total += (10 - int(char)) * weight
                count += 1
                weight *= 0.8  # Decrease weight for older results
                
        return total / count if count > 0 else 0

    def _calculate_jockey_rating(self, jockey_name):
        """Calculate jockey rating based on historical performance"""
        if not jockey_name or not self.historical_data is not None:
            return 0
            
        jockey_races = self.historical_data[self.historical_data['jockey'] == jockey_name]
        if len(jockey_races) == 0:
            return 0
            
        win_rate = len(jockey_races[jockey_races['position'] == 1]) / len(jockey_races)
        place_rate = len(jockey_races[jockey_races['position'] <= 3]) / len(jockey_races)
        
        return (win_rate * 0.7 + place_rate * 0.3) * 10

    def _calculate_baseline_probability(self, entry):
        """Calculate baseline probability when model prediction is not available"""
        base_prob = 11.0  # Base probability (1/9 for 9 horses)
        
        # Weight factor (higher weight might be advantageous in certain conditions)
        weight = float(entry.get('weight', 0))
        if weight >= 60:
            weight_factor = 2.0
        elif weight >= 55:
            weight_factor = 1.5
        else:
            weight_factor = 1.0
        
        # KGS factor (normalized to 0-5 range)
        kgs = float(entry.get('kgs', 0))
        kgs_factor = min((kgs / 70) * 5, 5.0)  # 70 is a reasonable max KGS
        
        # S20 factor (normalized to 0-5 range)
        s20 = float(entry.get('s20', 0))
        s20_factor = (s20 / 20) * 5
        
        # Recent form factor
        recent_form = entry.get('recent_form', '')
        form_factor = 0
        if recent_form:
            positions = [int(pos) for pos in str(recent_form) if pos.isdigit()]
            if positions:
                # Weight recent results more heavily
                weighted_positions = []
                weight = 1.0
                for pos in reversed(positions):
                    weighted_positions.append((10 - pos) * weight)
                    weight *= 0.8
                form_factor = (sum(weighted_positions) / len(weighted_positions)) / 2
        
        # Jockey factor
        jockey = entry.get('jockey', '').upper()
        jockey_factor = 0
        if 'AHMET ÇELİK' in jockey:
            jockey_factor = 3.0
        elif any(name in jockey for name in ['MEHMET KAYA', 'MERTCAN ÇELİK', 'AKIN SÖZEN']):
            jockey_factor = 2.0
        elif any(name in jockey for name in ['MUSTAFA', 'EREN', 'ENES']):
            jockey_factor = 1.5
        
        # Calculate total probability
        total_factors = weight_factor + kgs_factor + s20_factor + form_factor + jockey_factor
        win_prob = base_prob + (total_factors / 5)  # Normalize the impact of factors
        
        return win_prob  # No cap on baseline probability

    def _analyze_recent_performance(self, form_string):
        """Analyze recent performance trend from form string"""
        if not form_string:
            return 'No history'
            
        # Convert form string to list of positions
        positions = [int(pos) for pos in str(form_string) if pos.isdigit()]
        
        if len(positions) < 3:
            return 'Insufficient data'
            
        # Calculate trend
        if positions[0] < positions[1] < positions[2]:
            return 'Improving'
        elif positions[0] > positions[1] > positions[2]:
            return 'Declining'
        else:
            return 'Stable'

    def load_historical_data(self):
        """Load historical race data from the database."""
        query = """
            SELECT 
                r.race_id,
                r.date,
                r.venue as track,
                r.distance_track as distance,
                r.race_type,
                r.horse_type,
                r.weight as race_weight,
                rr.horse_no,
                rr.jockey,
                rr.weight,
                rr.start_position,
                rr.performance_score,
                rr.last_6_races,
                rr.score_1,
                rr.score_2,
                rr.score_3,
                rr.score_4,
                rr.score_5,
                rr.score_6,
                h.name,
                h.age,
                h.origin,
                h.owner_trainer
            FROM races r
            JOIN race_results rr ON r.race_id = rr.race_id
            JOIN horses h ON rr.horse_id = h.horse_id
            WHERE r.date >= date('now', '-2 years')
            ORDER BY r.date DESC
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def _find_horse_history(self, horse_name, df):
        """Find horse's history with fuzzy matching"""
        # Normalize the search name and remove common suffixes
        search_name = horse_name.replace('İ', 'I').replace('Ğ', 'G').replace('Ü', 'U').replace('Ş', 'S').replace('Ö', 'O').replace('Ç', 'C').upper()
        
        # Try exact match first
        history = df[df['normalized_name'] == search_name]
        
        # If no exact match, try matching with common suffixes
        if len(history) == 0:
            suffixes = ['KG', 'DB', 'SK', 'K', 'GKR', 'SKG', 'SGKR']
            for suffix in suffixes:
                # Try with suffix
                history = df[df['normalized_name'] == f"{search_name} {suffix}"]
                if len(history) > 0:
                    break
                
                # Try with multiple suffixes
                history = df[df['normalized_name'].str.startswith(f"{search_name} {suffix}", na=False)]
                if len(history) > 0:
                    break
        
        # If still no match and name has multiple words, try matching first word
        if len(history) == 0 and len(search_name.split()) > 1:
            first_word = search_name.split()[0]
            # Only match if it's a distinctive word (not common prefixes)
            if first_word not in ['KING', 'PRINCE', 'LADY', 'LORD', 'SUPER', 'MEGA', 'ULTRA']:
                history = df[df['normalized_name'].str.startswith(first_word + ' ', na=False)]
        
        return history 

    def calculate_speed_score(self, horse):
        """Calculate speed score based on EİD and historical times"""
        score = 0
        if 'eid' in horse and horse['eid']:
            try:
                time_parts = horse['eid'].split('.')
                if len(time_parts) == 2:
                    minutes = int(time_parts[0])
                    seconds = float(time_parts[1])
                    total_seconds = minutes * 60 + seconds
                    # Better times get higher scores
                    score = max(0, (150 - total_seconds) / 10)
            except:
                pass
        return score

    def calculate_form_score(self, horse):
        """Calculate form score based on recent results"""
        score = 0
        if 'last_six' in horse and horse['last_six']:
            positions = [int(pos) for pos in str(horse['last_six']) if pos.isdigit()]
            if positions:
                # Weight recent results more heavily
                weights = [1.5, 1.3, 1.1, 0.9, 0.7, 0.5][:len(positions)]
                weighted_positions = [pos * w for pos, w in zip(positions, weights)]
                avg_position = sum(weighted_positions) / sum(weights[:len(positions)])
                score = max(0, (10 - avg_position))
        return score

    def calculate_class_score(self, horse):
        """Calculate class score based on HP"""
        score = 0
        if 'hp' in horse and horse['hp']:
            try:
                hp = int(horse['hp'])
                score = hp / 10  # Scale HP to reasonable range
            except:
                pass
        return score

    def calculate_weight_score(self, horse):
        """Calculate weight score"""
        score = 0
        if 'weight' in horse and horse['weight']:
            try:
                weight = float(horse['weight'])
                # Lower weights get higher scores
                score = max(0, (62 - weight))  # Assume 62kg is maximum
            except:
                pass
        return score

    def calculate_track_score(self, horse):
        """Calculate track score with emphasis on S20"""
        score = 0
        if 's20' in horse and horse['s20']:
            try:
                s20 = int(horse['s20'])
                score = s20 / 2  # Scale S20 to reasonable range
            except:
                pass
        return score

    def calculate_performance_adjustment(self, entry):
        """Calculate performance adjustment factor based on recent metrics"""
        adjustment = 1.0
        
        # EİD time adjustment
        if 'eid' in entry and entry['eid']:
            try:
                time_parts = entry['eid'].split('.')
                if len(time_parts) == 2:
                    minutes = int(time_parts[0])
                    seconds = float(time_parts[1])
                    total_seconds = minutes * 60 + seconds
                    # Better times get higher adjustment
                    adjustment *= 1 + (self.performance_weights['eid_factor'] * (150 - total_seconds) / 150)
            except:
                pass

        # Recent form adjustment
        if 'last_six' in entry and entry['last_six']:
            positions = [int(pos) for pos in str(entry['last_six']) if pos.isdigit()]
            if positions:
                recent_avg = sum(positions) / len(positions)
                # Better average position gets higher adjustment
                form_adj = 1 + (self.performance_weights['recent_form'] * (10 - recent_avg) / 10)
                adjustment *= max(0.5, form_adj)  # Cap minimum at 0.5

        # Weight adjustment
        if 'weight' in entry and entry['weight']:
            try:
                weight = float(entry['weight'])
                # Lighter weights get higher adjustment
                weight_adj = 1 + (self.performance_weights['weight_adj'] * (62 - weight) / 62)
                adjustment *= max(0.7, weight_adj)  # Cap minimum at 0.7
            except:
                pass

        # KGS adjustment
        if 'kgs' in entry and entry['kgs']:
            try:
                kgs = int(entry['kgs'])
                optimal_kgs = 21  # Optimal days between races
                kgs_diff = abs(kgs - optimal_kgs)
                # Closer to optimal gets higher adjustment
                kgs_adj = 1 + (self.performance_weights['kgs_factor'] * (30 - kgs_diff) / 30)
                adjustment *= max(0.8, kgs_adj)  # Cap minimum at 0.8
            except:
                pass

        # S20 adjustment
        if 's20' in entry and entry['s20']:
            try:
                s20 = int(entry['s20'])
                # Higher S20 gets higher adjustment
                s20_adj = 1 + (self.performance_weights['s20_impact'] * s20 / 20)
                adjustment *= max(0.9, s20_adj)  # Cap minimum at 0.9
            except:
                pass

        return adjustment 
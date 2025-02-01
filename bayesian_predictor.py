import sqlite3
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler

class BayesianPredictor:
    def __init__(self, db_path='races_new.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.scaler = StandardScaler()
        
        # Prior weights
        self.prior_weights = {
            'speed_prior': 0.35,      # Increased speed importance
            'form_prior': 0.30,       # Increased form importance
            'class_prior': 0.15,      # Reduced class importance
            'weight_prior': 0.20      # Maintained weight importance
        }
        
        # Performance adjustments
        self.performance_factors = {
            'eid_impact': 1.3,        # Increased EİD importance
            'recent_form_boost': 1.4,  # Increased recent form impact
            'weight_penalty': 0.7,     # Reduced weight penalty
            'days_since_race': 1.2,    # Increased KGS importance
            'track_fitness': 0.8       # Reduced track fitness impact
        }
        
        # Prior parameters (to be updated with historical data)
        self.prior_mu = None  # Prior mean
        self.prior_sigma = None  # Prior standard deviation
        self.feature_weights = None  # Weights for different features
        
    def get_horse_history(self, horse_name):
        """Get historical performance data for a horse"""
        query = """
        SELECT 
            h.name,
            h.weight,
            h.start_position,
            h.hp,
            h.last_six,
            h.kgs,
            h.s20,
            h.eid,
            h.agf
        FROM current_race_data h
        WHERE h.name LIKE ?
        LIMIT 10
        """
        try:
            df = pd.read_sql_query(query, self.conn, params=(f"%{horse_name}%",))
            return df
        except:
            return pd.DataFrame()  # Return empty DataFrame if query fails
        
    def calculate_speed_figure(self, finish_time, distance, weight, track_condition='good', class_level=0):
        """Calculate normalized speed figure with track condition and class adjustments"""
        if pd.isna(finish_time) or pd.isna(distance) or pd.isna(weight):
            return None
        
        # Base calculation
        speed = distance / finish_time
        
        # Weight adjustment (0.1% per kg difference from standard)
        weight_adj = 1 + (weight - 58) * 0.001
        
        # Track condition adjustment
        condition_factors = {
            'heavy': 1.03,  # Slower times expected
            'soft': 1.02,
            'good': 1.0,
            'firm': 0.99  # Faster times expected
        }
        track_adj = condition_factors.get(track_condition.lower(), 1.0)
        
        # Class level adjustment (higher class races get a bonus)
        class_adj = 1 + (class_level * 0.01)
        
        # Apply all adjustments
        speed_fig = speed * weight_adj * track_adj * class_adj
        
        return speed_fig
        
    def calculate_form_cycle(self, recent_results, track_conditions=None):
        """Calculate form cycle from recent results with track condition consideration"""
        if not recent_results or len(recent_results) == 0:
            return 0
            
        weights = np.exp(-np.arange(len(recent_results)) * 0.5)  # Exponential decay
        
        # Adjust weights based on track conditions if available
        if track_conditions:
            condition_factors = {
                'heavy': 0.8,  # Less relevant for different conditions
                'soft': 0.9,
                'good': 1.0,
                'firm': 0.9
            }
            for i, condition in enumerate(track_conditions):
                weights[i] *= condition_factors.get(condition.lower(), 1.0)
        
        weighted_results = np.sum(weights * recent_results) / np.sum(weights)
        return weighted_results
        
    def calculate_class_factor(self, horse_entry):
        """Calculate class factor based on horse's entry and history"""
        history = self.get_horse_history(horse_entry['name'])
        
        # Get current class from HP rating
        current_class = horse_entry.get('hp', 0)
        if current_class is None:
            current_class = 0
        
        if len(history) == 0:
            return 1.0
        
        # Get average class level from history
        avg_class = history['class_level'].mean()
        
        if pd.isna(avg_class):
            return 1.0
        
        # Calculate factor based on class difference
        class_diff = current_class - avg_class
        
        # Positive factor if dropping in class, negative if moving up
        return 1 + (class_diff * -0.05)  # 5% adjustment per class level

    def calculate_track_factor(self, horse_entry):
        """Calculate track factor based on horse's entry and history"""
        history = self.get_horse_history(horse_entry['name'])
        
        if len(history) == 0:
            return 1.0
        
        # Get performance in similar conditions
        similar_condition_results = history[
            history['track_condition'].str.lower() == 'good'  # Assuming good track
        ]
        
        if len(similar_condition_results) == 0:
            return 1.0
        
        # Calculate average finish position in similar conditions
        avg_finish = similar_condition_results['finish_position'].mean()
        overall_avg = history['finish_position'].mean()
        
        if pd.isna(avg_finish) or pd.isna(overall_avg):
            return 1.0
        
        # Better performance in these conditions gets a bonus
        return 1 + (overall_avg - avg_finish) * 0.05
        
    def update_priors(self, horse_histories):
        """Update prior parameters based on historical data"""
        all_speed_figs = []
        
        for history in horse_histories:
            if len(history) > 0:
                speed_figs = [
                    self.calculate_speed_figure(
                        row['finish_time'], 
                        row['distance'], 
                        row['weight'],
                        row.get('track_condition', 'good'),
                        row.get('class_level', 0)
                    )
                    for _, row in history.iterrows()
                ]
                speed_figs = [sf for sf in speed_figs if sf is not None]
                all_speed_figs.extend(speed_figs)
        
        if len(all_speed_figs) > 0:
            self.prior_mu = np.mean(all_speed_figs)
            self.prior_sigma = np.std(all_speed_figs)
        else:
            # Default priors if no historical data
            self.prior_mu = 16.0  # Average speed figure
            self.prior_sigma = 2.0  # Conservative spread
            
        # Initialize feature weights with track condition and class importance
        self.feature_weights = {
            'recent_form': 0.20,  # Reduced slightly
            'weight_advantage': 0.20,  # Increased
            'distance_aptitude': 0.15,  # Unchanged
            'surface_preference': 0.10,  # Unchanged
            'class_level': 0.10,  # Reduced
            'track_condition': 0.10,  # Unchanged
            'performance_metrics': 0.15  # Increased for S20 and EİD
        }
        
    def normalize_probabilities(self, probabilities):
        # Shift all probabilities to be non-negative
        min_prob = min(probabilities.values())
        if min_prob < 0:
            shift = abs(min_prob) + 0.01  # Add small buffer
            probabilities = {k: v + shift for k, v in probabilities.items()}
        
        # Normalize to sum to 100%
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: (v / total) * 100 for k, v in probabilities.items()}
        
        return probabilities

    def calculate_win_probability(self, horse):
        """Calculate win probability for a horse using Bayesian approach"""
        # Base probability starts at prior mean
        prob = self.prior_mu if self.prior_mu else 16.0
        
        # Recent form adjustment with trajectory weighting
        if 'recent_form' in horse and horse['recent_form']:
            recent_results = [int(x) for x in str(horse['recent_form']) if x.isdigit()]
            if recent_results:
                form_factor = self.calculate_form_cycle(recent_results)
                # Add trend analysis - better recent results weighted more
                trend = sum(1 for i in range(len(recent_results)-1) if recent_results[i] < recent_results[i+1])
                prob += (5 - form_factor) * 0.6 + (trend * 0.2)  # Increased form impact
        
        # Weight adjustment - increased impact
        if 'weight' in horse:
            weight_factor = 58 - float(horse['weight'])  # Standard weight is 58kg
            prob += weight_factor * 0.3  # Increased from 0.2
        
        # Days since last race (KGS) adjustment
        if 'kgs' in horse and horse['kgs']:
            optimal_kgs = 21  # Optimal days between races
            kgs_factor = 1 - abs(int(horse['kgs']) - optimal_kgs) / 60
            prob += kgs_factor * 1.5  # Reduced from 2
        
        # Performance metrics adjustment - increased S20 impact
        if 'hp' in horse and horse['hp']:
            prob += (int(horse['hp']) / 12)  # Reduced HP impact
        if 's20' in horse and horse['s20']:
            prob += (int(horse['s20']) / 8)  # Increased S20 impact
        
        # Add EİD consideration if available
        if 'eid' in horse and horse['eid'] and horse['eid'].strip():
            try:
                time_parts = horse['eid'].split('.')
                if len(time_parts) == 2:
                    minutes = int(time_parts[0])
                    seconds = float(time_parts[1])
                    total_seconds = minutes * 60 + seconds
                    # Better times get higher probability
                    eid_factor = (150 - total_seconds) * 0.1
                    prob += eid_factor
            except:
                pass  # Skip if EİD is not in correct format
                
        return prob

    def has_historical_data(self, horse_name):
        """Check if horse has historical data"""
        history = self.get_horse_history(horse_name)
        return len(history) > 0

    def predict_race(self, race_entries):
        """Predict race outcomes using Bayesian regression"""
        predictions = []
        horse_histories = []
        
        # Collect historical data for all horses
        for horse in race_entries:
            history = self.get_horse_history(horse['name'])
            horse_histories.append(history)
            
        # Update priors with historical data
        self.update_priors(horse_histories)
        
        # Get raw probabilities
        probabilities = {}
        for horse in race_entries:
            prob = self.calculate_win_probability(horse)
            probabilities[horse['name']] = prob
        
        # Normalize probabilities
        probabilities = self.normalize_probabilities(probabilities)
        
        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        print("\nBayesian Race Predictions:")
        for horse, prob in sorted_probs:
            horse_entry = next(h for h in race_entries if h['name'] == horse)
            class_factor = self.calculate_class_factor(horse_entry)
            track_factor = self.calculate_track_factor(horse_entry)
            confidence = "Form-based" if not self.has_historical_data(horse) else "Historical"
            print(f"{horse}: {prob:.2f}% (Class Factor: {class_factor:.2f}, Track Factor: {track_factor:.2f}, {confidence})")
        
        return probabilities

if __name__ == '__main__':
    race_data = [
        {
            'name': 'UMUDUNU KAYBETME',
            'weight': 62,
            'recent_form': '2 4 4 5 2 7',
            'kgs': 63,
            'hp': 39,
            's20': 16,
            'odds': 5.90,
            'start_position': 6,
            'age': 4,
            'sex': 'a',
            'sire': 'GÜNTAY',
            'dam': 'UMUT IŞIĞI'
        },
        {
            'name': 'BOĞUŞLU',
            'weight': 60,
            'recent_form': '4 7 4 7 2 2',
            'kgs': 18,
            'hp': 36,
            's20': 19,
            'odds': 6.00,
            'start_position': 7,
            'age': 4,
            'sex': 'k',
            'sire': 'OVA',
            'dam': 'IŞILCAN'
        },
        {
            'name': 'RÜZGARIMA KAPIL',
            'weight': 59,
            'recent_form': '2 2 5 3 4 4',
            'kgs': 21,
            'hp': 38,
            's20': 17,
            'odds': 1.05,
            'start_position': 4,
            'age': 4,
            'sex': 'a',
            'sire': 'SEMEND',
            'dam': 'ŞİLAN'
        },
        {
            'name': 'AKALTON',
            'weight': 53,
            'recent_form': '0 0 6 6 8 7',
            'kgs': 18,
            'hp': 14,
            's20': 6,
            'odds': 112.90,
            'start_position': 8,
            'age': 4,
            'sex': 'a',
            'sire': 'BİÇER',
            'dam': 'KENDİR'
        },
        {
            'name': 'KING PELE',
            'weight': 58,
            'recent_form': '7 7 7 5',
            'kgs': 18,
            'hp': 12,
            's20': 16,
            'odds': 22.55,
            'start_position': 1,
            'age': 4,
            'sex': 'k',
            'sire': 'TURBO',
            'dam': 'AÇIK ARA'
        },
        {
            'name': 'KIRMÜJDE',
            'weight': 58,
            'recent_form': '8 0',
            'kgs': 18,
            'hp': None,
            's20': 18,
            'odds': 84.70,
            'start_position': 2,
            'age': 4,
            'sex': 'k',
            'sire': 'MADRABAZ',
            'dam': 'MÜJDEER'
        },
        {
            'name': 'YAZASLANI',
            'weight': 58,
            'recent_form': '4',
            'kgs': 18,
            'hp': None,
            's20': 19,
            'odds': 42.35,
            'start_position': 3,
            'age': 4,
            'sex': 'a',
            'sire': 'VAKKAS',
            'dam': 'YAZ YAĞMURU'
        },
        {
            'name': 'SONTURBO',
            'weight': 53,
            'recent_form': '3 5 3 3 6 4',
            'kgs': 9,
            'hp': 29,
            's20': 16,
            'odds': 13.00,
            'start_position': 9,
            'age': 4,
            'sex': 'a',
            'sire': 'TURBO',
            'dam': 'NURSOLMAZ'
        },
        {
            'name': 'TURBOLU',
            'weight': 53,
            'recent_form': '7 8 7 5',
            'kgs': 7,
            'hp': 15,
            's20': 16,
            'odds': 112.90,
            'start_position': 5,
            'age': 4,
            'sex': 'k',
            'sire': 'TURBO',
            'dam': 'SERRACEM'
        }
    ]
    
    predictor = BayesianPredictor()
    predictor.predict_race(race_data) 
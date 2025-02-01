import pandas as pd
import numpy as np

class RacePredictor:
    def __init__(self):
        # Form designation weights
        self.form_weights = {
            'DB SKG SK': 10,
            'SKG SK': 8,
            'DB SK': 7,
            'SK': 6,
            'DB': 5,
            'K DB': 4,
            'K': 3,
            'KG': 2
        }
        
        # EİD (odds) weights
        self.eid_ranges = {
            (0, 3): 10,    # Very strong chance
            (3, 6): 8,     # Strong chance
            (6, 10): 6,    # Good chance
            (10, 15): 4,   # Fair chance
            (15, 20): 2,   # Outside chance
            (20, float('inf')): 1  # Long shot
        }
        
        # Breeding weights
        self.breeding_bonus = {
            'USA': 2,
            'IRE': 2,
            'GB': 1.5,
            'GER': 1.5,
            'FR': 1.5
        }
        
        # Weight penalty impact
        self.weight_penalty_factor = -2  # Per 0.1kg
        
    def calculate_form_score(self, form):
        """Calculate score based on form designations"""
        if not form or form == '-':
            return 0
        
        score = 0
        for designation, weight in self.form_weights.items():
            if designation in form:
                score += weight
        return score
    
    def calculate_eid_score(self, eid):
        """Calculate score based on EİD (odds)"""
        for (min_eid, max_eid), weight in self.eid_ranges.items():
            if min_eid <= eid < max_eid:
                return weight
        return 0
    
    def calculate_breeding_score(self, origin):
        """Calculate score based on breeding"""
        score = 0
        for country, bonus in self.breeding_bonus.items():
            if country in origin:
                score += bonus
        return score
    
    def calculate_weight_penalty_score(self, weight):
        """Calculate score impact of weight penalty"""
        try:
            base_weight = float(str(weight).split('+')[0])
            penalty = float(str(weight).split('+')[1]) if '+' in str(weight) else 0
            return self.weight_penalty_factor * (penalty / 0.1)
        except:
            return 0
    
    def calculate_barrier_score(self, barrier, field_size=None):
        """Calculate score based on barrier position"""
        if not field_size:
            field_size = 12  # Default field size
            
        # Less impact on barrier position based on our analysis
        if barrier <= 4:
            return 2  # Inside barriers
        elif barrier >= field_size - 2:
            return 0  # Wide barriers
        else:
            return 1  # Middle barriers
    
    def predict_race(self, horses_data):
        """
        Predict race outcome based on horse data
        
        horses_data: List of dictionaries containing horse information
        Each dict should have: name, form, eid, origin, weight, barrier
        """
        predictions = []
        
        for horse in horses_data:
            score = 0
            
            # Calculate individual scores
            form_score = self.calculate_form_score(horse.get('form', ''))
            eid_score = self.calculate_eid_score(float(horse.get('eid', 99)))
            breeding_score = self.calculate_breeding_score(horse.get('origin', ''))
            weight_score = self.calculate_weight_penalty_score(horse.get('weight', 0))
            barrier_score = self.calculate_barrier_score(int(horse.get('barrier', 0)))
            
            # Combined score with weightings
            score = (
                form_score * 2.0 +      # Form most important
                eid_score * 1.5 +       # EİD second most important
                breeding_score * 1.2 +   # Breeding third
                weight_score * 1.0 +     # Weight penalties
                barrier_score * 0.5      # Barrier least important
            )
            
            predictions.append({
                'name': horse['name'],
                'score': score,
                'form_score': form_score,
                'eid_score': eid_score,
                'breeding_score': breeding_score,
                'weight_score': weight_score,
                'barrier_score': barrier_score
            })
        
        # Sort by total score
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions

    def analyze_race(self, horses_data):
        """Provide detailed analysis of race"""
        predictions = self.predict_race(horses_data)
        
        print("\nRace Analysis:")
        print("-" * 50)
        
        # Top contenders
        print("\nTop Contenders:")
        for i, horse in enumerate(predictions[:3], 1):
            print(f"\n{i}. {horse['name']}")
            print(f"Total Score: {horse['score']:.2f}")
            print(f"Form Score: {horse['form_score']}")
            print(f"EİD Score: {horse['eid_score']}")
            print(f"Breeding Score: {horse['breeding_score']}")
            print(f"Weight Score: {horse['weight_score']}")
            print(f"Barrier Score: {horse['barrier_score']}")
        
        # Value bets
        print("\nValue Bets:")
        for horse in predictions:
            if horse['eid_score'] < horse['score'] / 2:  # High score relative to odds
                print(f"{horse['name']} - Score: {horse['score']:.2f}, EİD Score: {horse['eid_score']}")
        
        # Red flags
        print("\nRed Flags:")
        for horse in predictions:
            flags = []
            if horse['weight_score'] < -4:
                flags.append("High weight penalty")
            if horse['form_score'] == 0:
                flags.append("No form")
            if horse['eid_score'] <= 2:
                flags.append("High odds")
            
            if flags:
                print(f"{horse['name']}: {', '.join(flags)}")
        
        return predictions

# Example usage:
if __name__ == "__main__":
    predictor = RacePredictor()
    
    # Example data from our analyzed race
    example_horses = [
        {
            'name': 'HARDENST',
            'form': 'DB SKG SK',
            'eid': 2.80,
            'origin': 'BODEMEISTER (USA)',
            'weight': '58',
            'barrier': 8
        },
        {
            'name': 'JOSHIRO',
            'form': 'SK',
            'eid': 1.60,
            'origin': 'MASAR (IRE)',
            'weight': '58',
            'barrier': 9
        }
    ]
    
    predictor.analyze_race(example_horses) 
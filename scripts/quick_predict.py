import pandas as pd
import numpy as np

def analyze_race(csv_file):
    # Read race data
    df = pd.read_csv(csv_file)
    
    predictions = []
    for _, horse in df.iterrows():
        score = 0
        factors = []
        
        # 1. Recent Form Analysis
        recent_form = [int(x) for x in horse['recent_form'].strip('"').split(',') if x.isdigit()]
        if recent_form:
            # Weight recent performances more heavily
            weighted_positions = [pos * (1.2 ** i) for i, pos in enumerate(reversed(recent_form))]
            avg_position = np.mean(weighted_positions)
            form_score = (10 - avg_position) * 2
            score += form_score
            factors.append(f"Recent form score: {form_score:.1f}")
            
            # Trend analysis
            if len(recent_form) >= 3:
                recent_trend = recent_form[-3:]
                if recent_trend[0] > recent_trend[1] > recent_trend[2]:  # Improving trend
                    score += 3
                    factors.append("Improving trend")
        
        # 2. Weight Analysis
        weight = float(horse['weight'])
        weight_factor = (60 - weight) * 0.5  # Advantage for carrying less weight
        score += weight_factor
        factors.append(f"Weight factor: {weight_factor:.1f}")
        
        # 3. Starting Position
        start_pos = int(horse['start_pos'])
        if start_pos <= 3:
            pos_bonus = (4 - start_pos) * 1.5
            score += pos_bonus
            factors.append(f"Good starting position: +{pos_bonus:.1f}")
        
        # 4. Handicap Points
        hp = int(horse['handicap_points'])
        hp_factor = hp * 0.2
        score += hp_factor
        factors.append(f"Handicap points: {hp_factor:.1f}")
        
        # 5. Market Confidence (Ganyan odds)
        odds = float(horse['ganyan_odds'])
        market_factor = (35 / odds) if odds > 0 else 0
        score += market_factor
        factors.append(f"Market confidence: {market_factor:.1f}")
        
        predictions.append({
            'horse': horse['horse_name'],
            'score': score,
            'win_probability': 0,  # Will be normalized later
            'factors': factors
        })
    
    # Normalize scores to probabilities
    total_score = sum(p['score'] for p in predictions)
    for p in predictions:
        p['win_probability'] = (p['score'] / total_score) * 100
    
    # Sort by probability
    predictions.sort(key=lambda x: x['win_probability'], reverse=True)
    return predictions

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python quick_predict.py <race_data.csv>")
        return
        
    predictions = analyze_race(sys.argv[1])
    
    print("\nRace Prediction Analysis")
    print("=======================")
    for i, pred in enumerate(predictions, 1):
        print(f"\n{i}. {pred['horse']}")
        print(f"Win Probability: {pred['win_probability']:.1f}%")
        print("Key Factors:")
        for factor in pred['factors']:
            print(f"  - {factor}")

if __name__ == "__main__":
    main() 
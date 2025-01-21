import pandas as pd
import numpy as np
from predict_race import RacePredictor

def load_race_data(csv_file):
    df = pd.read_csv(csv_file)
    race_entries = []
    
    for _, row in df.iterrows():
        entry = {
            'name': row['horse_name'],
            'age': row['age'],
            'type': 'e',  # Default to stallion if not specified
            'sire': row['sire'],
            'dam': row['dam'],
            'weight': str(row['weight']),
            'jockey': row['jockey'],
            'trainer': row['trainer'],
            'recent_form': row['recent_form'].replace(',', ' '),
            'best_time': None  # We don't have this data
        }
        race_entries.append(entry)
    
    return race_entries

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict_current_race.py <race_data.csv>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    race_entries = load_race_data(csv_file)
    
    predictor = RacePredictor()
    predictions = predictor.predict_race(race_entries)
    
    print("\nRace Prediction Analysis")
    print("=======================")
    print("\nPredictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"\n{i}. {pred['horse_name']}")
        print(f"Win Chance: {pred['win_chance']:.1f}%")
        print("Key Factors:")
        for factor in pred['factors']:
            print(f"- {factor}")

if __name__ == "__main__":
    main() 
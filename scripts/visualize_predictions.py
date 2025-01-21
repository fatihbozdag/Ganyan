import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from predict_race import RacePredictor
from datetime import datetime

def create_simple_visualization(race_entries, race_info=None):
    # Default race info if not provided
    race_info = race_info or {
        'track': 'Adana',
        'distance': '1400m',
        'track_condition': 'Kum',
        'weather': 'Açık',
        'temperature': '18°C',
        'race_class': '3 Yaşlı',
        'prize': '35.000 TL'
    }
    
    predictor = RacePredictor()
    predictions = predictor.predict_race(race_entries)
    
    # Convert predictions to DataFrame with additional info
    pred_df = pd.DataFrame([
        {
            'Horse': p['horse_name'],
            'Win %': p['win_chance'],
            'Form': p['recent_form'],
            'Weight': race_entries[i]['weight'],
            'Jockey': race_entries[i]['jockey'],
            'Trainer': race_entries[i]['trainer'],
            'Type': 'Dişi' if race_entries[i]['type'] == 'd' else 'Erkek',
            'Best Time': race_entries[i]['best_time'] or 'N/A'
        } for i, p in enumerate(predictions)
    ])
    
    # Sort by win chance
    pred_df = pred_df.sort_values('Win %', ascending=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Plot 1: Win Probabilities
    bars = ax1.barh(pred_df['Horse'], pred_df['Win %'])
    
    # Add percentage labels on bars
    for bar in bars:
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%', 
                ha='left', va='center', fontsize=10)
    
    ax1.set_title('Win Probability by Horse')
    ax1.set_xlabel('Win Chance (%)')
    
    # Plot 2: Race Details Table
    race_details = [
        [f"Track: {race_info['track']}", f"Distance: {race_info['distance']}", 
         f"Surface: {race_info['track_condition']}", f"Weather: {race_info['weather']}"],
        [f"Class: {race_info['race_class']}", f"Prize: {race_info['prize']}", 
         f"Temp: {race_info['temperature']}", f"Date: {datetime.now().strftime('%d/%m/%Y')}"]
    ]
    
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=race_details, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs('data/visualizations', exist_ok=True)
    plt.savefig('data/visualizations/race_prediction.png', dpi=300, bbox_inches='tight')
    
    # Print detailed summary
    print("\nRace Analysis Summary")
    print("====================")
    print(f"\nRace Details:")
    print(f"Track: {race_info['track']} ({race_info['track_condition']})")
    print(f"Distance: {race_info['distance']}")
    print(f"Weather: {race_info['weather']} ({race_info['temperature']})")
    print(f"Class: {race_info['race_class']}")
    print(f"Prize: {race_info['prize']}")
    
    print("\nHorse Rankings:")
    for _, row in pred_df.iloc[::-1].iterrows():
        print(f"\n{row['Horse']}:")
        print(f"  Win Chance: {row['Win %']:.1f}%")
        print(f"  Recent Form: {row['Form']}")
        print(f"  Weight: {row['Weight']}kg")
        print(f"  Jockey: {row['Jockey']}")
        print(f"  Trainer: {row['Trainer']}")
        print(f"  Type: {row['Type']}")
        print(f"  Best Time: {row['Best Time']}")

def main():
    # Race information
    race_info = {
        'track': 'Adana',
        'distance': '1400m',
        'track_condition': 'Kum',
        'weather': 'Açık',
        'temperature': '18°C',
        'race_class': '3 Yaşlı',
        'prize': '35.000 TL'
    }
    
    # Define race entries
    race_entries = [
        {
            'name': 'GENÇ ERKUŞ',
            'age': 3,
            'type': 'e',
            'sire': 'DAI JIN (GB)',
            'dam': 'SUNNY FACE',
            'weight': '58',
            'jockey': 'HIŞMAN ÇİZİK',
            'trainer': 'SERKAN SÜSLÜ',
            'recent_form': '4 3 2 1 5',
            'best_time': '1.32.15'
        },
        {
            'name': 'LION ROBERO',
            'age': 3,
            'type': 'e',
            'sire': 'VICTORY GALLOP (CAN)',
            'dam': 'RAFIA',
            'weight': '58',
            'jockey': 'MÜSLÜM ÇELİK',
            'trainer': 'ARZU KILIÇ',
            'recent_form': '1 2 3 1 3 4',
            'best_time': None
        },
        {
            'name': 'EXOSOME',
            'age': 3,
            'type': 'd',
            'sire': "SIDNEY'S CANDY (USA)",
            'dam': 'DESIRES PEARL',
            'weight': '56',
            'jockey': 'NİZAMETTİN DEMİR',
            'trainer': 'SERVET CEYLAN',
            'recent_form': '3 6 7 4 4 5',
            'best_time': '1.30.71'
        },
        {
            'name': 'PRINCESS ADEN',
            'age': 3,
            'type': 'd',
            'sire': 'TRAPPE SHOT (USA)',
            'dam': 'TULUMBACI KIZI',
            'weight': '56',
            'jockey': 'MERT GÖKALP ARSLAN',
            'trainer': 'SAİT ALTUNTAŞ',
            'recent_form': '2 4 5 8 1 4',
            'best_time': '1.31.11'
        },
        {
            'name': 'LION ATLAS',
            'age': 3,
            'type': 'e',
            'sire': 'GOLDEN TOWER',
            'dam': 'CEVHERE',
            'weight': '56',
            'jockey': 'MEHMET SALİH ÇELİK',
            'trainer': 'EMİN KARATAŞ',
            'recent_form': '2 1 3 0 6 4',
            'best_time': '1.37.41'
        },
        {
            'name': 'CAZKIZ',
            'age': 3,
            'type': 'd',
            'sire': 'BLUEGRASS CAT (USA)',
            'dam': 'BLESSED',
            'weight': '54',
            'jockey': 'FATİH SULTAN MEHMET',
            'trainer': 'OSMAN ALTUNTAŞ',
            'recent_form': '1 6 9 3 6 7',
            'best_time': '1.30.49'
        },
        {
            'name': 'TOORMORE',
            'age': 3,
            'type': 'd',
            'sire': 'MENDIP (USA)',
            'dam': 'GOLDEN ATTITUDE',
            'weight': '53',
            'jockey': 'EMRAH SİNCAN',
            'trainer': 'SERKAN SÜSLÜ',
            'recent_form': '4 2 1 4 5 5',
            'best_time': '1.33.05'
        },
        {
            'name': 'CAYMAZ OĞLU',
            'age': 3,
            'type': 'e',
            'sire': 'GÜNGÖR BABA',
            'dam': 'SUN RAIN',
            'weight': '52',
            'jockey': 'MEHMET ALİ AKGÖZ',
            'trainer': 'MEHMET EMİN SARIDAĞ',
            'recent_form': '5 1 4 1 7 0',
            'best_time': None
        },
        {
            'name': 'KING GALLE',
            'age': 3,
            'type': 'e',
            'sire': 'NATIVE KHAN (FR)',
            'dam': 'LULU NANA',
            'weight': '52',
            'jockey': 'MEHMET KIYAK',
            'trainer': 'İSMAİL ÇEÇAN',
            'recent_form': '5 9 5 8 6 7',
            'best_time': '1.39.82'
        }
    ]
    
    create_simple_visualization(race_entries, race_info)
    print("\nVisualization saved as 'race_prediction.png'")

if __name__ == "__main__":
    main() 
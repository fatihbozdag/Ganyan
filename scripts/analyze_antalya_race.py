from predict_race import RacePredictor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def analyze_antalya_race():
    race_entries = [
        {
            'name': 'LITTLE JOE',
            'age': 3,
            'type': 'e',
            'sire': 'TOCCET (USA)',
            'dam': 'LUCKY NUMB',
            'weight': '61',
            'jockey': 'MEHMET KAYA',
            'trainer': 'OZAN BULUT',
            'recent_form': '1 0 1 3 8 3',
            'best_time': None
        },
        {
            'name': 'CANAY BEY',
            'age': 3,
            'type': 'e',
            'sire': 'KING OF THE SUN',
            'dam': 'POCO POCO',
            'weight': '55.5',
            'jockey': 'ENES BOZDAĞ',
            'trainer': 'DAVUT SAV',
            'recent_form': '7 4 3 2 4 1',
            'best_time': None
        },
        {
            'name': 'DROP OF TIME',
            'age': 3,
            'type': 'e',
            'sire': 'GRAYSTORM',
            'dam': 'LADY TUĞÇE',
            'weight': '58',
            'jockey': 'ERHAN AKTUĞ',
            'trainer': 'KENAN KORKMAZ',
            'recent_form': '7 7 5 3 4 1',
            'best_time': None
        },
        {
            'name': 'MEGA LOVE',
            'age': 3,
            'type': 'd',
            'sire': 'GOOD CURRY',
            'dam': 'SUDE SULTA',
            'weight': '58',
            'jockey': 'CANER TEPE',
            'trainer': 'ÖMER FARUK TAŞD',
            'recent_form': '1 1 7 2 3 8',
            'best_time': None
        },
        {
            'name': 'TIZONA',
            'age': 3,
            'type': 'e',
            'sire': 'TOCCET (USA)',
            'dam': 'INDYS SONA',
            'weight': '55',
            'jockey': 'İBRAHİM SEFA SÖ',
            'trainer': 'HALİL ERSİN UYS',
            'recent_form': '7 1 1 3 3 9',
            'best_time': None
        },
        {
            'name': 'BERN PRENSİ',
            'age': 3,
            'type': 'e',
            'sire': 'EUPRHATES (USA)',
            'dam': 'DREAM CITY',
            'weight': '55',
            'jockey': 'SERKAN YILDIZ',
            'trainer': 'HALİL ERSİN UYS',
            'recent_form': '8 8 8 8 7 0',
            'best_time': None
        },
        {
            'name': 'COMING RAIN',
            'age': 3,
            'type': 'd',
            'sire': 'GOOD CURRY',
            'dam': 'NO DELUSIO',
            'weight': '54',
            'jockey': 'HAKAN YILDIZ',
            'trainer': 'HAMZA KAYA',
            'recent_form': '6 2 1 5',
            'best_time': None
        },
        {
            'name': 'SUPER PLANE',
            'age': 3,
            'type': 'd',
            'sire': 'SUPER SAVER (USA)',
            'dam': 'SILVER PLA',
            'weight': '53',
            'jockey': 'SERCAN TIRPAN',
            'trainer': 'MURAT AYDOĞDİ',
            'recent_form': '8 4 3 2 4 0',
            'best_time': None
        }
    ]

    predictor = RacePredictor()
    predictions = predictor.predict_race(race_entries)

    print("\nAntalya Race Prediction Analysis")
    print("==============================")
    print("Race details: 3 year olds, Mixed gender")
    print("Track: Antalya")
    print("Distance: 1400m")
    print("\nPredictions:\n")

    # Sort predictions by win chance
    sorted_predictions = sorted(predictions, key=lambda x: x['win_chance'], reverse=True)

    for i, pred in enumerate(sorted_predictions, 1):
        print(f"{i}. {pred['horse_name']}")
        print(f"Win Chance: {pred['win_chance']:.1f}%")
        print("Key Factors:")
        for factor in pred['factors']:
            print(f"  - {factor}")
        print(f"Recent Form: {pred['recent_form']}\n")

if __name__ == "__main__":
    analyze_antalya_race() 
import json
from race_analyzer import RaceAnalyzer
from bayesian_predictor import BayesianPredictor

def load_test_race():
    """Load the race data for testing"""
    with open('current_race.json', 'r') as f:
        return json.load(f)

def run_simulation():
    """Run prediction simulation with new weights"""
    print("\n=== Running Race Prediction Simulation ===")
    
    # Load race data
    race_data = load_test_race()
    
    # Initialize analyzers with new weights
    analyzer = RaceAnalyzer()
    predictor = BayesianPredictor()
    
    print("\n1. ML Predictions with Updated Weights:")
    print("---------------------------------------")
    ml_predictions = analyzer.analyze_race(race_data['race_info'], race_data['entries'])
    
    # Sort and display ML predictions
    sorted_ml = sorted(ml_predictions, key=lambda x: x['total_score'], reverse=True)
    for pred in sorted_ml[:5]:  # Show top 5
        print(f"{pred['horse_name']}:")
        print(f"  Total Score: {pred['total_score']:.2f}")
        print(f"  Base Score: {pred['base_score']:.2f}")
        
    print("\n2. Bayesian Predictions with Updated Weights:")
    print("--------------------------------------------")
    bayesian_probs = predictor.predict_race(race_data['entries'])
    
    # Sort and display Bayesian predictions
    sorted_bayes = sorted(bayesian_probs.items(), key=lambda x: x[1], reverse=True)
    for horse, prob in sorted_bayes[:5]:  # Show top 5
        print(f"{horse}: {prob:.2f}%")
    
    print("\n3. Combined Analysis:")
    print("--------------------")
    # Combine both predictions for a final ranking
    combined_scores = {}
    
    for horse in race_data['entries']:
        name = horse['name']
        # Get ML score
        ml_score = next((pred['total_score'] for pred in ml_predictions if pred['horse_name'] == name), 0)
        # Get Bayesian probability
        bayes_prob = bayesian_probs.get(name, 0)
        
        # Normalize and combine scores (60% ML, 40% Bayesian)
        normalized_ml = ml_score / max(pred['total_score'] for pred in ml_predictions)
        normalized_bayes = bayes_prob / 100
        
        combined_scores[name] = (normalized_ml * 0.6) + (normalized_bayes * 0.4)
    
    # Sort and display combined predictions
    sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    for horse, score in sorted_combined[:5]:  # Show top 5
        print(f"{horse}: {score:.3f}")

if __name__ == '__main__':
    run_simulation() 
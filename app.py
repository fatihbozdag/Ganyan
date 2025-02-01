from flask import Flask, render_template, request, jsonify
import json
from race_analyzer import RaceAnalyzer
from bayesian_predictor import BayesianPredictor

app = Flask(__name__)
analyzer = RaceAnalyzer()
predictor = BayesianPredictor()

def load_race_data():
    try:
        with open('current_race.json', 'r') as f:
            return json.load(f)
    except:
        return {'race_info': {}, 'entries': []}

def save_race_data(data):
    with open('current_race.json', 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/')
def index():
    # Load current race data to display on page load
    race_data = load_race_data()
    return render_template('index.html', current_data=race_data)

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    try:
        race_data = load_race_data()
        if not race_data['entries']:
            return jsonify({
                'success': True,
                'ml_predictions': [],
                'bayesian_predictions': [],
                'combined_predictions': []
            })
            
        # Run predictions
        ml_predictions = analyzer.analyze_race(race_data['race_info'], race_data['entries'])
        bayesian_probs = predictor.predict_race(race_data['entries'])
        
        # Combine predictions
        combined_scores = {}
        for horse in race_data['entries']:
            name = horse['name']
            ml_score = next((pred['total_score'] for pred in ml_predictions if pred['horse_name'] == name), 0)
            bayes_prob = bayesian_probs.get(name, 0)
            
            normalized_ml = ml_score / max(pred['total_score'] for pred in ml_predictions) if ml_predictions else 0
            normalized_bayes = bayes_prob / 100
            
            combined_scores[name] = (normalized_ml * 0.6) + (normalized_bayes * 0.4)
        
        # Sort predictions
        sorted_ml = sorted(ml_predictions, key=lambda x: x['total_score'], reverse=True)[:5]
        sorted_bayes = sorted(bayesian_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return jsonify({
            'success': True,
            'ml_predictions': [{
                'name': pred['horse_name'],
                'total_score': round(pred['total_score'], 2),
                'base_score': round(pred['base_score'], 2)
            } for pred in sorted_ml],
            'bayesian_predictions': [{
                'name': name,
                'probability': round(prob, 2)
            } for name, prob in sorted_bayes],
            'combined_predictions': [{
                'name': name,
                'score': round(score, 3)
            } for name, score in sorted_combined]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/add_horse', methods=['POST'])
def add_horse():
    try:
        # Get form data
        horse_data = {
            'name': request.form['name'],
            'age': int(request.form['age']),
            'weight': float(request.form['weight']),
            'jockey': request.form['jockey'],
            'start_position': int(request.form['start_position']),
            'hp': int(request.form['hp']) if request.form['hp'] else None,
            'last_six': request.form['last_six'],
            'kgs': int(request.form['kgs']) if request.form['kgs'] else None,
            's20': int(request.form['s20']) if request.form['s20'] else None,
            'eid': request.form['eid'],
            'gny': float(request.form['gny']) if request.form['gny'] else None,
            'agf': float(request.form['agf']) if request.form['agf'] else None
        }

        # Load current race data
        race_data = load_race_data()
        
        # Add new horse
        race_data['entries'].append(horse_data)
        
        # Save updated data
        save_race_data(race_data)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear_race', methods=['POST'])
def clear_race():
    try:
        empty_race = {
            'race_info': {
                'track': '',
                'time': '',
                'distance': '',
                'surface': '',
                'race_type': '',
                'horse_type': '',
                'race_weight': '',
                'eid_record': ''
            },
            'entries': []
        }
        save_race_data(empty_race)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_race_data')
def get_race_data():
    try:
        race_data = load_race_data()
        return jsonify(race_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 
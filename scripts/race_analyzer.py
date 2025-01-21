import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json
from predict_race import RacePredictor
from collections import defaultdict

class RaceAnalyzer:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.history_file = 'data/history/prediction_history.json'
        self.load_prediction_history()
        
    def load_prediction_history(self):
        try:
            with open(self.history_file, 'r') as f:
                self.prediction_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.prediction_history = []
            
    def save_prediction_history(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.prediction_history, f)
            
    def get_horse_history(self, horse_name, last_n_races=10):
        """Get historical performance data for a horse"""
        horse_races = []
        for race in self.historical_data:
            for horse in race['horses']:
                if horse['horse_name'] == horse_name:
                    horse_races.append({
                        'date': race['race_info']['date'],
                        'track': race['race_info']['track'],
                        'finish_time': horse['finish_time'],
                        'weight': horse['weight'],
                        'jockey': horse['jockey']
                    })
        return sorted(horse_races, key=lambda x: x['date'], reverse=True)[:last_n_races]
        
    def get_jockey_stats(self, jockey_name, days=180):
        """Get performance statistics for a jockey"""
        jockey_races = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for race in self.historical_data:
            try:
                race_date = datetime.strptime(race['race_info']['date'], '%d/%m/%Y')
                if race_date >= cutoff_date:
                    for horse in race['horses']:
                        if horse['jockey'] == jockey_name:
                            finish_time = horse.get('finish_time')
                            if finish_time and ':' in finish_time:
                                time_parts = finish_time.split(':')
                                if len(time_parts) == 2:
                                    seconds = float(time_parts[1])
                                    jockey_races.append({
                                        'date': race['race_info']['date'],
                                        'track': race['race_info']['track'],
                                        'horse_name': horse['horse_name'],
                                        'finish_time': seconds,
                                        'position': 1 if seconds < 25 else 2  # Simplified position based on time
                                    })
            except (ValueError, KeyError):
                continue
        return jockey_races
        
    def analyze_horse(self, horse, race_info):
        """Analyze a horse's chances based on historical data"""
        score = 0
        factors = []
        
        # Get historical races for this horse
        horse_history = self.get_horse_history(horse['horse_name'])
        
        # Track surface performance
        surface_races = [h for h in horse_history if h.get('track_condition') == race_info['track_condition']]
        if surface_races:
            surface_wins = len([r for r in surface_races if r.get('position') == '1'])
            surface_places = len([r for r in surface_races if r.get('position') in ['1', '2', '3']])
            
            if surface_wins > 0:
                win_rate = surface_wins / len(surface_races)
                score += win_rate * 35
                factors.append(f"Surface win rate: {win_rate:.1%} ({surface_wins}/{len(surface_races)})")
            
            if surface_places > 0:
                place_rate = surface_places / len(surface_races)
                score += place_rate * 15
                factors.append(f"Surface place rate: {place_rate:.1%}")
            
            # Surface consistency
            if len(surface_races) >= 3:
                avg_position = sum(float(r.get('position', 0)) for r in surface_races[-3:]) / 3
                if avg_position <= 3:
                    score += 20
                    factors.append(f"Strong surface form (avg pos: {avg_position:.1f})")
        
        # Track specific performance
        track_races = [h for h in horse_history if h['track'] == race_info['track']]
        if track_races:
            track_wins = len([r for r in track_races if r.get('position') == '1'])
            track_places = len([r for r in track_races if r.get('position') in ['1', '2', '3']])
            
            if track_wins > 0:
                win_rate = track_wins / len(track_races)
                score += win_rate * 40
                factors.append(f"Track win rate: {win_rate:.1%} ({track_wins}/{len(track_races)})")
            
            if track_places > 0:
                place_rate = track_places / len(track_races)
                score += place_rate * 20
                factors.append(f"Track place rate: {place_rate:.1%}")
        
        # Distance performance
        distance_races = [h for h in horse_history if h.get('distance') == race_info['distance']]
        if distance_races:
            distance_wins = len([r for r in distance_races if r.get('position') == '1'])
            if distance_wins > 0:
                win_rate = distance_wins / len(distance_races)
                score += win_rate * 30
                factors.append(f"Distance win rate: {win_rate:.1%} ({distance_wins}/{len(distance_races)})")
        
        # Recent form analysis
        if 'recent_form' in horse:
            recent_positions = [int(pos) for pos in horse['recent_form'].split() if pos.isdigit()]
            if recent_positions:
                # Calculate weighted recent form (more recent races count more)
                weighted_positions = [pos * (1.2 ** i) for i, pos in enumerate(reversed(recent_positions))]
                avg_position = sum(weighted_positions) / len(weighted_positions)
                
                if avg_position <= 2:
                    score += 50
                    factors.append(f"Excellent recent form (avg pos: {avg_position:.1f})")
                elif avg_position <= 3:
                    score += 40
                    factors.append(f"Strong recent form (avg pos: {avg_position:.1f})")
                elif avg_position <= 4:
                    score += 30
                    factors.append(f"Good recent form (avg pos: {avg_position:.1f})")
                elif avg_position <= 5:
                    score += 20
                    factors.append(f"Consistent top-5 finisher (avg pos: {avg_position:.1f})")
                
                # Consistency analysis
                if len(recent_positions) >= 4:
                    positions_set = set(recent_positions[-4:])
                    if all(pos <= 3 for pos in positions_set):
                        score += 25
                        factors.append("Consistently finishing in top 3")
                    elif all(pos <= 5 for pos in positions_set):
                        score += 20
                        factors.append("Consistently finishing in top 5")
        
        # Best time analysis
        if 'best_time' in horse and horse['best_time']:
            try:
                time_parts = horse['best_time'].split('.')
                if len(time_parts) == 3:
                    minutes, seconds, hundredths = map(int, time_parts)
                    total_seconds = minutes * 60 + seconds + hundredths/100
                    if total_seconds <= 84:  # 1:24.00 for 1400m
                        score += 40
                        factors.append(f"Excellent best time ({horse['best_time']})")
                    elif total_seconds <= 85:  # 1:25.00
                        score += 25
                        factors.append(f"Good best time ({horse['best_time']})")
                    else:
                        score -= 10
                        factors.append(f"Below average time ({horse['best_time']})")
            except (ValueError, IndexError):
                pass
        
        # Handicap points analysis
        if 'handicap' in horse and horse['handicap'] != '0':
            try:
                handicap = float(horse['handicap'])
                if handicap >= 85:
                    score += 45
                    factors.append(f"Excellent handicap rating ({handicap})")
                elif handicap >= 75:
                    score += 35
                    factors.append(f"Strong handicap rating ({handicap})")
                elif handicap >= 65:
                    score += 25
                    factors.append(f"Good handicap rating ({handicap})")
                else:
                    score -= 5
                    factors.append(f"Below average handicap ({handicap})")
            except ValueError:
                pass
        else:
            score -= 15
            factors.append("No handicap rating")
        
        # Weight analysis
        try:
            weight = float(horse['weight'].split()[0])
            if weight <= 54:
                score += 30
                factors.append("Very favorable weight")
            elif weight <= 56:
                score += 20
                factors.append("Favorable weight")
            elif weight <= 58:
                score += 10
                factors.append("Acceptable weight")
            else:
                score -= 5
                factors.append("High weight")
        except (ValueError, IndexError):
            pass
        
        # Jockey performance at this track
        jockey_races = [h for h in self.historical_data if h.get('jockey') == horse['jockey'] and h['track'] == race_info['track']]
        if jockey_races:
            jockey_wins = len([r for r in jockey_races if r.get('position') == '1'])
            if jockey_wins > 0:
                win_rate = jockey_wins / len(jockey_races)
                score += win_rate * 25
                factors.append(f"Jockey track win rate: {win_rate:.1%} ({jockey_wins}/{len(jockey_races)})")
        
        # Ensure minimum score is 0
        score = max(0, score)
                
        return score, factors
        
    def analyze_race(self, race_entries, race_info):
        """Analyze all horses in a race and return predictions"""
        predictions = []
        
        # Analyze each horse
        total_score = 0
        for horse in race_entries:
            score, factors = self.analyze_horse(horse, race_info)
            total_score += score
            predictions.append({
                'horse_name': horse['horse_name'],
                'score': score,
                'factors': factors
            })
            
        # Convert scores to probabilities
        if total_score > 0:
            for pred in predictions:
                pred['win_probability'] = pred['score'] / total_score
        else:
            # Equal probabilities if no scores
            prob = 1.0 / len(predictions)
            for pred in predictions:
                pred['win_probability'] = prob
                
        # Sort by probability
        predictions.sort(key=lambda x: x['win_probability'], reverse=True)
        
        # Save prediction
        self.prediction_history.append({
            'date': race_info['date'],
            'track': race_info['track'],
            'predictions': predictions
        })
        self.save_prediction_history()
        
        return predictions

    def calculate_head_to_head(self, horse1, horse2):
        """Calculate head-to-head statistics between two horses"""
        h2h_stats = {
            'total_races': 0,
            'wins_horse1': 0,
            'wins_horse2': 0,
            'better_finish_horse1': 0
        }
        
        for pred in self.prediction_history:
            if horse1 in pred['predictions'] and horse2 in pred['predictions']:
                h2h_stats['total_races'] += 1
                pos1 = pred['predictions'].index(horse1)
                pos2 = pred['predictions'].index(horse2)
                
                if pos1 == 0: h2h_stats['wins_horse1'] += 1
                if pos2 == 0: h2h_stats['wins_horse2'] += 1
                if pos1 < pos2: h2h_stats['better_finish_horse1'] += 1
        
        return h2h_stats

    def get_track_performance(self, horse, track):
        """Get horse's performance statistics at specific track"""
        track_stats = {
            'races': 0,
            'wins': 0,
            'places': 0,
            'avg_position': 0,
            'best_time': None
        }
        
        positions = []
        for pred in self.prediction_history:
            if pred['track'] == track and horse in pred['predictions']:
                track_stats['races'] += 1
                pos = pred['predictions'].index(horse)
                positions.append(pos)
                
                if pos == 0: track_stats['wins'] += 1
                if pos <= 3: track_stats['places'] += 1
                
                if 'times' in pred and horse in pred['times']:
                    time = pred['times'][horse]
                    if not track_stats['best_time'] or time < track_stats['best_time']:
                        track_stats['best_time'] = time
        
        if positions:
            track_stats['avg_position'] = np.mean(positions)
            
        return track_stats

    def get_city_performance(self, horse, city):
        """Get horse's performance statistics in specific city"""
        city_stats = {
            'races': 0,
            'wins': 0,
            'places': 0,
            'earnings': 0,
            'best_tracks': defaultdict(int)
        }
        
        for pred in self.prediction_history:
            if pred['city'] == city and horse in pred['predictions']:
                city_stats['races'] += 1
                pos = pred['predictions'].index(horse)
                
                if pos == 0:
                    city_stats['wins'] += 1
                    city_stats['best_tracks'][pred['track']] += 1
                if pos <= 3:
                    city_stats['places'] += 1
                    
                # Calculate earnings if prize information is available
                if 'prize' in pred and pred['prize']:
                    prize = float(pred['prize'].replace('.', '').replace('TL', ''))
                    if pos == 0: city_stats['earnings'] += prize * 0.6
                    elif pos == 1: city_stats['earnings'] += prize * 0.2
                    elif pos == 2: city_stats['earnings'] += prize * 0.1
        
        return city_stats

    def get_seasonal_city_performance(self, horse, city, season):
        """Analyze horse's performance in a city by season"""
        seasonal_stats = {
            'races': 0,
            'wins': 0,
            'places': 0,
            'earnings': 0,
            'avg_position': []
        }
        
        for pred in self.prediction_history:
            if pred['city'] == city and horse in pred['predictions']:
                race_date = datetime.strptime(pred['date'], '%Y-%m-%d')
                race_season = (race_date.month % 12 + 3) // 3  # Convert month to season (1-4)
                
                if race_season == season:
                    seasonal_stats['races'] += 1
                    pos = pred['predictions'].index(horse)
                    seasonal_stats['avg_position'].append(pos)
                    
                    if pos == 0: seasonal_stats['wins'] += 1
                    if pos <= 3: seasonal_stats['places'] += 1
                    
                    if 'prize' in pred and pred['prize']:
                        prize = float(pred['prize'].replace('.', '').replace('TL', ''))
                        if pos == 0: seasonal_stats['earnings'] += prize * 0.6
                        elif pos == 1: seasonal_stats['earnings'] += prize * 0.2
                        elif pos == 2: seasonal_stats['earnings'] += prize * 0.1
        
        if seasonal_stats['avg_position']:
            seasonal_stats['avg_position'] = np.mean(seasonal_stats['avg_position'])
        
        return seasonal_stats

    def get_travel_impact(self, horse, from_city, to_city):
        """Analyze how the horse performs when traveling between cities"""
        impact_stats = {
            'races_after_travel': 0,
            'wins_after_travel': 0,
            'avg_position_after_travel': [],
            'recovery_time': []  # Days between races when traveling
        }
        
        sorted_predictions = sorted(self.prediction_history, key=lambda x: x['date'])
        
        for i in range(1, len(sorted_predictions)):
            prev_race = sorted_predictions[i-1]
            curr_race = sorted_predictions[i]
            
            if horse in prev_race['predictions'] and horse in curr_race['predictions']:
                if prev_race['city'] == from_city and curr_race['city'] == to_city:
                    impact_stats['races_after_travel'] += 1
                    pos = curr_race['predictions'].index(horse)
                    impact_stats['avg_position_after_travel'].append(pos)
                    
                    if pos == 0:
                        impact_stats['wins_after_travel'] += 1
                    
                    # Calculate days between races
                    prev_date = datetime.strptime(prev_race['date'], '%Y-%m-%d')
                    curr_date = datetime.strptime(curr_race['date'], '%Y-%m-%d')
                    recovery_days = (curr_date - prev_date).days
                    impact_stats['recovery_time'].append(recovery_days)
        
        if impact_stats['avg_position_after_travel']:
            impact_stats['avg_position_after_travel'] = np.mean(impact_stats['avg_position_after_travel'])
        if impact_stats['recovery_time']:
            impact_stats['avg_recovery_time'] = np.mean(impact_stats['recovery_time'])
        
        return impact_stats

    def get_last_race_city(self, horse):
        """Get the city of horse's last race"""
        sorted_predictions = sorted(self.prediction_history, 
                                  key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'),
                                  reverse=True)
        
        for pred in sorted_predictions:
            if horse in pred['predictions']:
                return pred['city']
        return None

    def get_weather_performance(self, horse, weather_condition):
        """Analyze horse's performance in specific weather conditions"""
        weather_stats = {
            'races': 0,
            'wins': 0,
            'places': 0,
            'avg_position': [],
            'by_city': defaultdict(lambda: {'races': 0, 'wins': 0})
        }
        
        for pred in self.prediction_history:
            if horse in pred['predictions'] and pred.get('weather') == weather_condition:
                weather_stats['races'] += 1
                pos = pred['predictions'].index(horse)
                weather_stats['avg_position'].append(pos)
                
                if pos == 0:
                    weather_stats['wins'] += 1
                    weather_stats['by_city'][pred['city']]['wins'] += 1
                if pos <= 3:
                    weather_stats['places'] += 1
                
                weather_stats['by_city'][pred['city']]['races'] += 1
        
        if weather_stats['avg_position']:
            weather_stats['avg_position'] = np.mean(weather_stats['avg_position'])
        
        return weather_stats

    def get_surface_preferences(self, horse, city=None):
        """Analyze horse's performance on different track surfaces"""
        surface_stats = {
            'kum': {'races': 0, 'wins': 0, 'places': 0, 'avg_position': []},
            'çim': {'races': 0, 'wins': 0, 'places': 0, 'avg_position': []},
            'sentetik': {'races': 0, 'wins': 0, 'places': 0, 'avg_position': []}
        }
        
        for pred in self.prediction_history:
            if horse in pred['predictions']:
                if city and pred['city'] != city:
                    continue
                    
                surface = pred.get('surface', '').lower()
                if surface in surface_stats:
                    stats = surface_stats[surface]
                    pos = pred['predictions'].index(horse)
                    
                    stats['races'] += 1
                    stats['avg_position'].append(pos)
                    if pos == 0: stats['wins'] += 1
                    if pos <= 3: stats['places'] += 1
        
        # Calculate averages
        for surface in surface_stats:
            if surface_stats[surface]['avg_position']:
                surface_stats[surface]['avg_position'] = np.mean(surface_stats[surface]['avg_position'])
                surface_stats[surface]['win_rate'] = surface_stats[surface]['wins'] / surface_stats[surface]['races'] \
                    if surface_stats[surface]['races'] > 0 else 0
        
        return surface_stats

    def get_distance_performance(self, horse, distance, venue=None):
        """Analyze horse's performance at specific distances"""
        distance_range = 100  # Consider races within ±100m
        distance_stats = {
            'races': 0,
            'wins': 0,
            'places': 0,
            'best_time': None,
            'avg_position': [],
            'by_surface': defaultdict(lambda: {'races': 0, 'wins': 0}),
            'progression': []  # Track improvement over time
        }
        
        sorted_races = sorted(self.prediction_history, 
                             key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
        
        for pred in sorted_races:
            if horse in pred['predictions']:
                race_distance = pred.get('distance')
                if not race_distance:
                    continue
                    
                # Convert distance to integer if it's a string
                if isinstance(race_distance, str):
                    race_distance = int(race_distance.replace('m', ''))
                
                # Check if race is within distance range
                if abs(race_distance - distance) <= distance_range:
                    if venue and pred['city'] != venue:
                        continue
                        
                    pos = pred['predictions'].index(horse)
                    distance_stats['races'] += 1
                    distance_stats['avg_position'].append(pos)
                    
                    if pos == 0:
                        distance_stats['wins'] += 1
                    if pos <= 3:
                        distance_stats['places'] += 1
                    
                    # Track surface performance
                    surface = pred.get('surface', '').lower()
                    distance_stats['by_surface'][surface]['races'] += 1
                    if pos == 0:
                        distance_stats['by_surface'][surface]['wins'] += 1
                    
                    # Track progression
                    if 'times' in pred and horse in pred['times']:
                        time = pred['times'][horse]
                        distance_stats['progression'].append({
                            'date': pred['date'],
                            'time': time,
                            'position': pos
                        })
                        
                        # Update best time
                        if not distance_stats['best_time'] or time < distance_stats['best_time']:
                            distance_stats['best_time'] = time
        
        if distance_stats['avg_position']:
            distance_stats['avg_position'] = np.mean(distance_stats['avg_position'])
        
        # Calculate improvement trend
        if len(distance_stats['progression']) > 1:
            times = [p['time'] for p in distance_stats['progression']]
            distance_stats['improving'] = times[-1] < times[0]  # True if getting faster
        
        return distance_stats

    def create_visualizations(self, pred_df, race_info):
        """Create comprehensive visualizations with all metrics"""
        # Create a larger figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 3, figure=fig)
        
        # 1. Win Probability Chart (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_win_probabilities(ax1, pred_df)
        
        # 2. Recent Form Heatmap (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_form_heatmap(ax2, pred_df)
        
        # 3. Surface Performance (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_surface_performance(ax3, pred_df, race_info)
        
        # 4. Distance Performance (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_distance_performance(ax4, pred_df, race_info)
        
        # 5. Weather Impact (Middle Middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self.plot_weather_performance(ax5, pred_df, race_info)
        
        # 6. Track History (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_track_history(ax6, pred_df, race_info)
        
        # 7. Race Details Table (Bottom)
        ax7 = fig.add_subplot(gs[2, :])
        self.create_race_details_table(ax7, race_info)
        
        plt.tight_layout()
        os.makedirs('data/visualizations', exist_ok=True)
        plt.savefig('data/visualizations/race_analysis.png', dpi=300, bbox_inches='tight')

    def create_h2h_matrix(self, pred_df):
        """Create head-to-head comparison matrix"""
        horses = pred_df['Horse'].tolist()
        h2h_matrix = pd.DataFrame(index=horses, columns=horses)
        
        for h1 in horses:
            for h2 in horses:
                if h1 != h2:
                    stats = self.calculate_head_to_head(h1, h2)
                    h2h_matrix.loc[h1, h2] = f"{stats['wins_horse1']}-{stats['wins_horse2']}"
        
        # Save matrix visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(h2h_matrix.notna(), cmap='YlOrRd')
        plt.xticks(range(len(horses)), horses, rotation=45, ha='right')
        plt.yticks(range(len(horses)), horses)
        
        for i in range(len(horses)):
            for j in range(len(horses)):
                text = h2h_matrix.iloc[i, j]
                if pd.notna(text):
                    plt.text(j, i, text, ha='center', va='center')
        
        plt.title('Head-to-Head Record')
        plt.tight_layout()
        plt.savefig('data/visualizations/h2h_matrix.png', dpi=300, bbox_inches='tight')

    def print_analysis(self, pred_df, race_info):
        """Print comprehensive analysis report"""
        print("\n=== Comprehensive Race Analysis ===")
        print("\nRace Details:")
        for key, value in race_info.items():
            print(f"{key}: {value}")
        
        print("\nHorse Analysis:")
        for _, row in pred_df.sort_values('Win %', ascending=False).iterrows():
            print(f"\n{row['Horse']}:")
            print(f"  Win Probability: {row['Win %']:.1f}%")
            print(f"  Recent Form: {row['Form']}")
            print(f"  Track Record at {race_info['Track']}:")
            print(f"    Races: {row['Track Stats']['races']}")
            print(f"    Wins: {row['Track Stats']['wins']}")
            print(f"    Places: {row['Track Stats']['places']}")
            if row['Track Stats']['best_time']:
                print(f"    Best Time: {row['Track Stats']['best_time']}")
            print(f"  Weight: {row['Weight']}kg")
            print(f"  Jockey: {row['Jockey']}")
            print(f"  Trainer: {row['Trainer']}")

    def plot_surface_performance(self, ax, pred_df, race_info):
        """Plot surface performance for each horse"""
        horses = pred_df['Horse'].tolist()
        surface_data = []
        
        for horse in horses:
            stats = self.get_surface_preferences(horse, race_info['City'])
            surface_data.append([
                stats[surface]['win_rate'] * 100 
                for surface in ['kum', 'çim', 'sentetik']
            ])
        
        im = ax.imshow(surface_data, aspect='auto', cmap='YlOrRd')
        ax.set_yticks(range(len(horses)))
        ax.set_yticklabels(horses)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Kum', 'Çim', 'Sentetik'])
        plt.colorbar(im, ax=ax, label='Win Rate %')
        ax.set_title('Surface Performance by Horse')

    def plot_distance_performance(self, ax, pred_df, race_info):
        """Plot distance performance trends"""
        distance = int(race_info['Distance'].replace('m', ''))
        horses = pred_df['Horse'].tolist()
        
        for horse in horses:
            stats = self.get_distance_performance(horse, distance, race_info['City'])
            if stats['progression']:
                times = [p['time'] for p in stats['progression']]
                dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in stats['progression']]
                ax.plot(dates, times, marker='o', label=horse)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Time')
        ax.set_title(f'Distance Performance Trends ({distance}m)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.get_xticklabels(), rotation=45)

    def plot_weather_performance(self, ax, pred_df, race_info):
        """Plot weather performance heatmap"""
        horses = pred_df['Horse'].tolist()
        weather_conditions = ['Açık', 'Yağmurlu', 'Bulutlu', 'Rüzgarlı']
        weather_data = []
        
        for horse in horses:
            horse_data = []
            for weather in weather_conditions:
                stats = self.get_weather_performance(horse, weather)
                win_rate = stats['wins'] / stats['races'] * 100 if stats['races'] > 0 else 0
                horse_data.append(win_rate)
            weather_data.append(horse_data)
        
        im = ax.imshow(weather_data, aspect='auto', cmap='YlOrRd')
        ax.set_yticks(range(len(horses)))
        ax.set_yticklabels(horses)
        ax.set_xticks(range(len(weather_conditions)))
        ax.set_xticklabels(weather_conditions, rotation=45)
        plt.colorbar(im, ax=ax, label='Win Rate %')
        ax.set_title('Weather Performance')

    def plot_track_history(self, ax, pred_df, race_info):
        """Plot track-specific performance history"""
        horses = pred_df['Horse'].tolist()
        track_stats = []
        
        for horse in horses:
            stats = self.get_track_performance(horse, race_info['Track'])
            track_stats.append([
                stats['wins'],
                stats['places'],
                stats['avg_position']
            ])
        
        x = np.arange(len(horses))
        width = 0.25
        
        ax.bar(x - width, [s[0] for s in track_stats], width, label='Wins')
        ax.bar(x, [s[1] for s in track_stats], width, label='Places')
        ax.bar(x + width, [s[2] for s in track_stats], width, label='Avg Pos')
        
        ax.set_ylabel('Count')
        ax.set_title(f'Track History at {race_info["Track"]}')
        ax.set_xticks(x)
        ax.set_xticklabels(horses, rotation=45)
        ax.legend()

    def create_race_details_table(self, ax, race_info):
        """Create detailed race information table"""
        ax.axis('off')
        table_data = [
            [f"Track: {race_info['Track']}", f"City: {race_info['City']}", 
             f"Distance: {race_info['Distance']}", f"Surface: {race_info['Surface']}"],
            [f"Weather: {race_info['Weather']}", f"Temperature: {race_info['Temperature']}", 
             f"Class: {race_info['Class']}", f"Prize: {race_info['Prize']}"]
        ]
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

    def plot_win_probabilities(self, ax, pred_df):
        """Plot win probability chart"""
        # Sort by win percentage
        sorted_df = pred_df.sort_values('Win %', ascending=True)
        
        # Create horizontal bar chart
        bars = ax.barh(sorted_df['Horse'], sorted_df['Win %'])
        
        # Add percentage labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', 
                    ha='left', va='center', fontsize=10)
        
        ax.set_title('Win Probability by Horse')
        ax.set_xlabel('Win Chance (%)')

    def plot_form_heatmap(self, ax, pred_df):
        """Plot recent form heatmap"""
        # Convert recent form strings to numeric arrays
        form_data = []
        for form in pred_df['Form']:
            # Convert form string to list of numbers, padding with zeros if needed
            numbers = [int(x) for x in form.split() if x.isdigit()]
            # Take last 6 races, pad with zeros if fewer than 6
            padded = (numbers[-6:] + [0] * 6)[:6]
            form_data.append(padded)
        
        # Create heatmap
        im = ax.imshow(form_data, aspect='auto', cmap='RdYlGn_r')
        
        # Add labels
        ax.set_yticks(range(len(pred_df)))
        ax.set_yticklabels(pred_df['Horse'])
        ax.set_xticks(range(6))
        ax.set_xticklabels(['Last', '2nd', '3rd', '4th', '5th', '6th'])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Finish Position')
        ax.set_title('Recent Form (Last 6 Races)')
        
        # Add text annotations
        for i in range(len(form_data)):
            for j in range(len(form_data[i])):
                text = form_data[i][j]
                if text != 0:  # Only show non-zero values
                    ax.text(j, i, str(text), ha='center', va='center')

def main():
    analyzer = RaceAnalyzer(...)  # Your historical data
    
    # Your existing race_info and race_entries
    race_info = {...}  # Your race information
    race_entries = [...] # Your race entries
    
    predictions = analyzer.analyze_race(race_entries, race_info)
    print("\nAnalysis complete. Check data/visualizations/ for detailed charts.")

if __name__ == "__main__":
    main() 
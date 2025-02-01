import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
from predict_race import RacePredictor
from collections import defaultdict
from src.utils.db_utils import DatabaseManager

class RaceAnalyzer:
    def __init__(self):
        self.history_file = 'data/history/prediction_history.json'
        self.db = None
        try:
            self.db = DatabaseManager()
            self.db.connect()
            self.load_history()
        except Exception as e:
            print(f"Error initializing RaceAnalyzer: {e}")
            if self.db:
                self.db.close()
            raise
        
    def load_history(self):
        """Load prediction history"""
        try:
            os.makedirs('data/history', exist_ok=True)
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    try:
                        self.history = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding history file: {e}")
                        self._initialize_empty_history()
            else:
                self._initialize_empty_history()
        except Exception as e:
            print(f"Error loading history: {e}")
            self._initialize_empty_history()
    
    def _initialize_empty_history(self):
        """Initialize empty history structure"""
        self.history = {
            'predictions': [],
            'track_stats': {},
            'horse_stats': {},
            'accuracy': {'correct': 0, 'total': 0}
        }

    def save_history(self):
        """Save prediction history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4)

    def calculate_head_to_head(self, horse1, horse2):
        """Calculate head-to-head statistics between two horses using database"""
        return self.db.get_head_to_head(horse1, horse2)

    def get_track_performance(self, horse, track):
        """Get horse's performance statistics at specific track using database"""
        return self.db.get_track_statistics(track)

    def get_city_performance(self, horse, city):
        """Get horse's performance statistics in specific city"""
        city_stats = {
            'races': 0,
            'wins': 0,
            'places': 0,
            'earnings': 0,
            'best_tracks': defaultdict(int)
        }
        
        for pred in self.history['predictions']:
            if pred['city'] == city and horse in pred['results']:
                city_stats['races'] += 1
                pos = pred['results'][horse]
                
                if pos == 1:
                    city_stats['wins'] += 1
                    city_stats['best_tracks'][pred['track']] += 1
                if pos <= 3:
                    city_stats['places'] += 1
                    
                # Calculate earnings if prize information is available
                if 'prize' in pred and pred['prize']:
                    prize = float(pred['prize'].replace('.', '').replace('TL', ''))
                    if pos == 1: city_stats['earnings'] += prize * 0.6
                    elif pos == 2: city_stats['earnings'] += prize * 0.2
                    elif pos == 3: city_stats['earnings'] += prize * 0.1
        
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
        
        for pred in self.history['predictions']:
            if pred['city'] == city and horse in pred['results']:
                race_date = datetime.strptime(pred['date'], '%Y-%m-%d')
                race_season = (race_date.month % 12 + 3) // 3  # Convert month to season (1-4)
                
                if race_season == season:
                    seasonal_stats['races'] += 1
                    pos = pred['results'][horse]
                    seasonal_stats['avg_position'].append(pos)
                    
                    if pos == 1: seasonal_stats['wins'] += 1
                    if pos <= 3: seasonal_stats['places'] += 1
                    
                    if 'prize' in pred and pred['prize']:
                        prize = float(pred['prize'].replace('.', '').replace('TL', ''))
                        if pos == 1: seasonal_stats['earnings'] += prize * 0.6
                        elif pos == 2: seasonal_stats['earnings'] += prize * 0.2
                        elif pos == 3: seasonal_stats['earnings'] += prize * 0.1
        
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
        
        sorted_predictions = sorted(self.history['predictions'], key=lambda x: x['date'])
        
        for i in range(1, len(sorted_predictions)):
            prev_race = sorted_predictions[i-1]
            curr_race = sorted_predictions[i]
            
            if horse in prev_race['results'] and horse in curr_race['results']:
                if prev_race['city'] == from_city and curr_race['city'] == to_city:
                    impact_stats['races_after_travel'] += 1
                    pos = curr_race['results'][horse]
                    impact_stats['avg_position_after_travel'].append(pos)
                    
                    if pos == 1:
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
        sorted_predictions = sorted(self.history['predictions'], 
                                  key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'),
                                  reverse=True)
        
        for pred in sorted_predictions:
            if horse in pred['results']:
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
        
        for pred in self.history['predictions']:
            if horse in pred['results'] and pred.get('weather') == weather_condition:
                weather_stats['races'] += 1
                pos = pred['results'][horse]
                weather_stats['avg_position'].append(pos)
                
                if pos == 1:
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
        
        for pred in self.history['predictions']:
            if horse in pred['results']:
                if city and pred['city'] != city:
                    continue
                    
                surface = pred.get('surface', '').lower()
                if surface in surface_stats:
                    stats = surface_stats[surface]
                    pos = pred['results'][horse]
                    
                    stats['races'] += 1
                    stats['avg_position'].append(pos)
                    if pos == 1: stats['wins'] += 1
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
        
        sorted_races = sorted(self.history['predictions'], 
                             key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
        
        for pred in sorted_races:
            if horse in pred['results']:
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
                        
                    pos = pred['results'][horse]
                    distance_stats['races'] += 1
                    distance_stats['avg_position'].append(pos)
                    
                    if pos == 1:
                        distance_stats['wins'] += 1
                    if pos <= 3:
                        distance_stats['places'] += 1
                    
                    # Track surface performance
                    surface = pred.get('surface', '').lower()
                    distance_stats['by_surface'][surface]['races'] += 1
                    if pos == 1:
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

    def analyze_race(self, race_entries, race_info):
        """Enhanced comprehensive race analysis with weather, surface, and distance factors"""
        if not race_entries or not race_info:
            raise ValueError("Race entries and race info must be provided")

        try:
            predictor = RacePredictor()
            predictions = predictor.predict_race(race_entries)
            
            if not predictions:
                raise ValueError("No predictions generated")
                
            current_season = (datetime.now().month % 12 + 3) // 3
        except Exception as e:
            raise ValueError(f"Error during race prediction: {str(e)}")
        
        # Add enhanced location-specific analysis
        for horse in predictions:
            horse_name = horse['horse_name']  # Changed from horse['name'] to horse['horse_name']
            
            # City performance
            city_stats = self.get_city_performance(horse_name, race_info['City'])
            if city_stats['races'] > 0:
                city_win_rate = city_stats['wins'] / city_stats['races']
                horse['win_chance'] *= (1 + city_win_rate * 0.2)
                horse['factors'].append(f"City Win Rate: {city_win_rate:.1%}")
                horse['factors'].append(f"City Earnings: {city_stats['earnings']:,.0f} TL")
            
            # Seasonal performance at this location
            seasonal_stats = self.get_seasonal_city_performance(horse_name, race_info['City'], current_season)
            if seasonal_stats['races'] > 0:
                seasonal_win_rate = seasonal_stats['wins'] / seasonal_stats['races']
                horse['win_chance'] *= (1 + seasonal_win_rate * 0.15)
                horse['factors'].append(f"Seasonal Win Rate: {seasonal_win_rate:.1%}")
                horse['factors'].append(f"Avg Seasonal Position: {seasonal_stats['avg_position']:.1f}")
            
            # Travel impact if horse's last race was in different city
            last_race_city = self.get_last_race_city(horse_name)
            if last_race_city and last_race_city != race_info['City']:
                travel_stats = self.get_travel_impact(horse_name, last_race_city, race_info['City'])
                if travel_stats['races_after_travel'] > 0:
                    travel_win_rate = travel_stats['wins_after_travel'] / travel_stats['races_after_travel']
                    adjustment = 0.9 if travel_win_rate < 0.1 else (1 + travel_win_rate * 0.1)
                    horse['win_chance'] *= adjustment
                    horse['factors'].append(f"Travel Impact: {travel_win_rate:.1%} win rate after travel")
                    if 'avg_recovery_time' in travel_stats:
                        horse['factors'].append(f"Avg Recovery: {travel_stats['avg_recovery_time']:.0f} days")
            
            # Weather impact
            weather_stats = self.get_weather_performance(horse_name, race_info['Weather'])
            if weather_stats['races'] > 0:
                weather_win_rate = weather_stats['wins'] / weather_stats['races']
                horse['win_chance'] *= (1 + weather_win_rate * 0.15)
                horse['factors'].append(f"Weather Performance: {weather_win_rate:.1%} win rate")
                
                # City-specific weather performance
                city_weather_stats = weather_stats['by_city'][race_info['City']]
                if city_weather_stats['races'] > 0:
                    city_weather_win_rate = city_weather_stats['wins'] / city_weather_stats['races']
                    horse['factors'].append(f"Local Weather Performance: {city_weather_win_rate:.1%}")
            
            # Surface preferences
            surface_stats = self.get_surface_preferences(horse_name, race_info['City'])
            current_surface = race_info['Surface'].lower()
            if current_surface in surface_stats:
                surface_perf = surface_stats[current_surface]
                if surface_perf['races'] > 0:
                    surface_win_rate = surface_perf['wins'] / surface_perf['races']
                    horse['win_chance'] *= (1 + surface_win_rate * 0.2)
                    horse['factors'].append(f"Surface Win Rate: {surface_win_rate:.1%}")
            
            # Distance analysis
            distance = int(race_info['Distance'].replace('m', ''))
            distance_stats = self.get_distance_performance(horse_name, distance, race_info['City'])
            if distance_stats['races'] > 0:
                distance_win_rate = distance_stats['wins'] / distance_stats['races']
                horse['win_chance'] *= (1 + distance_win_rate * 0.25)
                horse['factors'].append(f"Distance Win Rate: {distance_win_rate:.1%}")
                
                if distance_stats.get('improving'):
                    horse['win_chance'] *= 1.1
                    horse['factors'].append("Improving at this distance")
                
                if distance_stats['best_time']:
                    horse['factors'].append(f"Best Time: {distance_stats['best_time']}")
        
        # Normalize probabilities
        total_prob = sum(h['win_chance'] for h in predictions)
        for horse in predictions:
            horse['win_chance'] = (horse['win_chance'] / total_prob) * 100
        
        # Enhanced prediction dataframe
        pred_df = pd.DataFrame([
            {
                'Horse': p['horse_name'],
                'Win %': p['win_chance'],
                'Form': p['recent_form'],
                'Weight': race_entries[i]['weight'],
                'Jockey': race_entries[i]['jockey'],
                'Trainer': race_entries[i]['trainer'],
                'Type': 'Dişi' if race_entries[i]['type'] == 'd' else 'Erkek',
                'Best Time': race_entries[i]['best_time'] or 'N/A',
                'Track Stats': self.get_track_performance(p['horse_name'], race_info['Track'])
            } for i, p in enumerate(predictions)
        ])
        
        # Create visualizations
        self.create_visualizations(pred_df, race_info)
        
        # Generate head-to-head matrix
        self.create_h2h_matrix(pred_df)
        
        # Print comprehensive analysis
        self.print_analysis(pred_df, race_info)
        
        return predictions

    def create_visualizations(self, pred_df, race_info):
        """Create comprehensive visualizations with all metrics"""
        try:
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
        finally:
            plt.close(fig)  # Ensure figure is closed to prevent memory leaks

    def create_h2h_matrix(self, pred_df):
        """Create head-to-head comparison matrix"""
        try:
            horses = pred_df['Horse'].tolist()
            h2h_matrix = pd.DataFrame(index=horses, columns=horses)
            
            for h1 in horses:
                for h2 in horses:
                    if h1 != h2:
                        stats = self.calculate_head_to_head(h1, h2)
                        h2h_matrix.loc[h1, h2] = f"{stats['wins_horse1']}-{stats['wins_horse2']}"
            
            # Save matrix visualization
            fig = plt.figure(figsize=(10, 8))
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
            os.makedirs('data/visualizations', exist_ok=True)
            plt.savefig('data/visualizations/h2h_matrix.png', dpi=300, bbox_inches='tight')
        finally:
            plt.close('all')  # Ensure all figures are closed to prevent memory leaks

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
    analyzer = RaceAnalyzer()
    
    # Your existing race_info and race_entries
    race_info = {...}  # Your race information
    race_entries = [...] # Your race entries
    
    analyzer.analyze_race(race_entries, race_info)
    print("\nAnalysis complete. Check data/visualizations/ for detailed charts.")

if __name__ == "__main__":
    main()
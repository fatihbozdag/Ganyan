import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob
import numpy as np
from collections import defaultdict

class RaceAnalyzer:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.races_df = None
        self.results_df = None
        self.load_data()
        
        # Set up plotting style - using a valid style
        plt.style.use('seaborn-v0_8')  # or just remove this line
        sns.set_theme()  # This is enough for good-looking plots
    
    def load_data(self):
        """Load all CSV files from the data directory"""
        all_races = []
        all_results = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    date_str = file.split('-')[0]  # Extract date from filename
                    track = '-'.join(file.split('-')[1:-2])  # Extract track name
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            
                        current_race = None
                        for line in lines:
                            if not line.strip():
                                continue
                            
                            fields = [f.strip() for f in line.split(';')]
                            
                            # Race header
                            if 'Kosu' in line:
                                race_info = fields[0].split('.')
                                race_no = int(race_info[0])
                                race_time = fields[0].split(':')[-1].strip()
                                
                                current_race = {
                                    'date': date_str,
                                    'track': track,
                                    'race_no': race_no,
                                    'race_time': race_time,
                                    'distance': fields[4].strip() if len(fields) > 4 else None,
                                    'track_condition': fields[5].strip() if len(fields) > 5 else None
                                }
                                all_races.append(current_race)
                            
                            # Horse results
                            elif current_race and len(fields) >= 9 and fields[0].isdigit():
                                result = {
                                    'race_id': f"{date_str}_{track}_{current_race['race_no']}",
                                    'finish_position': int(fields[0]),
                                    'horse_name': fields[1],
                                    'age': fields[2],
                                    'origin_father': fields[3],
                                    'origin_mother': fields[4],
                                    'weight': self.parse_weight(fields[5]),
                                    'jockey': fields[6],
                                    'owner': fields[7],
                                    'trainer': fields[8],
                                    'finish_time': fields[12] if len(fields) > 12 else None,
                                    'odds': self.parse_odds(fields[13]) if len(fields) > 13 else None
                                }
                                all_results.append(result)
                    
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
        
        self.races_df = pd.DataFrame(all_races)
        self.results_df = pd.DataFrame(all_results)
        print(f"Loaded {len(self.races_df)} races and {len(self.results_df)} results")
    
    @staticmethod
    def parse_weight(weight_str):
        try:
            return float(weight_str.replace(',', '.'))
        except:
            return None
    
    @staticmethod
    def parse_odds(odds_str):
        try:
            return float(odds_str.replace(',', '.'))
        except:
            return None
    
    def analyze_tracks(self):
        """Analyze race distribution across tracks"""
        plt.figure(figsize=(12, 6))
        track_counts = self.races_df['track'].value_counts()
        track_counts.plot(kind='bar')
        plt.title('Number of Races by Track')
        plt.xlabel('Track')
        plt.ylabel('Number of Races')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/processed/races_by_track.png')
        return track_counts
    
    def analyze_jockeys(self):
        """Analyze jockey performance"""
        jockey_stats = self.results_df.groupby('jockey').agg({
            'race_id': 'count',
            'finish_position': lambda x: sum(x == 1)
        }).rename(columns={
            'race_id': 'rides',
            'finish_position': 'wins'
        })
        
        jockey_stats['win_rate'] = (jockey_stats['wins'] / jockey_stats['rides'] * 100).round(2)
        
        # Plot top jockeys
        plt.figure(figsize=(12, 6))
        top_jockeys = jockey_stats[jockey_stats['rides'] >= 5].sort_values('win_rate', ascending=False).head(10)
        top_jockeys['win_rate'].plot(kind='bar')
        plt.title('Top 10 Jockeys by Win Rate (min. 5 rides)')
        plt.xlabel('Jockey')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/processed/top_jockeys.png')
        return jockey_stats
    
    def analyze_odds(self):
        """Analyze betting odds and favorites"""
        valid_odds = self.results_df.dropna(subset=['odds'])
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=valid_odds, x='odds', bins=50)
        plt.title('Distribution of Betting Odds')
        plt.xlabel('Odds')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('data/processed/odds_distribution.png')
        
        # Analyze favorite performance
        favorites = valid_odds.loc[valid_odds.groupby('race_id')['odds'].idxmin()]
        favorite_win_rate = (favorites['finish_position'] == 1).mean() * 100
        
        return {
            'favorite_win_rate': favorite_win_rate,
            'avg_odds': valid_odds['odds'].mean(),
            'median_odds': valid_odds['odds'].median()
        }
    
    def analyze_track_details(self):
        """Detailed analysis of each track's performance"""
        track_stats = {}
        
        for track in self.races_df['track'].unique():
            track_races = self.races_df[self.races_df['track'] == track]
            track_results = self.results_df[self.results_df['race_id'].str.contains(track)]
            
            stats = {
                'total_races': len(track_races),
                'avg_horses_per_race': len(track_results) / len(track_races),
                'avg_distance': pd.to_numeric(track_races['distance'].str.replace('m', ''), errors='coerce').mean(),
                'common_condition': track_races['track_condition'].mode().iloc[0] if not track_races['track_condition'].empty else 'Unknown',
                'fastest_time': track_results['finish_time'].min(),
                'avg_odds': track_results['odds'].mean()
            }
            track_stats[track] = stats
        
        # Create a DataFrame for better visualization
        track_df = pd.DataFrame(track_stats).T
        
        # Plot average horses per race by track
        plt.figure(figsize=(12, 6))
        track_df['avg_horses_per_race'].plot(kind='bar')
        plt.title('Average Number of Horses per Race by Track')
        plt.xlabel('Track')
        plt.ylabel('Average Horses')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/processed/avg_horses_by_track.png')
        
        return track_df
    
    def analyze_horses(self):
        """Analyze horse performance and statistics"""
        horse_stats = self.results_df.groupby('horse_name').agg({
            'race_id': 'count',
            'finish_position': ['mean', 'min', lambda x: sum(x == 1)],
            'odds': 'mean',
            'weight': 'mean',
            'age': 'first',
            'origin_father': 'first',
            'origin_mother': 'first'
        })
        
        horse_stats.columns = ['races', 'avg_position', 'best_position', 'wins', 'avg_odds', 'avg_weight', 'age', 'father', 'mother']
        horse_stats['win_rate'] = (horse_stats['wins'] / horse_stats['races'] * 100).round(2)
        
        # Plot top horses by win rate (minimum 2 races)
        plt.figure(figsize=(12, 6))
        top_horses = horse_stats[horse_stats['races'] >= 2].sort_values('win_rate', ascending=False).head(10)
        top_horses['win_rate'].plot(kind='bar')
        plt.title('Top 10 Horses by Win Rate (min. 2 races)')
        plt.xlabel('Horse')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/processed/top_horses.png')
        
        # Analyze pedigree
        top_sires = self.results_df.groupby('origin_father').size().sort_values(ascending=False).head(10)
        plt.figure(figsize=(12, 6))
        top_sires.plot(kind='bar')
        plt.title('Top 10 Sires by Number of Offspring')
        plt.xlabel('Sire')
        plt.ylabel('Number of Offspring')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/processed/top_sires.png')
        
        return horse_stats
    
    def analyze_age_performance(self):
        """Analyze performance by age"""
        age_stats = self.results_df.groupby('age').agg({
            'race_id': 'count',
            'finish_position': ['mean', lambda x: sum(x == 1)],
        })
        
        age_stats.columns = ['races', 'avg_position', 'wins']
        age_stats['win_rate'] = (age_stats['wins'] / age_stats['races'] * 100).round(2)
        
        plt.figure(figsize=(10, 6))
        age_stats['win_rate'].plot(kind='bar')
        plt.title('Win Rate by Age')
        plt.xlabel('Age')
        plt.ylabel('Win Rate (%)')
        plt.tight_layout()
        plt.savefig('data/processed/win_rate_by_age.png')
        
        return age_stats
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        # Create processed directory
        os.makedirs('data/processed', exist_ok=True)
        
        print("\n=== Race Analysis Report ===")
        
        print("\nTrack Statistics:")
        track_stats = self.analyze_tracks()
        print(track_stats)
        
        print("\nJockey Statistics (Top 10):")
        jockey_stats = self.analyze_jockeys()
        print(jockey_stats.sort_values('win_rate', ascending=False).head(10))
        
        print("\nOdds Analysis:")
        odds_stats = self.analyze_odds()
        print(f"Favorite Win Rate: {odds_stats['favorite_win_rate']:.2f}%")
        print(f"Average Odds: {odds_stats['avg_odds']:.2f}")
        print(f"Median Odds: {odds_stats['median_odds']:.2f}")

def main():
    analyzer = RaceAnalyzer()
    analyzer.generate_report()

if __name__ == "__main__":
    main() 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import numpy as np
import zipfile

class RaceDataAnalyzer:
    def __init__(self, data_dir='data/raw', output_dir='data/processed'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.races_df = None
        self.results_df = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_data()
        
    def process_csv_content(self, content, filename):
        """Process CSV content and return races and results"""
        races = []
        results = []
        
        date_str = filename.split('-')[0]  # Extract date from filename
        track = filename.split('-')[1].replace('.csv', '')  # Extract track from filename
        
        lines = content.split('\n')
        current_race = None
        header_found = False
        
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
                races.append(current_race)
            
            # Process horse results
            elif current_race and 'At No' not in line and 'İkramiye' not in line:
                if len(fields) >= 9 and fields[0].isdigit():
                    result = {
                        'date': date_str,
                        'track': track,
                        'race_no': current_race['race_no'],
                        'finish_position': int(fields[0]),
                        'horse_name': fields[1],
                        'age': fields[2],
                        'origin_father': fields[3],
                        'origin_mother': fields[4],
                        'weight': fields[5],
                        'jockey': fields[6],
                        'owner': fields[7],
                        'trainer': fields[8],
                        'finish_time': fields[12] if len(fields) > 12 else None,
                        'odds': fields[13] if len(fields) > 13 else None
                    }
                    results.append(result)
        
        return races, results

    def load_data(self):
        """Load all CSV files and combine them into DataFrames"""
        all_races = []
        all_results = []
        
        # Look for consolidated zip files
        consolidated_dir = 'data/consolidated'
        if os.path.exists(consolidated_dir):
            for zip_file in os.listdir(consolidated_dir):
                if zip_file.endswith('.zip'):
                    print(f"\nProcessing {zip_file}...")
                    with zipfile.ZipFile(os.path.join(consolidated_dir, zip_file), 'r') as zf:
                        for filename in zf.namelist():
                            if filename.endswith('.csv'):
                                try:
                                    with zf.open(filename) as f:
                                        content = f.read().decode('utf-8')
                                        races, results = self.process_csv_content(content, filename)
                                        all_races.extend(races)
                                        all_results.extend(results)
                                        print(f"Added {len(races)} races and {len(results)} results from {filename}")
                                except Exception as e:
                                    print(f"Error processing {filename}: {str(e)}")
        
        # Also check raw directory for any unconsolidated files
        if os.path.exists(self.data_dir):
            for year_dir in os.listdir(self.data_dir):
                year_path = os.path.join(self.data_dir, year_dir)
                if os.path.isdir(year_path):
                    for filename in os.listdir(year_path):
                        if filename.endswith('.csv'):
                            try:
                                with open(os.path.join(year_path, filename), 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    races, results = self.process_csv_content(content, filename)
                                    all_races.extend(races)
                                    all_results.extend(results)
                                    print(f"Added {len(races)} races and {len(results)} results from {filename}")
                            except Exception as e:
                                print(f"Error processing {filename}: {str(e)}")
        
        self.races_df = pd.DataFrame(all_races)
        self.results_df = pd.DataFrame(all_results)
        
        print(f"\nTotal loaded: {len(self.races_df)} races and {len(self.results_df)} results")
    
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
        plt.savefig(os.path.join(self.output_dir, 'races_by_track.png'))
        plt.close()
        
        return track_counts
    
    def analyze_jockeys(self):
        """Analyze jockey performance"""
        # Get win percentages for jockeys with at least 5 rides
        jockey_stats = self.results_df.groupby('jockey').agg({
            'finish_position': ['count', lambda x: sum(x == 1)]
        }).fillna(0)
        
        jockey_stats.columns = ['rides', 'wins']
        jockey_stats['win_pct'] = (jockey_stats['wins'] / jockey_stats['rides'] * 100).round(2)
        jockey_stats = jockey_stats[jockey_stats['rides'] >= 5].sort_values('win_pct', ascending=False)
        
        plt.figure(figsize=(12, 6))
        jockey_stats['win_pct'].head(10).plot(kind='bar')
        plt.title('Top 10 Jockeys by Win Percentage (min. 5 rides)')
        plt.xlabel('Jockey')
        plt.ylabel('Win Percentage')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_jockeys.png'))
        plt.close()
        
        return jockey_stats
    
    def analyze_odds(self):
        """Analyze odds distribution and favorites performance"""
        try:
            # Clean and convert odds to numeric, removing any invalid values
            self.results_df['odds_numeric'] = pd.to_numeric(
                self.results_df['odds'].str.replace(',', '.'),  # Replace comma with dot
                errors='coerce'
            )
            
            # Remove rows where odds are NaN
            valid_odds_df = self.results_df.dropna(subset=['odds_numeric'])
            
            if len(valid_odds_df) == 0:
                print("No valid odds data found")
                return {
                    'favorite_win_pct': None,
                    'avg_odds': None,
                    'median_odds': None
                }
            
            plt.figure(figsize=(10, 6))
            sns.histplot(data=valid_odds_df, x='odds_numeric', bins=50)
            plt.title('Distribution of Odds')
            plt.xlabel('Odds')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'odds_distribution.png'))
            plt.close()
            
            # Analyze favorite performance
            # Group by track and race_no, then find the minimum odds for each race
            race_favorites = valid_odds_df.loc[
                valid_odds_df.groupby(['track', 'race_no'])['odds_numeric']
                .idxmin()
                .dropna()
            ]
            
            if len(race_favorites) > 0:
                favorite_wins = (race_favorites['finish_position'] == 1).mean() * 100
            else:
                favorite_wins = None
            
            return {
                'favorite_win_pct': favorite_wins,
                'avg_odds': valid_odds_df['odds_numeric'].mean(),
                'median_odds': valid_odds_df['odds_numeric'].median(),
                'total_races_with_odds': len(race_favorites),
                'total_valid_odds': len(valid_odds_df)
            }
            
        except Exception as e:
            print(f"Error in odds analysis: {str(e)}")
            return {
                'error': str(e),
                'favorite_win_pct': None,
                'avg_odds': None,
                'median_odds': None
            }
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n=== Race Analysis Report ===")
        
        print("\nTrack Statistics:")
        track_stats = self.analyze_tracks()
        print(track_stats)
        
        print("\nJockey Statistics (Top 10):")
        jockey_stats = self.analyze_jockeys()
        print(jockey_stats.head(10))
        
        print("\nOdds Analysis:")
        odds_stats = self.analyze_odds()
        if odds_stats['favorite_win_pct'] is not None:
            print(f"Favorite Win Percentage: {odds_stats['favorite_win_pct']:.2f}%")
        else:
            print("Favorite Win Percentage: N/A")
        print(f"Average Odds: {odds_stats['avg_odds']:.2f}")
        print(f"Median Odds: {odds_stats['median_odds']:.2f}")
        print(f"Total Races with Odds: {odds_stats['total_races_with_odds']}")
        print(f"Total Valid Odds: {odds_stats['total_valid_odds']}")

def main():
    analyzer = RaceDataAnalyzer()
    analyzer.generate_report()

if __name__ == "__main__":
    main() 
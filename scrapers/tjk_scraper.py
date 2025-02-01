import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
import sqlite3
import random
import json
import os
import logging
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TJKScraper:
    def __init__(self):
        """Initialize the TJK scraper"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.db_path = 'horse_racing.db'
        self.setup_database()
        
        # Update headers to match browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://www.tjk.org/TR/yarissever/Info/Page/GunlukYarisSonuclari',
        }
        
        # Configure session
        self.session.verify = False  # Disable SSL verification
        
    def setup_database(self):
        """Create SQLite database and tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create races table
        c.execute('''CREATE TABLE IF NOT EXISTS races
                    (race_id TEXT PRIMARY KEY,
                     date TEXT,
                     track TEXT,
                     race_no INTEGER,
                     distance INTEGER,
                     track_condition TEXT,
                     race_type TEXT,
                     prize REAL,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        # Create results table
        c.execute('''CREATE TABLE IF NOT EXISTS race_results
                    (result_id TEXT PRIMARY KEY,
                     race_id TEXT,
                     horse_name TEXT,
                     jockey TEXT,
                     trainer TEXT,
                     weight REAL,
                     handicap INTEGER,
                     position INTEGER,
                     finish_time TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY(race_id) REFERENCES races(race_id))''')
        
        conn.commit()
        conn.close()
        
    def get_available_dates(self):
        """Get list of available race dates"""
        # Implementation depends on TJK's API/website structure
        pass
    
    def get_csv_urls(self, date_str):
        """Get URLs for CSV files containing race data"""
        # Implementation depends on TJK's API/website structure
        pass
    
    def scrape_daily_races(self, date_str):
        """Scrape all races for a given date"""
        try:
            # Format date for the request
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%d.%m.%Y')
            
            # Get the race programs page
            url = "https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisSonuclari"
            response = self.session.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get the verification token
            token = soup.find('input', {'name': '__RequestVerificationToken'})['value']
            
            # Make POST request to get the data for the specific date
            post_data = {
                '__RequestVerificationToken': token,
                'QueryParameter.DateParameter': formatted_date,
                'QueryParameter.TrackId': ''
            }
            
            response = self.session.post(url, data=post_data, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all race result links
            race_links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if 'RaceResult' in href:
                    race_links.append(href)
            
            # Process each race
            for race_link in race_links:
                race_data = self.scrape_race_details(race_link)
                if race_data:
                    race_data['date'] = date_str
                    self.save_to_database(race_data)
                    
                    # Save to CSV in processed directory
                    track = race_data['track']
                    if track:
                        os.makedirs('data/processed', exist_ok=True)
                        csv_filename = f"data/processed/{formatted_date.replace('.', '-')}-{track}.csv"
                        
                        # Convert race data to DataFrame
                        results_df = pd.DataFrame(race_data['results'])
                        results_df['race_no'] = race_data['race_no']
                        results_df['distance'] = race_data['distance']
                        results_df['track_condition'] = race_data['track_condition']
                        
                        # Save to CSV
                        results_df.to_csv(csv_filename, index=False, encoding='utf-8')
                
                # Add delay between requests
                time.sleep(1)
                
            return True
            
        except Exception as e:
            logging.error(f"Error scraping races for {date_str}: {str(e)}")
            return False
    
    def scrape_date_range(self, start_date, end_date):
        """Scrape race data for a date range"""
        current_date = start_date
        while current_date <= end_date:
            try:
                self.scrape_daily_races(current_date.strftime('%Y-%m-%d'))
                logging.info(f"Scraped data for {current_date.date()}")
            except Exception as e:
                logging.error(f"Error scraping {current_date.date()}: {str(e)}")
            
            current_date += timedelta(days=1)
            # Add a small delay to avoid overwhelming the server
            time.sleep(1)
            
    def scrape_race_details(self, race_url):
        """Scrape details for a specific race"""
        try:
            full_url = f"https://medya-cdn.tjk.org/raporftp/TJKPDF{race_url}"
            response = self.session.get(full_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract race info
            race_info = {
                'race_id': race_url.split('/')[-1],
                'track': self.extract_text(soup, 'hipodrom'),
                'race_no': self.extract_text(soup, 'koşu_no'),
                'distance': self.extract_text(soup, 'mesafe'),
                'track_condition': self.extract_text(soup, 'pist_durumu'),
                'race_type': self.extract_text(soup, 'koşu_cinsi'),
                'prize': self.extract_text(soup, 'ikramiye')
            }
            
            # Extract results
            results_table = soup.find('table', {'class': 'sonuclar'})
            if results_table:
                results = []
                rows = results_table.find_all('tr')[1:]  # Skip header row
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 8:  # Ensure row has enough columns
                        result = {
                            'horse_name': cols[2].text.strip(),
                            'jockey': cols[3].text.strip(),
                            'trainer': cols[4].text.strip(),
                            'weight': self.parse_weight(cols[5].text),
                            'finish_position': self.parse_position(cols[0].text),
                            'finish_time': cols[6].text.strip(),
                            'odds': self.parse_odds(cols[7].text)
                        }
                        results.append(result)
                
                race_info['results'] = results
                
            return race_info
            
        except Exception as e:
            print(f"Error scraping race details for {race_url}: {str(e)}")
            return None
            
    def save_to_database(self, race_data):
        """Save scraped data to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            # Insert race info
            c.execute('''
                INSERT OR REPLACE INTO races
                (race_id, date, track, race_no, distance, track_condition, race_type, prize)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_data['race_id'],
                race_data.get('date'),
                race_data['track'],
                race_data['race_no'],
                race_data['distance'],
                race_data['track_condition'],
                race_data['race_type'],
                race_data['prize']
            ))
            
            # Insert results
            for result in race_data.get('results', []):
                result_id = f"{race_data['race_id']}_{result['horse_name']}"
                c.execute('''
                    INSERT OR REPLACE INTO race_results
                    (result_id, race_id, horse_name, jockey, trainer, weight,
                     position, finish_time, odds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_id,
                    race_data['race_id'],
                    result['horse_name'],
                    result['jockey'],
                    result['trainer'],
                    result['weight'],
                    result['finish_position'],
                    result['finish_time'],
                    result['odds']
                ))
            
            conn.commit()
            
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
            conn.rollback()
            
        finally:
            conn.close()
    
    @staticmethod
    def extract_text(soup, class_name):
        """Helper method to extract text from elements"""
        element = soup.find('div', {'class': class_name})
        return element.text.strip() if element else None
        
    @staticmethod
    def parse_weight(weight_str):
        """Parse weight string to float"""
        try:
            return float(weight_str.replace(',', '.').strip())
        except:
            return None
            
    @staticmethod
    def parse_position(pos_str):
        """Parse finish position to integer"""
        try:
            return int(pos_str.strip())
        except:
            return None
            
    @staticmethod
    def parse_odds(odds_str):
        """Parse odds string to float"""
        try:
            return float(odds_str.replace(',', '.').strip())
        except:
            return None
            
    def get_track_list(self):
        """Return a list of track names exactly as they appear in URLs"""
        return [
            'Bursa',
            'Şanlıurfa',
            'İstanbul',
            'İzmir',
            'Ankara',
            'Adana',
            'Elazığ',
            'Diyarbakır',
            'Kocaeli',
            'Antalya',
            'KemptonParkBirleşikKrallık',
            'SantaAnitaParkABD',
            'WolverhamptonBirleşikKrallık',
            'CagnesSurMerFransa'
        ]

    def generate_date_range(self, start_date, end_date):
        """Generate a list of dates between start and end date"""
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime('%d.%m.%Y'))
            current += timedelta(days=1)
        return dates

    def get_active_tracks_for_date(self, date_str):
        """Get list of active tracks for a specific date from the website"""
        try:
            # Format date for the request
            date_obj = datetime.strptime(date_str, '%d.%m.%Y')
            formatted_date = date_obj.strftime('%d.%m.%Y')
            
            # Make POST request to get the data
            url = "https://www.tjk.org/TR/YarisSever/Info/GetRacePrograms"
            data = {
                'QueryParameter.RaceDate': formatted_date,
                'QueryParameter.TrackId': '',
                'QueryParameter.ProgramType': '0'
            }
            
            response = self.session.post(url, data=data, headers=self.headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                track_buttons = soup.find_all('button', {'class': 'track-button'})
                
                active_tracks = []
                for button in track_buttons:
                    track_name = button.text.strip()
                    if '(' in track_name:  # Format: "Bursa (5. Y.G.)"
                        track_base = track_name.split('(')[0].strip()
                        active_tracks.append(track_base)
                
                return active_tracks
        except Exception as e:
            print(f"Error getting active tracks for {date_str}: {str(e)}")
        return []

    def get_active_tracks_and_links(self, date_str):
        """Get active tracks and their CSV links from the TJK website"""
        try:
            # Format date for the request
            formatted_date = datetime.strptime(date_str, '%d/%m/%Y').strftime('%d.%m.%Y')
            
            # Get the race programs page
            url = "https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisSonuclari"
            response = self.session.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get the verification token
            token = soup.find('input', {'name': '__RequestVerificationToken'})['value']
            
            # Make POST request to get the data for the specific date
            post_data = {
                '__RequestVerificationToken': token,
                'QueryParameter.DateParameter': formatted_date,
                'QueryParameter.TrackId': ''
            }
            
            response = self.session.post(url, data=post_data, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the CSV download buttons/links
            csv_links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if 'GunlukYarisSonuclari' in href and href.endswith('.csv'):
                    csv_links.append(href)
                    print(f"Found CSV link: {href}")
            
            return csv_links
            
        except Exception as e:
            print(f"Error getting active tracks for {date_str}: {str(e)}")
            return []

    def get_csv_download_url(self, date_str):
        """Get the CSV download URL for a specific date"""
        try:
            # Format date for the request
            formatted_date = datetime.strptime(date_str, '%d/%m/%Y').strftime('%d.%m.%Y')
            
            # The direct download URL
            download_url = f"https://www.tjk.org/TR/YarisSever/Info/GetRaceResultByDate"
            
            # Make POST request to get the CSV
            post_data = {
                'date': formatted_date,
                'trackId': '',
                'programType': '0'
            }
            
            # Add necessary headers for the request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
                'Referer': 'https://www.tjk.org/TR/yarissever/Info/Page/GunlukYarisSonuclari',
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            }
            
            response = self.session.post(download_url, data=post_data, headers=headers)
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response content: {response.text[:200]}")  # Print first 200 chars
            
            if response.status_code == 200:
                return response.content
            
        except Exception as e:
            print(f"Error getting CSV for {date_str}: {str(e)}")
        return None

    def brute_force_scrape(self, date_str):
        """Download race data using direct URLs"""
        try:
            # Convert DD/MM/YYYY to YYYY/YYYY-MM-DD
            date_parts = date_str.split('/')
            year = date_parts[2]
            formatted_date = f"{year}-{date_parts[1]}-{date_parts[0]}"
            dot_date = f"{date_parts[0]}.{date_parts[1]}.{date_parts[2]}"
            
            # Create year directory
            year_dir = f"data/raw/{year}"
            os.makedirs(year_dir, exist_ok=True)
            
            successful_urls = []
            
            # Try each track
            for track in self.get_track_list():
                # Construct the URL
                url = f"https://medya-cdn.tjk.org/raporftp/TJKPDF{formatted_date}/{dot_date}-{track}-GunlukYarisSonuclari-TR.csv"
                print(f"Trying URL: {url}")
                
                try:
                    response = requests.get(url, verify=False, timeout=10)
                    if response.status_code == 200:
                        try:
                            content = response.content.decode('utf-8')
                            if 'Kosu' in content and len(content) > 100:
                                print(f"Found valid data for {track}")
                                successful_urls.append(url)
                                
                                # Save the file
                                filename = os.path.join(year_dir, f"{dot_date}-{track}.csv")
                                with open(filename, 'w', encoding='utf-8') as f:
                                    f.write(content)
                                print(f"Saved: {filename}")
                        except UnicodeDecodeError:
                            print(f"Error decoding content for {track}")
                    else:
                        print(f"No data for {track} (Status: {response.status_code})")
                except Exception as e:
                    print(f"Error downloading {track}: {str(e)}")
                
                time.sleep(1)  # Polite delay
            
            return successful_urls
            
        except Exception as e:
            print(f"Error processing date {date_str}: {str(e)}")
            return []
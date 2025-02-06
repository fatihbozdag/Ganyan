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
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TJKScraper:
    def __init__(self):
        """Initialize the TJK scraper with Safari WebDriver"""
        self.db_path = 'horse_racing.db'
        self.setup_database()
        
        # Initialize Safari WebDriver
        self.driver = webdriver.Safari()
        self.driver.maximize_window()
        self.wait = WebDriverWait(self.driver, 10)  # Wait up to 10 seconds
        
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
        c.execute('''CREATE TABLE IF NOT EXISTS results
                    (result_id TEXT PRIMARY KEY,
                     race_id TEXT,
                     horse_name TEXT,
                     jockey TEXT,
                     trainer TEXT,
                     weight REAL,
                     handicap INTEGER,
                     position INTEGER,
                     finish_time TEXT,
                     odds REAL,
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
        """Scrape all races for a given date using Safari WebDriver"""
        try:
            # Format date for the request
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%d.%m.%Y')
            
            # Navigate to the results page
            url = f"https://www.tjk.org/TR/YarisSever/Info/Sehir/GunlukYarisSonuclari/{formatted_date}"
            self.driver.get(url)
            
            # Wait for the page to load
            try:
                self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'race-results')))
            except TimeoutException:
                logging.error("Timeout waiting for race results to load")
                return False
            
            # Find all race cards
            race_cards = self.driver.find_elements(By.CLASS_NAME, 'race-card')
            if not race_cards:
                logging.error("No race cards found")
                return False
            
            for race_card in race_cards:
                try:
                    # Extract race info
                    race_info = {
                        'race_id': race_card.get_attribute('data-race-id'),
                        'date': date_str,
                        'track': race_card.find_element(By.CLASS_NAME, 'track-name').text.strip(),
                        'race_no': int(race_card.find_element(By.CLASS_NAME, 'race-number').text.strip()),
                        'distance': int(race_card.find_element(By.CLASS_NAME, 'distance').text.replace('m', '').strip()),
                        'track_condition': race_card.find_element(By.CLASS_NAME, 'track-condition').text.strip(),
                        'race_type': race_card.find_element(By.CLASS_NAME, 'race-type').text.strip(),
                        'prize': float(race_card.find_element(By.CLASS_NAME, 'prize').text.replace('TL', '').replace('.', '').strip())
                    }
                    
                    # Extract results
                    results_table = race_card.find_element(By.CLASS_NAME, 'results-table')
                    rows = results_table.find_elements(By.TAG_NAME, 'tr')[1:]  # Skip header row
                    
                    results = []
                    for row in rows:
                        cols = row.find_elements(By.TAG_NAME, 'td')
                        if len(cols) >= 8:
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
                    
                    # Save to database
                    self.save_to_database(race_info)
                    
                    # Save to CSV
                    track_name = race_info['track']
                    if track_name:
                        os.makedirs('data/processed', exist_ok=True)
                        csv_filename = f"data/processed/{formatted_date.replace('.', '-')}-{track_name}.csv"
                        
                        # Convert race data to DataFrame
                        results_df = pd.DataFrame(results)
                        results_df['race_no'] = race_info['race_no']
                        results_df['distance'] = race_info['distance']
                        results_df['track_condition'] = race_info['track_condition']
                        
                        # Save to CSV
                        results_df.to_csv(csv_filename, index=False, encoding='utf-8')
                    
                except Exception as e:
                    logging.error(f"Error processing race card: {str(e)}")
                    continue
            
            return True
            
        except Exception as e:
            logging.error(f"Error scraping races for {date_str}: {str(e)}")
            return False
            
        finally:
            # Add a delay before the next request
            time.sleep(random.uniform(2, 5))
    
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
            full_url = f"https://www.tjk.org{race_url}"
            response = requests.get(full_url, headers=self.headers)
            if response.status_code != 200:
                logging.error(f"Failed to get race details: {response.status_code}")
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract race info
            race_info = {
                'race_id': race_url.split('/')[-2],  # Get the race ID from URL
                'track': self.extract_text(soup, 'hipodrom'),
                'race_no': self.extract_text(soup, 'kosu_no'),
                'distance': self.extract_text(soup, 'mesafe'),
                'track_condition': self.extract_text(soup, 'pist_durumu'),
                'race_type': self.extract_text(soup, 'kosu_cinsi'),
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
            logging.error(f"Error scraping race details for {race_url}: {str(e)}")
            return None
            
    def save_to_database(self, race_data):
        """Save race data to database"""
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
                race_data['date'],
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
                    INSERT OR REPLACE INTO results
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
            logging.error(f"Error saving to database: {str(e)}")
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
        """Parse weight value from string"""
        try:
            return float(weight_str.replace('kg', '').strip())
        except:
            return 0.0
            
    @staticmethod
    def parse_position(pos_str):
        """Parse finishing position from string"""
        try:
            return int(pos_str.strip())
        except:
            return 0
            
    @staticmethod
    def parse_odds(odds_str):
        """Parse odds value from string"""
        try:
            return float(odds_str.replace(',', '.').strip())
        except:
            return 0.0
            
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
        """Get list of active tracks for a given date"""
        try:
            url = "https://www.tjk.org/TR/YarisSever/Info/GetRacePrograms"
            
            # Add random delay
            time.sleep(random.uniform(1, 3))
            
            # Make POST request to get the data
            data = {
                'QueryParameter.RaceDate': date_str,
                'QueryParameter.TrackId': '',
                'QueryParameter.ProgramType': '0'
            }
            
            response = self.session.post(
                url,
                data=data,
                headers=self.headers
            )
            
            if response.status_code != 200:
                logging.error(f"Failed to get active tracks: {response.status_code}")
                return []
                
            soup = BeautifulSoup(response.content, 'html.parser')
            track_buttons = soup.find_all('button', {'class': 'track-button'})
            
            tracks = []
            for button in track_buttons:
                track_name = button.text.strip()
                track_id = button.get('data-track-id', '')
                if track_id and track_name:
                    tracks.append({
                        'id': track_id,
                        'name': track_name
                    })
            
            return tracks
                
        except Exception as e:
            logging.error(f"Error getting active tracks: {str(e)}")
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

    def __del__(self):
        """Clean up WebDriver when done"""
        if hasattr(self, 'driver'):
            self.driver.quit()
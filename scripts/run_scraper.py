import os
import sys
import logging
from datetime import datetime, timedelta
import sqlite3
from scrapers.tjk_scraper import TJKScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scraper.log')
    ]
)

def verify_scraping():
    """Verify if data was scraped by checking the database"""
    try:
        conn = sqlite3.connect('horse_racing.db')
        cursor = conn.cursor()
        
        # Check races count
        cursor.execute("SELECT COUNT(*) FROM races")
        races_count = cursor.fetchone()[0]
        
        # Check results count
        cursor.execute("SELECT COUNT(*) FROM results")
        results_count = cursor.fetchone()[0]
        
        # Get a sample race
        cursor.execute("""
            SELECT r.date, r.track, r.race_no, r.distance,
                   COUNT(res.result_id) as horses_count
            FROM races r
            LEFT JOIN results res ON r.race_id = res.race_id
            GROUP BY r.race_id
            ORDER BY r.date DESC
            LIMIT 1
        """)
        sample_race = cursor.fetchone()
        
        conn.close()
        return races_count, results_count, sample_race
        
    except sqlite3.Error as e:
        logging.error(f"Database error: {str(e)}")
        return 0, 0, None

def main():
    """Main function to run the scraper"""
    try:
        # Create scraper instance
        scraper = TJKScraper()
        
        # Get date range for last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        logging.info(f"Starting scraper for dates: {start_date.date()} to {end_date.date()}")
        
        # Scrape each date
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            logging.info(f"Scraping data for {date_str}")
            
            try:
                success = scraper.scrape_daily_races(date_str)
                if success:
                    logging.info(f"Successfully scraped data for {date_str}")
                else:
                    logging.error(f"Failed to scrape data for {date_str}")
                
            except Exception as e:
                logging.error(f"Error scraping {date_str}: {str(e)}")
            
            current_date += timedelta(days=1)
        
        # Verify results
        races_count, results_count, sample_race = verify_scraping()
        
        if races_count > 0 and results_count > 0:
            logging.info(f"Scraping completed successfully!")
            logging.info(f"Total races scraped: {races_count}")
            logging.info(f"Total results scraped: {results_count}")
            if sample_race:
                logging.info("Sample race data:")
                logging.info(f"Date: {sample_race[0]}")
                logging.info(f"Track: {sample_race[1]}")
                logging.info(f"Race #: {sample_race[2]}")
                logging.info(f"Distance: {sample_race[3]}m")
                logging.info(f"Horses: {sample_race[4]}")
        else:
            logging.error("No data was scraped!")
            
    except Exception as e:
        logging.error(f"Scraper failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
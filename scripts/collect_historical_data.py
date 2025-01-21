from scrapers.tjk_scraper import TJKScraper
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def collect_historical_data(start_year=2010, end_year=2025):
    """Collect historical race data from TJK"""
    
    scraper = TJKScraper()
    
    # Calculate date range
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 1, 1)
    
    logging.info(f"Starting data collection from {start_date.date()} to {end_date.date()}")
    
    try:
        # Scrape data for the date range
        scraper.scrape_date_range(start_date, end_date)
        logging.info("Data collection completed successfully")
    except Exception as e:
        logging.error(f"Error during data collection: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    logging.info("Starting historical data collection process")
    success = collect_historical_data()
    if success:
        logging.info("Historical data collection completed")
    else:
        logging.error("Historical data collection failed") 
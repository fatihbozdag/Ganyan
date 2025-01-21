import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import logging
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from tjk_scraper.spiders.tjk_spider import TJKSpider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def verify_scraping():
    """Verify if data was actually scraped by checking the database"""
    conn = sqlite3.connect('horse_racing.db')
    cursor = conn.cursor()
    
    # Check races count
    cursor.execute("SELECT COUNT(*) FROM races")
    races_count = cursor.fetchone()[0]
    
    # Check results count
    cursor.execute("SELECT COUNT(*) FROM results")
    results_count = cursor.fetchone()[0]
    
    # Get sample race data if exists
    cursor.execute("SELECT date, track, race_no FROM races LIMIT 1")
    sample_race = cursor.fetchone()
    
    conn.close()
    
    return races_count, results_count, sample_race

def consolidate_csv_files(year_dir, output_dir):
    """Consolidate CSV files for a year into a single compressed file"""
    print(f"\nConsolidating files for year {Path(year_dir).name}...")
    
    # Create consolidated directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a zip file for the year
    zip_path = os.path.join(output_dir, f"races_{Path(year_dir).name}.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for csv_file in Path(year_dir).glob('*.csv'):
            # Add file to zip
            zipf.write(csv_file, csv_file.name)
    
    print(f"Created consolidated file: {zip_path}")
    
    # Remove original files after successful consolidation
    for csv_file in Path(year_dir).glob('*.csv'):
        csv_file.unlink()
    
    return zip_path

def main():
    """Run the TJK scraper using Scrapy for historical data collection"""
    # Create output directories
    os.makedirs('data/raw', exist_ok=True)
    
    # Configure Scrapy settings
    settings = get_project_settings()
    settings.update({
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'ROBOTSTXT_OBEY': False,
        'CONCURRENT_REQUESTS': 1,  # Be nice to the server
        'DOWNLOAD_DELAY': 1,  # 1 second delay between requests
        'COOKIES_ENABLED': False,
        'LOG_LEVEL': 'INFO',
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        'TELNETCONSOLE_ENABLED': False,
    })
    
    # Create crawler process
    process = CrawlerProcess(settings)
    
    # Start with a single year (2024) for testing
    year = 2024
    start_date = datetime(year, 1, 1)
    end_date = datetime(year + 1, 1, 1)
    
    logging.info(f"Starting data collection for year {year}")
    
    # Generate list of dates to scrape
    current_date = start_date
    while current_date < end_date:
        date_str = current_date.strftime("%d/%m/%Y")
        process.crawl(TJKSpider, start_date=date_str, output_dir='data/raw')
        current_date += timedelta(days=1)
    
    # Start the crawling process
    logging.info("Starting crawl process...")
    process.start()
    
    logging.info("Data collection completed!")

if __name__ == "__main__":
    main() 
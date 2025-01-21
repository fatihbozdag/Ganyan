from datetime import datetime, timedelta
from src.scrapers.tjk_scraper import TJKScraper

def main():
    scraper = TJKScraper()
    # Start with just a few days of data first
    start_date = datetime.now() - timedelta(days=7)  # Last 7 days
    end_date = datetime.now()
    
    print("Starting scraping process...")
    races = scraper.scrape_date_range(start_date, end_date)
    print(f"Scraped {len(races)} races successfully")

if __name__ == "__main__":
    main() 
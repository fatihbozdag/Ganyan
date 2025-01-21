import scrapy
from datetime import datetime
import os
import logging
import json
from urllib.parse import quote

class TJKSpider(scrapy.Spider):
    name = 'tjk_spider'
    allowed_domains = ['tjk.org', 'medya-cdn.tjk.org']
    
    # City IDs mapping
    CITY_IDS = {
        'İstanbul': 3,
        'Ankara': 1,
        'İzmir': 2,
        'Bursa': 4,
        'Adana': 5,
        'Antalya': 6,
        'Elazığ': 7,
        'Şanlıurfa': 8,
        'Diyarbakır': 9,
        'Kocaeli': 10
    }
    
    def __init__(self, start_date=None, output_dir=None, *args, **kwargs):
        super(TJKSpider, self).__init__(*args, **kwargs)
        self.start_date = start_date
        self.output_dir = output_dir
        self.tracks = list(self.CITY_IDS.keys())
    
    def start_requests(self):
        # Parse the date
        date_obj = datetime.strptime(self.start_date, '%d/%m/%Y')
        
        # Format date components
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        
        # Try each track
        for track in self.tracks:
            # Build CDN URL with the new structure
            url = (
                f"https://medya-cdn.tjk.org/raporftp/TJKPDF/{year}/{year}-{month}-{day}/CSV/GunlukYarisProgrami/"
                f"{day}.{month}.{year}-{track}-GunlukYarisProgrami-TR.csv"
            )
            
            self.logger.info(f"Requesting URL: {url}")
            
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                errback=self.handle_error,
                cb_kwargs={'track': track, 'date': self.start_date},
                dont_filter=True,
                meta={
                    'dont_redirect': True,
                    'handle_httpstatus_list': [302, 403, 404]
                },
                headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://www.tjk.org',
                    'Connection': 'keep-alive',
                }
            )
    
    def parse(self, response, track, date):
        if response.status == 200:
            try:
                content = response.body.decode('utf-8')
                if 'Kosu' in content and len(content) > 100:
                    self.logger.info(f"Found valid data for {track} on {date}")
                    
                    # Convert date from DD/MM/YYYY to components
                    date_parts = date.split('/')
                    year = date_parts[2]
                    
                    # Save the file
                    year_dir = os.path.join(self.output_dir, 'raw', year)
                    os.makedirs(year_dir, exist_ok=True)
                    
                    filename = os.path.join(year_dir, f"{date.replace('/', '.')}-{track}.csv")
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.logger.info(f"Saved: {filename}")
                    
                    yield {
                        'status': 'success',
                        'track': track,
                        'date': date,
                        'filename': filename
                    }
            except UnicodeDecodeError:
                self.logger.error(f"Error decoding content for {track} on {date}")
        else:
            self.logger.warning(f"No data for {track} on {date} (Status: {response.status})")
    
    def handle_error(self, failure):
        self.logger.error(f"Request failed: {failure.value}")
        yield {
            'status': 'error',
            'error': str(failure.value),
            'url': failure.request.url
        } 
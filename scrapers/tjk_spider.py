import scrapy
from datetime import datetime
import os
import logging
import json
from urllib.parse import quote
import time
from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.utils.response import response_status_message

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
        self.auth_token = None
        self.retry_times = 5
        self.retry_delay = 10  # seconds
        
    def start_requests(self):
        # First authenticate
        auth_url = "https://www.tjk.org/TR/YarisSever/Query/ConnectedPage/RaceArchive"
        yield scrapy.Request(
            url=auth_url,
            callback=self.after_auth,
            dont_filter=True,
            meta={'dont_redirect': True},
            headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
                'Connection': 'keep-alive',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )

    def after_auth(self, response):
        # Extract authentication token from cookies or response
        self.auth_token = response.headers.get('Set-Cookie', '').decode('utf-8')
        
        # Parse the date
        date_obj = datetime.strptime(self.start_date, '%d/%m/%Y')
        
        # Format date components
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        
        # Try each track with delay
        for track in self.tracks:
            city_id = self.CITY_IDS[track]
            url = (
                f"https://www.tjk.org/TR/YarisSever/Info/GetRaceResults/{day}.{month}.{year}/{track}"
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
                    'handle_httpstatus_list': [302, 403, 404, 503],
                    'max_retry_times': self.retry_times,
                    'retry_delay': self.retry_delay
                },
                headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
                    'Referer': 'https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisSonuclari',
                    'Connection': 'keep-alive',
                    'Cookie': self.auth_token,
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
            # Add delay between requests
            time.sleep(2)
    
    def parse(self, response, track, date):
        if response.status == 200:
            try:
                content = response.body.decode('utf-8')
                if len(content) > 100:  # Basic check for valid content
                    self.logger.info(f"Found race data for {track} on {date}")
                    
                    # Save the file
                    date_parts = date.split('/')
                    year = date_parts[2]
                    year_dir = os.path.join(self.output_dir, year)
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
                else:
                    self.logger.warning(f"No race data found for {track} on {date}")
            except Exception as e:
                self.logger.error(f"Error processing data for {track} on {date}: {str(e)}")
                yield {
                    'status': 'error',
                    'track': track,
                    'date': date,
                    'error': str(e)
                }
        elif response.status in [403, 503]:
            retry_times = response.meta.get('retry_times', 0)
            if retry_times < self.retry_times:
                self.logger.info(f"Retrying {track} on {date} (attempt {retry_times + 1})")
                time.sleep(self.retry_delay)
                yield response.request.replace(dont_filter=True)
            else:
                self.logger.error(f"Max retries reached for {track} on {date}")
        else:
            self.logger.warning(f"Unexpected status {response.status} for {track} on {date}")
    
    def handle_error(self, failure):
        self.logger.error(f"Request failed: {failure.value}")
        yield {
            'status': 'error',
            'error': str(failure.value),
            'url': failure.request.url
        } 
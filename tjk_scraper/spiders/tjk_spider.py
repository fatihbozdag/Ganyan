import scrapy
from datetime import datetime
import os
import logging

class TJKSpider(scrapy.Spider):
    name = 'tjk_spider'
    allowed_domains = ['tjk.org', 'medya-cdn.tjk.org']
    
    def __init__(self, start_date=None, output_dir=None, *args, **kwargs):
        super(TJKSpider, self).__init__(*args, **kwargs)
        self.start_date = start_date
        self.output_dir = output_dir
        self.tracks = [
            'Bursa', 'Şanlıurfa', 'İstanbul', 'İzmir', 'Ankara', 'Adana',
            'Elazığ', 'Diyarbakır', 'Kocaeli', 'Antalya'
        ]
    
    def start_requests(self):
        # Convert DD/MM/YYYY to YYYY-MM-DD for URL
        date_parts = self.start_date.split('/')
        year = date_parts[2]
        month = date_parts[1]
        day = date_parts[0]
        
        # Format dates for URL
        formatted_date = f"{year}-{month}-{day}"
        dot_date = f"{day}.{month}.{year}"
        
        # Try each track
        for track in self.tracks:
            url = f"https://www.tjk.org/TR/YarisSever/Info/GetRaceResults/{dot_date}/{track}"
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                cb_kwargs={'track': track, 'date': dot_date},
                headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
                    'Referer': 'https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisSonuclari',
                },
                dont_filter=True
            )
    
    def parse(self, response, track, date):
        try:
            content = response.body.decode('utf-8')
            if len(content) > 100:  # Basic check for valid content
                self.logger.info(f"Found race data for {track} on {date}")
                
                # Save the file
                year = date.split('.')[-1]
                year_dir = os.path.join(self.output_dir, year)
                os.makedirs(year_dir, exist_ok=True)
                
                filename = os.path.join(year_dir, f"{date}-{track}.csv")
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
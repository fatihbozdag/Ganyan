import logging

class TJKScraperPipeline:
    def process_item(self, item, spider):
        if item.get('status') == 'success':
            logging.info(f"Successfully processed {item['track']} data for {item['date']}")
        elif item.get('status') == 'error':
            logging.error(f"Error processing {item['track']} data for {item['date']}: {item.get('error', 'Unknown error')}")
        return item 
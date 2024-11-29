import scrapy
import pandas as pd
import json
import os
from scrapy.crawler import CrawlerProcess

class WeatherSpider(scrapy.Spider):
    """
    Web Scraping Assignment - Weather Data Extraction
    
    Objectives:
    - Demonstrate Scrapy web scraping techniques
    - Extract weather information from multiple cities
    """
    name = 'weather_spider'
    
    # Predefined list of cities for demonstration
    start_urls = [
        'https://openweathermap.org/city/1850147',  # Tokyo
        'https://openweathermap.org/city/2643743',  # London
        'https://openweathermap.org/city/5128638',  # New York
        'https://openweathermap.org/city/2988507',  # Paris
        'https://openweathermap.org/city/2172517'   # Melbourne
    ]
    
    def parse(self, response):
        """
        Primary data extraction method
        
        Scraping Techniques:
        - Use CSS selectors for data extraction
        - Handle potential missing data
        - Extract structured information
        """
        try:
            # City name extraction
            city_name = response.css('h2.weather-widget__city::text').get().strip()
            
            # Temperature extraction
            temp_raw = response.css('span.weather-widget__temperature::text').get()
            temperature = float(temp_raw.replace('°C', '')) if temp_raw else None
            
            # Weather description
            description = response.css('p.weather-widget__description::text').get().strip()
            
            # Additional weather details
            details = response.css('ul.weather-widget__details li::text').getall()
            
            # Parse additional details
            humidity = wind_speed = pressure = None
            for detail in details:
                if 'Humidity' in detail:
                    humidity = float(detail.split(':')[1].strip().replace('%', ''))
                elif 'Wind' in detail:
                    wind_speed = float(detail.split(':')[1].strip().split()[0])
                elif 'Pressure' in detail:
                    pressure = float(detail.split(':')[1].strip().split()[0])
            
            yield {
                'City': city_name,
                'Temperature_Celsius': temperature,
                'Description': description,
                'Humidity_Percentage': humidity,
                'Wind_Speed_KMH': wind_speed,
                'Pressure_hPa': pressure
            }
        
        except Exception as e:
            self.logger.error(f"Error parsing {response.url}: {e}")

def run_spider():
    """
    Execute spider and save results
    
    Data Storage Techniques:
    - Use Scrapy CrawlerProcess
    - Convert results to pandas DataFrame
    - Save as CSV
    """
    process = CrawlerProcess(settings={
        'USER_AGENT': 'Assignment 3 Web Scraping (Educational Purpose)',
        'FEED_FORMAT': 'json',
        'FEED_URI': 'weather_data.json'
    })
    
    process.crawl(WeatherSpider)
    process.start()
    
    # Convert JSON to CSV
    df = pd.read_json('weather_data.json')
    df.to_csv('hande_gursoy_weather_data.csv', index=False)
    
    # Display results
    print("\nWeather Data Extraction Complete:")
    print(df)
    
    # Clean up temporary files
    os.remove('weather_data.json')

if __name__ == '__main__':
    run_spider()

'''
data_fetch.py

Copyright (c) 2021 Jude Davis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import io
import os
import sys
import time
import requests
import selenium.webdriver

from PIL import Image
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# - Constants
N_IMGS = sys.argv[1]
CLASS_TAG = 'Q4LuWd'
URL = sys.argv[2]

def main():
    sc = ImageScraper(URL)

    sc.init()
    sc.parse()
    sc.extract_imgs()


class ImageScraper:

    '''
    Class:
        Scrapes images from the web and extracts them into files.
    '''

    def __init__(self, url, parser='html.parser'):
        self.tree = {}
        self.url = url
        self.soup = None
        self.parser = parser
        self.contents = None
        self.elements = None
        self.img_sources = set()

        self.driver = selenium.webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.logs = []
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
        }
    
    def init(self):
        self.driver.get(self.url)

        self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight - 100);')
        time.sleep(3)

        self.contents = self.driver.page_source
        self.elements = self.driver.find_elements(By.CLASS_NAME, CLASS_TAG)

        self.logs.append(LogCodes.INIT + 'Request submitted and content souped...')
    
    def parse(self):
        '''
        Function:
            Parse response and find tags to traverse.
            Identify images, download them, and save them.
        Args:
            None
        Returns:
            A soup object which can parse html.
        '''
        
        self.logs.append(LogCodes.Event + 'Initialized parsing')

        i = 0
        last_height = 0

        for element in self.elements:

            try:
                element.click()
            except:
                continue

            images = self.driver.find_elements(By.CLASS_NAME, 'n3VNCb')

            for image in images:
                src = image.get_attribute('src')

                if 'http' in src:
                    self.img_sources.add(src)
            if i == N_IMGS:
                return self.soup
            i += 1

            if i % 10 == 0:
                self.driver.execute_script(f'window.scrollTo(document.body.scrollHeight + {last_height}, document.body.scrollHeight + {last_height + 5});')
                time.sleep(1)
                self.elements = self.driver.find_elements(By.CLASS_NAME, CLASS_TAG)
            
            time.sleep(.1)

            last_height += 1
    
    def extract_imgs(self) -> list:
        '''
        Function:
            Loops through each source URL and extracts the image from the URL.
            Sends a get request to each URL in order to extract the images.
        Args:
            None
        Returns:
            A list of images converted to matrices
        '''

        n_imgs_dataset = len(os.listdir('data'))

        i = n_imgs_dataset + 1
        print(len(self.img_sources))

        for src in self.img_sources:
            try:
                img = requests.get(src).content
                img_file = Image.open(io.BytesIO(img))
                print(i)

                with open(f'data/face{i}.jpg', 'wb') as f:
                    img_file.save(f, 'JPEG')
            except:
                continue
            i += 1


class LogCodes():
    INIT = 'Init: '
    Event = 'Event: '


if __name__ == '__main__':
    main()
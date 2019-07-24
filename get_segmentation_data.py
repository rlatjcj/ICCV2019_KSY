import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
from functools import partial
import os
import os.path as path
import wget

def get_metadata(url, path='./'):
    # filename = url.split('/')[-1]
    # data = requests.get(url).data
    # with open(path.join(path, filename), 'wb') as f:
    #     f.write(data)
    wget.download(url, out=path)

if __name__=='__main__':
    target_url = 'https://storage.googleapis.com/openimages/web/download.html'
    page = requests.get(target_url).text
    
    soup = BeautifulSoup(page, 'lxml')
    main = soup.select_one('div.main')
    rows = main.select('div.row')
    rows = [row for row in rows 
                if row.select_one('div.col-10') and row.select_one('div.col-2.titlecol')]

    base_path = path.join('metadata', 'Segmentations')
    os.makedirs(base_path, exist_ok=True)
    base_url = 'https://storage.googleapis.com/openimages/v5/{}-masks/{}-masks-{}.zip'

    for sub_name in ['train', 'validation', 'test']:
        sub_path = path.join(base_path, sub_name)
        os.makedirs(sub_path, exist_ok=True)
        urls = [base_url.format(sub_name, sub_name, offset) 
                for offset in list(range(10))+['a','b','c','d','e','f']]
        download_func = partial(get_metadata, path=sub_path)
        with Pool(8) as pool:
            pool.map(download_func, urls)
    
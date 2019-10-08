import requests
from bs4 import BeautifulSoup

import os
# Python이 실행될 때 DJANGO_SETTINGS_MODULE이라는 환경 변수에 현재 프로젝트의 settings.py파일 경로를 등록합니다.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_tutorial.settings")
# 이제 장고를 가져와 장고 프로젝트를 사용할 수 있도록 환경을 만듭니다.
import django
django.setup()

from crawler.models import RepositData

def parse_reposit():
    BASE_URL = 'https://github.com'
    SUB_URL = '/joyoon729?tab=repositories'
    req = requests.get(BASE_URL + SUB_URL)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    uls = soup.select_one('div#user-repositories-list ul')
    reposits = uls.select('div h3 a')

    data = {}
    for reposit in reposits:
        data[reposit.text.strip()] = BASE_URL + reposit.get('href')
    return data

if __name__=='__main__':
    reposit_data_dict = parse_reposit()
    for t, l in reposit_data_dict.items():
        RepositData(title=t, link=l).save()
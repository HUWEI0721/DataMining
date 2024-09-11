import requests
import concurrent.futures 
from bs4 import BeautifulSoup
import json
from functools import wraps
import re

with open("news_source.json", "r", encoding="utf8") as f:
    news_raw = f.read()
    news_list = json.loads(news_raw)


total = len(news_list)
now = 0

def fetch_news_decorator(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        global now, total
        print(f"[{now}/{total}]", end=" ")
        now += 1
        return f(*args, **kwargs)
    return decorated 


@fetch_news_decorator
def fetch_one_news(news):
    title = f"{news['showTitle']}-{news['contentId'][-6:]}"
    print("Processing", title)

    title = re.sub(r'[\\/:*?"<>|\r\n]+', "", title)
    
    response = requests.get(f"http://www.news.cn{news['publishUrl']}")
    if response.status_code != 200:
        print(f"[WARNING] fetch failed: {title}")
        return "NOT OK"
    else:
        resp_html = response.text
        soup = BeautifulSoup(resp_html, "lxml")
        text_node = soup.find(id="detailContent")
        text = text_node.get_text()
        with open(f"result/{title}.txt", "w", encoding="utf8") as f:
            f.write(text)
        return "OK"


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(fetch_one_news, news_list))



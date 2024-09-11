from time import sleep
from hanlp_restful import HanLPClient
import os
import tqdm

HanLP = HanLPClient('https://www.hanlp.com/api', auth='NTAyN0BiYnMuaGFubHAuY29tOkw5clFLbGhTRTIyNUppUDE=',
                    language='zh')  # auth不填则匿名，zh中文，mul多语种
directory='./55DAMScriper/result'

filenames = os.listdir(directory)

news_num=100

texts=[]

print('开始读取新闻数据...')
for filename in tqdm.tqdm(filenames[:news_num]):
    filepath=os.path.join(directory, filename)

    with open(filepath,'r',encoding='utf-8') as f:
        news_text=f.read()

    texts.append(news_text[:4000])


print('开始提取摘要并保存...')
for news_text in tqdm.tqdm(texts):
    news_summary= [key for key, value in HanLP.extractive_summarization(news_text, topk=6).items() if value > 0.1]
    with open('./news_summary.txt','a',encoding='utf-8') as f:
        for i in news_summary:
            f.write(i+'\n')
    sleep(1.3)
from time import sleep

from hanlp_restful import HanLPClient
from openai import OpenAI
import tqdm
from database import NDB
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from response import CommonResponse
from contextlib import asynccontextmanager

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
HanLP = HanLPClient('https://www.hanlp.com/api', auth='NTAyN0BiYnMuaGFubHAuY29tOkw5clFLbGhTRTIyNUppUDE=',
                    language='zh')  # auth不填则匿名，zh中文，mul多语种
SYS_PROMPT = '''
你是一名出色的语言学家，你的研究领域是自然语言处理。接下来我将给你一个中文原句与语义角色提取后的句子，你需要仔细理解句子中的各种语法关系，根据原句，
判断语义角色提取后的句子是否*完整*、*正确*与*合理*，如果是请回复“T”，否则请回复“F”，请不要回复其他内容。
'''
NDB = NDB("bolt://localhost:7687", "neo4j", "dry330500")


async def close_db_connection():
    NDB.close()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def summarize(text):
    sents = [key for key, value in HanLP.extractive_summarization(text, topk=6).items() if value > 0.1]  # 参数要根据具体情况调整
    return sents
    

def get_triples(sent):
    triple_groups = []
    sleep(1)
    doc = HanLP.parse(sent, tasks=['srl'])

    for i in range(len(doc['tok/fine'])):
        ent1 = ""
        ent2 = ""
        relation = ""
        for j, pas in enumerate(doc['srl'][i]):
            for form, role, begin, end in pas:
                if role == 'ARG0':
                    ent1 = form
                if role == 'ARG1':
                    ent2 = form
                if role == 'PRED':
                    relation = form
            triple_groups.append((ent1.strip(), relation.strip(), ent2.strip()))

    return triple_groups


def filter_triples(sent, triples):
    results = []
    for i in triples:
        srl_text = ""
        if i[0] == "" or i[1] == "" or i[2] == "":
            continue
        # srl_text += i[0] + i[1] + i[2]
        # USER_INPUT = "原句：" + sent + "\n" + "语义角色提取后的句子：" + srl_text
        # completion = client.chat.completions.create(
        #     model="Qwen/Qwen1.5-7B-Chat-GGUF",
        #     messages=[
        #         {"role": "system", "content": SYS_PROMPT},
        #         {"role": "user", "content": USER_INPUT}
        #     ],
        #     temperature=0.7,  # 这里要结合具体模型调整
        # )
        # if completion.choices[0].message.content == "T":
        results.append(i)
    return results


def get_triple_groups(sents):
    final_result = []
    for sent in tqdm.tqdm(sents):
        triples = get_triples(sent)
        results = filter_triples(sent, triples)
        final_result.extend(results)
    return final_result


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_db_connection()


@app.post("/clear", summary="清空图数据库")
async def clear_all():
    NDB.clear_all()
    return CommonResponse.success("清空成功")


@app.post("/articles", summary="插入文章, 并提取三元组，存入图数据库")
async def add_articles(text: str):
    sents = summarize(text)
    result = get_triple_groups(sents)
    for i in result:
        NDB.add_triplet(i[0], i[1], i[2])
    return CommonResponse.success("插入完成")


@app.post("/triplets", summary="插入三元组")
async def add_triplets(subject: str, relation: str, _object: str):
    NDB.add_triplet(subject, relation, _object)
    return CommonResponse.success("插入完成")


@app.get("/triplets", summary="查询指定subject对应的三元组")
async def find_triplets(subject: str):
    results = NDB.find_triplets(subject)
    if results:
        return CommonResponse.success(results)
    else:
        return CommonResponse.error(404, "未找到相关数据")


@app.get("/subjects", summary="查询所有subject")
async def get_all_subjects():
    results = NDB.get_all_subjects()
    if results:
        return CommonResponse.success(results)
    else:
        return CommonResponse.error(404, "未找到相关数据")


@app.get("/relationships", summary="查询所有关系")
async def get_all_relationships():
    results = NDB.get_all_relationships()
    if results:
        return CommonResponse.success(results)
    else:
        return CommonResponse.error(404, "未找到相关数据")


@app.post("/init", summary="载入自带数据")
async def init():
    with open('./news_summary.txt', 'r', encoding='utf-8') as f:
        sents = f.readlines()
    result = get_triple_groups(sents)
    for i in result:
        NDB.add_triplet(i[0], i[1], i[2])
    return CommonResponse.success("载入自带数据成功")


if __name__ == '__main__':
    uvicorn.run(app="main:app", host="localhost", port=8000, reload=True)

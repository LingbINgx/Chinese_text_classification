#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 今日头条新闻分类数据爬取

import json
import time
import random

'''
100 民生 故事 news_story
101 文化 文化 news_culture
102 娱乐 娱乐 news_entertainment
103 体育 体育 news_sports
104 财经 财经 news_finance
105 时政 新时代 nineteenth
106 房产 房产 news_house
107 汽车 汽车 news_car
108 教育 教育 news_edu 
109 科技 科技 news_tech
110 军事 军事 news_military
111 宗教 无，凤凰佛教等来源
112 旅游 旅游 news_travel
113 国际 国际 news_world
114 证券 股票 stock
115 农业 三农 news_agriculture
116 电竞 游戏 news_game
'''

g_cnns = [
[100, '民生 故事', 'news_story'],
[101, '文化 文化', 'news_culture'],
[102, '娱乐 娱乐', 'news_entertainment'],
[103, '体育 体育', 'news_sports'],
[104, '财经 财经', 'news_finance'],
# [105, '时政 新时代', 'nineteenth'],
[106, '房产 房产', 'news_house'],
[107, '汽车 汽车', 'news_car'],
[108, '教育 教育', 'news_edu' ],
[109, '科技 科技', 'news_tech'],
[110, '军事 军事', 'news_military'],
# [111 宗教 无，凤凰佛教等来源],
[112, '旅游 旅游', 'news_travel'],
[113, '国际 国际', 'news_world'],
[114, '证券 股票', 'stock'],
[115, '农业 三农', 'news_agriculture'],
[116, '电竞 游戏', 'news_game']
]

g_ua = 'Dalvik/1.6.0 (Linux; U; Android 4.4.4; MuMu Build/V417IR) NewsArticle/6.3.1 okhttp/3.7.0.2'


g_id_cache = {}
g_count = 0

def get_data(tup):
    with open('../data/toutiao_cat_data.txt', 'a') as f:
        f.write(json.dumps(tup, ensure_ascii=False) + '\n')
        
    
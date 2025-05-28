import numpy as np
import math as m
import requests
import json
import time 
import akshare as ak

id = '000899'

# A股
stock_sse_summary_df = ak.stock_sse_summary()
print(stock_sse_summary_df)

# 人气榜
stock_hot_rank_em_df = ak.stock_hot_rank_em()
print(stock_hot_rank_em_df)

# 个股信息
stock_individual_info_em_df = ak.stock_individual_info_em(symbol=id)
print(stock_individual_info_em_df)

# 当日分时数据
stock_intraday_em_df = ak.stock_intraday_em(symbol=id)
print(stock_intraday_em_df)


# 人气
prefix = 'sz' if id[0:2] == '00' else 'sh' if id[0:2] == '60' else 'bj'
stock_hot_rank_detail_realtime_em_df = ak.stock_hot_rank_detail_realtime_em(symbol=f"{prefix}{id}")
print(stock_hot_rank_detail_realtime_em_df)

class Stocraft:
    def __init__(self):
        pass
import multiprocessing as mp
import akshare as ak
import pandas as pd
import numpy as np
import time
import os
import sys
import argparse
import requests

from datetime import datetime,timedelta
from requests.exceptions import ReadTimeout
from queue import Queue

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer,encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer,encoding='utf-8')
    
import threading
import multiprocessing as mp
from multiprocessing.managers import ValueProxy,ListProxy,DictProxy

import glob

db_dir = 'db'

def feature(code,date,interval_seconds = 5):
    特征 = pd.DataFrame(columns=['起时','终时','起价','终价','代价','涨幅','均价'])
    
    transaction_filepath = os.path.join(db_dir,f'{code}-{date}-交易.csv')
    info_filepath = os.path.join(db_dir,f'{code}-{date}-信息.csv')
    rank_filepath = os.path.join(db_dir,f'{code}-{date}-人气.csv')

    if not os.path.exists(transaction_filepath):
        return 特征

    交易 = pd.read_csv(transaction_filepath)
    信息 = pd.read_csv(info_filepath)
    人气 = pd.read_csv(rank_filepath)

    基价 = float(信息[信息['item'] == '昨收'].iloc[0]['value'])
    起价 = 基价

    for i,r in 交易.iterrows():
        时间,成交价,手数,买卖盘性质 = r['时间'],r['成交价'],r['手数'],r['买卖盘性质']
        if 买卖盘性质 == '中性盘':
            continue

        if 时间.startswith('09:25'):
            起时 = 终时 = 时间
            终价 = 成交价
            代价 = round((成交价 * 手数 * 100) / 1e4,1)
            涨幅 = round((成交价 - 起价) / 基价 * 100,2)
            均价 = round((终价 - 起价) / 2 + 起价,2)
            当前 = 起时,终时,起价,终价,代价,涨幅,均价,买卖盘性质
        else:
            _,终时,_,终价,_,_,_,性质 = 当前

            time1 = datetime.strptime(终时, "%H:%M:%S")
            time2 = datetime.strptime(时间, "%H:%M:%S")
            time_diff = time2 - time1
            incre_diff = round((终价 - 起价) / 基价 * 100,2) - round((成交价 - 起价) / 基价 * 100,2)

            if (买卖盘性质 != 性质 and 成交价 == 终价):
                买卖盘性质 = 性质

            if time_diff.total_seconds() > interval_seconds or (性质 != 买卖盘性质 and 成交价 != 终价):
                当前位置 = len(特征.index)
                涨幅 = round((终价 - 起价) / 基价 * 100,2)
                均价 = round((终价 + 起价) / 2,2)

                if 特征.shape[0]:
                    最近 = 特征.loc[当前位置-1]
                    最近涨幅 = 最近['涨幅']
                    time1 = datetime.strptime(最近['终时'], "%H:%M:%S")
                    time2 = datetime.strptime(起时, "%H:%M:%S")
                    time_diff = time2 - time1

                    if time_diff.total_seconds() < interval_seconds:
                        if 最近['起价'] == 最近['终价'] \
                            or 最近['起价'] < 最近['终价'] and 最近涨幅 + 涨幅 >= 最近涨幅 \
                            or 最近['起价'] > 最近['终价'] and 最近涨幅 + 涨幅 <= 最近涨幅: 
                        
                            起时 = 最近['起时']
                            起价 = 最近['起价']
                            代价 = 最近['代价'] + 代价
                            当前位置 = 当前位置 - 1

                特征.loc[当前位置] = [起时,终时,起价,终价,代价,涨幅,均价]

                涨幅 = 0
                代价 = 0
                起时 = 时间
                起价 = 终价

            代价 += round((成交价 * 手数 * 100) / 1e4,1)
            终时 = 时间
            终价 = 成交价
            当前 = 起时,终时,起价,终价,代价,涨幅,均价,买卖盘性质
    return 特征


def measure(deals,ranks,info,freq=10):
    df = pd.DataFrame(columns=['时间','买手','卖手','买额','卖额','涨跌','价格'])
    deals = deals[(deals['买卖盘性质'] != '中性盘')].copy()

    if 0 == deals.shape[0]: return df
    deals : pd.DataFrame

    deals['金额'] = deals['成交价'] * deals['手数'] * 100
    deals['_时间'] = pd.to_datetime(deals['时间'])
    deals.set_index('_时间',inplace=True)
    date = info[info['item'] == '日期'].iloc[0]['value']
    价格 = 昨收价 = float(info[info['item'] == '昨收'].iloc[0]['value'])
    涨跌 = 0
    window_size = int(60 / freq * 2.5)
    last_time = '9:25'
    a = pd.date_range(date + ' 9:25',date + ' 15:00',freq=f'{freq}s',inclusive='right')
    b = pd.date_range(date + ' 11:30',date + ' 13:00',freq=f'{freq}s',inclusive='right')
    for i,cur in enumerate(a.difference(b)):
        cur_time = cur.strftime('%H:%M:%S')
        买卖盘 = deals.between_time(last_time,cur_time)
        买盘 = 买卖盘[买卖盘['买卖盘性质'] == '买盘']
        卖盘 = 买卖盘[买卖盘['买卖盘性质'] == '卖盘']
        买手 = 买盘['手数']
        卖手 = 卖盘['手数']
        买总手 = round(买手.sum(),1)
        卖总手 = round(卖手.sum(),1)
        买盘金额 = 买盘['金额']
        卖盘金额 = 卖盘['金额']
        买总额 = round(买盘金额.sum() / 1e7,1)
        卖总额 = round(卖盘金额.sum() / 1e7,1)
        成交价 = 买卖盘['成交价']        
        
        if not 成交价.empty:
            涨跌 = round((成交价.iloc[-1] / 昨收价 - 1) * 100,2)
            价格 = 成交价.iloc[-1]
        
        if i < window_size:
            df.loc[i] = (date + ' ' + cur_time,买总手/window_size,卖总手/window_size,买总额/window_size,卖总额/window_size,涨跌,价格)
        else:
            df.loc[i] = (date + ' ' + cur_time,买总手,卖总手,买总额,卖总额,涨跌,价格)

        if cur > deals.index.values[-1]:
            break

        last_time = cur_time

    def gaussian_weights(n):
        x = np.linspace(-1, 0, n)
        sigma = 0.5
        weights = np.exp(-(x ** 2) / (2 * sigma ** 2))
        return weights
    
    weights = gaussian_weights(window_size)
    df[['买流量', '卖流量']] = df[['买额', '卖额']].rolling(window=window_size).apply(lambda x: np.sum(x * weights) / np.sum(weights), raw=True).round(1)
    df.loc[df.index[:window_size],'买流量'] = np.linspace(0,df.loc[0,'买额'],window_size)
    df.loc[df.index[:window_size],'卖流量'] = np.linspace(0,df.loc[0,'卖额'],window_size)
    
    return df

def up(worker_req : mp.Queue,worker_res : mp.Queue,*args):
    codes,date,days,cap,strategy = args
    up_df = pd.DataFrame(columns=['日期','代码','名称','市值','行业','涨幅','评分'])

    stocks = get_stock_spot()
    
    stocks_seleted = stocks[stocks['代码'].isin(codes)]
    stocks = stocks[(stocks['流通市值'] >= cap[0] * 1e8) & (stocks['流通市值'] <= cap[1] * 1e8)]
    stocks = pd.concat([stocks, stocks_seleted], ignore_index=True).drop_duplicates()
    end = datetime.strptime(date,'%Y%m%d')
    start = end - timedelta(days)
    dates = [d.strftime('%Y%m%d') for d in pd.date_range(start,end,freq='1D')]
    
    from matplotlib import pyplot as plt
    for date in dates:
        stocks.apply(lambda r: worker_req.put(('feature',r['代码'],date)),axis=1)
    dfs = dict()

    for _ in range(stocks.shape[0]*len(dates)):
        fun,code,date,feature_df = worker_res.get()
        feature_df : pd.DataFrame
        feature_df.insert(0,'时间', date + ' ' + feature_df['起时'])
        if code not in dfs:
            dfs[code] = feature_df
        else:
            dfs[code] = pd.concat([dfs[code], feature_df], ignore_index=True)

    if strategy == '触底反弹':
        feature_df = dfs['002067']
        bins = [0, 100, 500, 1000, 1000000]
        labels = ['小散', '牛散', '游资', '主力']
        feature_df['资金类别'] = pd.cut(feature_df['代价'], bins=bins, labels=labels)
        # distribution = feature_df.groupby('代价区间',observed=True).agg({'代价':'sum','涨幅':'sum','均价':'mean',}).round(2)
        

        pass       

    score = 0

def play(worker_req : mp.Queue,worker_res : mp.Queue,codes,date,days):
    start = datetime.strptime(date,'%y%m%d')
    end = start - timedelta(days)
    if days < 0: start,end = end,start
    dates = [d.strftime('%Y%m%d') for d in pd.date_range(start,end,freq='1D')]

    for date in dates:
        up(worker_req,worker_res,codes,date,-3,'')

def get_stock_spot():
    h15 = datetime.now().replace(hour=15, minute=0, second=0)

    stocks_file_path = os.path.join(db_dir,f'0-0-行情.csv')

    try:
        if os.path.exists(stocks_file_path) and datetime.now().date() == datetime.fromtimestamp(os.path.getmtime(stocks_file_path)).date():
            stocks = pd.read_csv(stocks_file_path,dtype={'代码':str})
        elif not os.path.exists(stocks_file_path) or datetime.now().weekday() < 5 and h15 < datetime.now():
            stocks = ak.stock_zh_a_spot_em()
            condition = ((stocks['代码'].str.startswith('00')) | (stocks['代码'].str.startswith('60'))) & \
                        (~stocks['名称'].str.startswith('PT')) & \
                        (~stocks['名称'].str.startswith('ST')) & \
                        (~stocks['名称'].str.startswith('*ST')) & \
                        (~stocks['最新价'].isnull()) & \
                        (stocks['换手率'] > 0.001)
            stocks = stocks[condition].reset_index(drop=True)
            stocks.to_csv(stocks_file_path,index=False)
            spot_filepath = os.path.join(db_dir,f'0-{datetime.now().strftime("%Y%m%d")}-行情.csv')
            stocks.to_csv(spot_filepath,index=False) 
        else:
            raise Exception('获取股票行情失败')
    except:
        file_paths = glob.glob(os.path.join(db_dir, '0-*-行情.csv'))
        if not file_paths:
            return None
        
        stocks = pd.read_csv(file_paths[-1],dtype={'代码':str})

    return stocks

def get_stock_intraday(code,date):
    try:
        filepath = os.path.join(db_dir,f'{code}-{date}-信息.csv')
        if not os.path.exists(filepath) or 0 == os.path.getsize(filepath):
            info = ak.stock_individual_info_em(code,10)
            if info[info['item'] == '总市值'].iloc[0]['value'] == '-': return True
            if info[info['item'] == '股票代码'].iloc[0]['value'] != code: return True
            bid = ak.stock_bid_ask_em(symbol=code)
            pd.concat([info,bid[20:]],ignore_index=True).to_csv(filepath,index=False)

        filepath = os.path.join(db_dir,f'{code}-{date}-交易.csv')
        if not os.path.exists(filepath) or 0 == os.path.getsize(filepath):
            deals = ak.stock_intraday_em(symbol=code)
            deals.to_csv(filepath,index=False)

        filepath = os.path.join(db_dir,f'{code}-{date}-人气.csv')
        if not os.path.exists(filepath) or 0 == os.path.getsize(filepath):
            prefix = 'SH' if code[0:2] == '60' else 'SZ' if code[0:2] == '00' else ''
            ranks = ak.stock_hot_rank_detail_realtime_em(prefix + code)
            ranks = ranks[50:].reset_index(drop=True)
            ranks['时间'] = ranks['时间'].str[11:]
            ranks.to_csv(filepath,index=False)
    except (ConnectionError, ReadTimeout, ValueError, ConnectionResetError,requests.exceptions.ChunkedEncodingError,requests.exceptions.ConnectionError):
        return False
    except (BrokenPipeError, KeyboardInterrupt):
        return False
    except:
        return True

    return True

def worker(id,req : mp.Queue,res : mp.Queue):
    while True:
        args = req.get()
    
        if args[0] == 'sync':
            if not get_stock_intraday(args[1],args[2]):
                req.put(args)
        else:
            fun = args[0]
            code = args[1]
            date = args[2]
            val = eval(f'{fun}("{code}","{date}")')
            res.put((fun,code,date,val))

def data_syncing_of_stock_intraday(worker_req : mp.Queue,worker_res : mp.Queue):
    while True:
        os.makedirs(db_dir,exist_ok=True)

        now = datetime.now()
        h9 = now.replace(hour=9, minute=15, second=0)
        h11 = now.replace(hour=11, minute=30, second=0)
        h13 = now.replace(hour=13, minute=0, second=0)
        h15 = now.replace(hour=15, minute=0, second=0)
        h24 = now.replace(hour=23, minute=59, second=59)

        while datetime.now() < h15:
            time.sleep(60)
            continue
        
        stocks = get_stock_spot()
        date = now.strftime('%Y%m%d')
        start = now.replace(hour=9, minute=30, second=0).strftime('%Y-%m-%d %H:%M:%S')
        end = now.replace(hour=9, minute=40, second=0).strftime('%Y-%m-%d %H:%M:%S')
        df = ak.stock_zh_a_hist_min_em(symbol="000001", start_date=start, end_date=end, period="5", adjust="")
        if now.weekday() < 5 and not df.empty: stocks.apply(lambda r: worker_req.put(('sync',r['代码'],date)),axis=1)

        while datetime.now() < h24:
            time.sleep(60)
            continue

if __name__ == '__main__':
    shared = mp.Manager()
    worker_req = shared.Queue()
    worker_res = shared.Queue()

    for i in range(min(4,os.cpu_count())):
        process = mp.Process(target=worker,name=f'牛马-{i}',args=[i,worker_req,worker_res],daemon=True)
        process.start()
 
    while True:
        if len(sys.argv) > 1:
            cmd = sys.argv[1:]
            sys.argv = sys.argv[0:1]
        else:
            try:
                cmd = input('> ').strip().split(' ')
            except (EOFError,KeyboardInterrupt):
                break

        parser = argparse.ArgumentParser()
        parser.add_argument('mode', type=str, help='The mode to run the program')
        parser.add_argument('--name',type=str,default='')
        parser.add_argument('--code',type=str,default='[]')
        parser.add_argument('--date',type=str,default=datetime.now().strftime('%Y%m%d'))
        parser.add_argument('--days',type=int,default=0)
        parser.add_argument('--cap',type=str,default='(50,150)')
        
        args = parser.parse_args(cmd)
        
        if args.mode == 'sync':
            threading.Thread(target=data_syncing_of_stock_intraday,name='股票数据同步',args=[worker_req,worker_res],daemon=True).start()
        elif args.mode == 'up':
            codes = args.code.split(',')
            up(worker_req,worker_res,codes,args.date,args.days,eval(args.cap),'触底反弹')
        elif args.mode == 'play':
            codes = args.code.split(',')
            play(worker_req,worker_res,codes,args.date,args.days)
        else:
            pass
        
        print('-')
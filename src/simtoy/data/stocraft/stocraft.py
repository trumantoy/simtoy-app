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

def sync(args : DictProxy=dict(),records : DictProxy=dict()):
    os.makedirs(db_dir,exist_ok=True)
    now = datetime.now(); date = now.strftime('%Y%m%d')

    h9 = datetime.now().replace(hour=9, minute=15, second=0)
    if datetime.now() < h9 and len(records) != 0:
        args['status'] = '没有到数据采集时间'
        return 60

    stocks_file_path = os.path.join(db_dir,f'0-0-行情.csv')
    dtype = {'代码':str}

    try:
        if datetime.now().weekday() > 4:
            raise Exception('不是工作日')
        elif datetime.now() < h9 or os.path.exists(stocks_file_path) and datetime.now().date() == datetime.fromtimestamp(os.path.getmtime(stocks_file_path)).date():
            stocks = pd.read_csv(stocks_file_path,dtype=dtype)
        else:
            stocks = ak.stock_zh_a_spot_em()
            condition = ((stocks['代码'].str.startswith('00')) | (stocks['代码'].str.startswith('60'))) & \
                        (~stocks['名称'].str.startswith('PT')) & \
                        (~stocks['名称'].str.startswith('ST')) & \
                        (~stocks['名称'].str.startswith('*ST')) & \
                        (~stocks['最新价'].isnull())
            stocks = stocks[condition].reset_index(drop=True)
            stocks.to_csv(stocks_file_path,index=False)
    except:
        file_paths = glob(os.path.join('db', '0-*-行情.csv'))
        if not file_paths:
            args['status'] = '获取股票列表失败'
            return 60*60
        
        stocks_file_path = file_paths[-1]
        stocks = pd.read_csv(stocks_file_path,dtype=dtype)

    stocks['更新时间'] = datetime.now().strftime('%H:%M:%S')
    stocks.sort_values(by='代码',inplace=True)
    args['indexes'] = dict(zip(stocks['代码'],range(stocks.shape[0])))
    args['codes'] = list()
    records.update(stocks[['代码','名称','最新价','涨跌幅','换手率','总市值','流通市值','更新时间']].to_dict(orient='list'))
    args['status'] = '更新数据'

    h15 = datetime.now().replace(hour=15, minute=0, second=0)
    while datetime.now() < h15:
        args['status'] = f'跟踪{len(args["codes"])}支股票'
        time.sleep(5)

    spot_filepath = os.path.join(db_dir,f'0-{date}-行情.csv')
    if os.path.exists(spot_filepath):
        args['status'] = '数据已采集'
        return 60*60
        
    start = datetime.now().replace(hour=9, minute=30, second=0).strftime('%Y-%m-%d %H:%M:%S')
    end = datetime.now().replace(hour=9, minute=40, second=0).strftime('%Y-%m-%d %H:%M:%S')
    stock_zh_a_hist_min_em_df = ak.stock_zh_a_hist_min_em(symbol="000001", start_date=start, end_date=end, period="5", adjust="")
    if stock_zh_a_hist_min_em_df.shape[0] == 0:
        args['status'] = '休假日'
        return 60*60

    part_stocks = stocks.copy(True)
    err_part_stocks = pd.DataFrame(columns=part_stocks.columns)
    while part_stocks.shape[0]:
        for i,(_,r) in enumerate(part_stocks.iterrows()):
            try:
                code = r['代码']
                filepath = os.path.join(db_dir,f'{code}-{date}-信息.csv')
                if not os.path.exists(filepath) or 0 == os.path.getsize(filepath):
                    info = ak.stock_individual_info_em(code,10)
                    if info[info['item'] == '总市值'].iloc[0]['value'] == '-':
                        continue

                    if info[info['item'] == '股票代码'].iloc[0]['value'] != code:
                        raise Exception(code,info)
                    
                    bid = ak.stock_bid_ask_em(symbol=code)
                    info.loc[len(info.index)] = ['市盈率-动态',r['市盈率-动态']]
                    info.loc[len(info.index)] = ['昨收',bid[bid['item'] == '昨收'].iloc[0]['value']]
                    info.loc[len(info.index)] = ['今开',bid[bid['item'] == '今开'].iloc[0]['value']]
                    info.to_csv(filepath,index=False)

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

                msg = f'{i+1}/{part_stocks.shape[0]} {code}'
                args['status'] = msg
            except (ConnectionError, ReadTimeout, ValueError, ConnectionResetError,requests.exceptions.ChunkedEncodingError,requests.exceptions.ConnectionError):
                err_part_stocks.loc[err_part_stocks.shape[0]] = r
            except (BrokenPipeError, KeyboardInterrupt):
                print('强行终止')
                args['status'] = '强行终止'
                return 60
            except:
                print('出现错误')
                args['status'] = '出现错误'
                return 60

        part_stocks = err_part_stocks
        err_part_stocks = pd.DataFrame(columns=part_stocks.columns)
        if len(part_stocks): print('错误集',part_stocks,len(part_stocks),file=sys.stderr)
    
    stocks.to_csv(spot_filepath,index=False)
    args['status'] = '数据采集完成'
    return 60 * 60

def feature(code,date,interval_seconds = 5):
    特征 = pd.DataFrame(columns=['起时','终时','起价','终价','代价','涨幅','支撑价'])
    
    transaction_filepath = os.path.join(db_dir,f'{code}-{date}-交易.csv')
    info_filepath = os.path.join(db_dir,f'{code}-{date}-行情.csv')
    rank_filepath = os.path.join(db_dir,f'{code}-{date}-人气.csv')

    if not os.path.exists(transaction_filepath):
        return 特征

    交易 = pd.read_csv(transaction_filepath)
    行情 = pd.read_csv(info_filepath)
    人气 = pd.read_csv(rank_filepath)

    基价 = float(行情[行情['item'] == '昨收'].iloc[0]['value'])
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
            支撑价 = round((终价 - 起价) / 2 + 起价,2)
            当前 = 起时,终时,起价,终价,代价,涨幅,支撑价,买卖盘性质
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
                支撑价 = round((终价 + 起价) / 2,2)

                if 特征.shape[0]:
                    最近 = 特征.loc[当前位置-1]
                    最近涨幅 = 最近['涨幅']
                    time1 = datetime.strptime(最近['终时'], "%H:%M:%S")
                    time2 = datetime.strptime(起时, "%H:%M:%S")
                    time_diff = time2 - time1

                    # if time_diff.total_seconds() < interval_seconds:
                    #     if 最近['起价'] == 最近['终价'] \
                    #         or 最近['起价'] < 最近['终价'] and 最近涨幅 + 涨幅 >= 最近涨幅 \
                    #         or 最近['起价'] > 最近['终价'] and 最近涨幅 + 涨幅 <= 最近涨幅: 
                        
                    #         起时 = 最近['起时']
                    #         起价 = 最近['起价']
                    #         代价 = 最近['代价'] + 代价
                    #         当前位置 = 当前位置 - 1

                特征.loc[当前位置] = [起时,终时,起价,终价,代价,涨幅,支撑价]

                涨幅 = 0
                代价 = 0
                起时 = 时间
                起价 = 终价

            代价 += round((成交价 * 手数 * 100) / 1e4,1)
            终时 = 时间
            终价 = 成交价
            当前 = 起时,终时,起价,终价,代价,涨幅,支撑价,买卖盘性质
    return 特征

def up(worker_req : mp.Queue,worker_res : mp.Queue,codes,date,days):
    up_df = pd.DataFrame(columns=['日期','代码','名称','市值','行业','涨幅','评分'])

    stocks = get_stock_spot()
    stocks_seleted = stocks[stocks['代码'].isin(codes)]
    min_market_cap = 0 * 1e8  # 50亿
    max_market_cap = 0 * 1e8  # 150亿
    stocks = stocks[(stocks['流通市值'] >= min_market_cap) & (stocks['流通市值'] <= max_market_cap)]
    stocks = pd.concat([stocks, stocks_seleted], ignore_index=True).drop_duplicates()

    start = datetime.strptime(date,'%y%m%d')
    end = start - timedelta(days)
    if days < 0: start,end = end,start
    dates = [d.strftime('%Y%m%d') for d in pd.date_range(start,end,freq='1D')]

    from matplotlib import pyplot as plt

    for date in dates:
        stocks.apply(lambda r: worker_req.put(('feature',r['代码'],date)),axis=1)
    
        for _ in range(stocks.shape[0]):
            fun,code,date,feature_df = worker_res.get()
            feature_df : pd.DataFrame
            bins = [0, 100, 500, 1000, 1000000]
            labels = ['小散', '牛散', '游资', '主力']
            feature_df['代价区间'] = pd.cut(feature_df['代价'], bins=bins, labels=labels)
            distribution = feature_df.groupby('代价区间',observed=True).agg({'代价':'sum','涨幅':'sum','支撑价':'mean',}).round(2)
            
            print(distribution)
            评分 = 0
            up_df.loc[up_df.index] = [date,code,stocks[stocks['代码'] == code]['名称'].values[0],
                                      stocks[stocks['代码'] == code]['流通市值'].values[0],
                                      stocks[stocks['代码'] == code]['行业'].values[0],
                                      distribution['涨幅'].sum(),评分]

def play(worker_req : mp.Queue,worker_res : mp.Queue,codes,date,days):
    start = datetime.strptime(date,'%y%m%d')
    end = start - timedelta(days)
    if days < 0: start,end = end,start
    dates = [d.strftime('%Y%m%d') for d in pd.date_range(start,end,freq='1D')]

    for date in dates:
        up(worker_req,worker_res,codes,date,-3)

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
        parser.add_argument('--code',type=str,default='')
        parser.add_argument('--date',type=str,default=datetime.now().strftime('%y%m%d'))
        parser.add_argument('--days',type=int,default=0)
        
        args = parser.parse_args(cmd)
        
        if args.mode == 'sync':
            threading.Thread(target=data_syncing_of_stock_intraday,name='股票数据同步',args=[worker_req,worker_res],daemon=True).start()
        elif args.mode == 'up':
            codes = args.code.split(',')
            up(worker_req,worker_res,codes,args.date,args.days)
        elif args.mode == 'play':
            codes = args.code.split(',')
            play(worker_req,worker_res,codes,args.date,args.days)
        else:
            pass
        
        print('-')
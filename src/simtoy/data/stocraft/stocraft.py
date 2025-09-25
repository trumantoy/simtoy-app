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

def feature(code,days=7):
    date_end = datetime.now().date()
    date_start = datetime.now().date() - timedelta(days=days)

    特征 = pd.DataFrame(columns=['时间','买卖比','买总额','卖总额','涨幅','小额占比','中额占比','大额占比','巨额占比'])
    for date_cur in pd.date_range(date_start,date_end,freq='1D'):
        file_date = date_cur.date().strftime('%Y%m%d')
        transaction_filepath = os.path.join(db_dir,f'{code}-{file_date}-交易.csv')
        info_filepath = os.path.join(db_dir,f'{code}-{file_date}-信息.csv')
        rank_filepath = os.path.join(db_dir,f'{code}-{file_date}-人气.csv')
   
        if not os.path.exists(transaction_filepath):
            continue

        交易 = pd.read_csv(transaction_filepath)
        信息 = pd.read_csv(info_filepath)
        # 人气 = pd.read_csv(rank_filepath)

        # 人气['time'] = pd.to_datetime(file_date + ' ' + 人气['时间'])
        # 人气 = 人气.set_index('time').between_time('9:20','15:0').reset_index(drop=True)
        # 人气['时间'] = file_date + ' ' + 人气['时间']
        # 人气['热度'] = np.log2(人气['排名'])

        信息.loc[len(信息.index)] = ['日期',file_date]
        交易['时间'] = file_date + ' ' + 交易['时间']

        交易 = 交易[(交易['买卖盘性质'] != '中性盘')].copy()
        if 0 == 交易.shape[0]: continue
        交易['金额'] = 交易['成交价'] * 交易['手数'] * 100
        买盘 = 交易[交易['买卖盘性质'] == '买盘']
        卖盘 = 交易[交易['买卖盘性质'] == '卖盘']
        买总手 = round(买盘['手数'].sum())
        卖总手 = round(卖盘['手数'].sum())
        买总额 = round(买盘['金额'].sum() / 1e8,1)
        卖总额 = round(卖盘['金额'].sum() / 1e8,1)
        昨收价 = float(信息[信息['item'] == '昨收'].iloc[0]['value'])
        涨幅 = round((交易['成交价'].iloc[-1] / 昨收价 - 1) * 100,2)

        总额 = (买总额 + 卖总额) * 1e4
        金额 = 交易['金额'] / 1e4
        小额占比 = 金额[(金额 < 10)].sum() / 总额
        中额占比 = 金额[(10 < 金额) & (金额 < 50)].sum() / 总额
        大额占比 = 金额[(50 < 金额) & (金额 < 100)].sum() / 总额
        巨额占比 = 金额[(100 < 金额)].sum() / 总额

        特征.loc[len(特征.index)] = [
            file_date,round(买总手/卖总手 if 卖总手>0 else 0,2),
            买总额,卖总额,涨幅,
            round(小额占比,2),round(中额占比,2),
            round(大额占比,2),round(巨额占比,2)]
    return 特征

def up(args,records,up_records : dict):
    df_records = pd.DataFrame(dict(zip(records.keys(),records.values())))
    df_up_records = pd.DataFrame(columns=['代码','名称','涨幅','流通市值','评分'])


    for i,row in df_records.iterrows(): 
        code = row['代码']
        name = row['名称']

        feature_filepath = os.path.join(db_dir,f'{code}-0-特征.csv')
        if os.path.exists(feature_filepath) and datetime.now().date() == datetime.fromtimestamp(os.path.getmtime(feature_filepath)).date():
            特征 = pd.read_csv(feature_filepath)
        else:
            特征 = feature(code)
            特征.to_csv(feature_filepath)

        a = sum(特征['买卖比'] > 1.5)
        b = sum(特征['巨额占比'] > 0.5)
        涨跌 = 特征['涨幅'].iloc[-1]
        c = 1 if 0 < 涨跌 < 5 else -1
        d = 2 if 60 < row['流通市值'] / 1e8 < 150 else -2

        s = a + b + c + d
        df_up_records.loc[len(df_up_records.index)] = [code,name,涨跌,round(row['流通市值'] / 1e8,1),s]

        if i % 10 == 0:
            up_records.update(df_up_records.to_dict(orient='list'))
    return 60*60

def get_stock_spot():
    h15 = datetime.now().replace(hour=15, minute=0, second=0)

    stocks_file_path = os.path.join(db_dir,f'0-0-行情.csv')

    try:
        if os.path.exists(stocks_file_path) and datetime.now().date() == datetime.fromtimestamp(os.path.getmtime(stocks_file_path)).date():
            stocks = pd.read_csv(stocks_file_path,dtype={'代码':str})
        elif datetime.now().weekday() < 5 and h15 < datetime.now():
            stocks = ak.stock_zh_a_spot_em()
            
            condition = ((stocks['代码'].str.startswith('00')) | (stocks['代码'].str.startswith('60'))) & \
                        (~stocks['名称'].str.startswith('PT')) & \
                        (~stocks['名称'].str.startswith('ST')) & \
                        (~stocks['名称'].str.startswith('*ST')) & \
                        (~stocks['最新价'].isnull()) & \
                        (stocks['换手率'] > 0)
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
            if info[info['item'] == '总市值'].iloc[0]['value'] == '-':
                return True

            if info[info['item'] == '股票代码'].iloc[0]['value'] != code:
                return True
            
            bid = ak.stock_bid_ask_em(symbol=code)
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
    except (ConnectionError, ReadTimeout, ValueError, ConnectionResetError,requests.exceptions.ChunkedEncodingError,requests.exceptions.ConnectionError):
        return False
    except (BrokenPipeError, KeyboardInterrupt):
        return False
    except:
        return True

    return True

def worker(id,req : Queue,res : Queue):
    while True:
        fun,code,date,sleep = req.get()
    
        if fun == 'sync':
            print(req.qsize(),code,flush=True)
            if not get_stock_intraday(code,date):
                req.put((fun,code,date,10))
        else:
            pass

def player(req : Queue):
    shared = mp.Manager()
    worker_req = shared.Queue()
    worker_res = shared.Queue()

    for i in range(os.cpu_count()):
        process = mp.Process(target=worker,name=f'牛马-{i}',args=[i,worker_req,worker_res],daemon=True)
        process.start()

    os.makedirs(db_dir,exist_ok=True)

    while True:
        now = datetime.now()
        h9 = now.replace(hour=9, minute=15, second=0)
        h11 = now.replace(hour=11, minute=30, second=0)
        h13 = now.replace(hour=13, minute=0, second=0)
        h15 = now.replace(hour=15, minute=0, second=0)
        h24 = now.replace(hour=23, minute=59, second=59)

        stocks = get_stock_spot()

        while datetime.now() < h9:
            worker_req(*req.get())
        
        while datetime.now() < h11:
            worker_req(*req.get())

        while datetime.now() < h13:
            worker_req(*req.get())

        while datetime.now() < h15:
            worker_req(*req.get())
        
        stocks = get_stock_spot()
        
        date = now.strftime('%Y%m%d')
        start = now.replace(hour=9, minute=30, second=0).strftime('%Y-%m-%d %H:%M:%S')
        end = now.replace(hour=9, minute=40, second=0).strftime('%Y-%m-%d %H:%M:%S')
        df = ak.stock_zh_a_hist_min_em(symbol="000001", start_date=start, end_date=end, period="5", adjust="")
        if not df.empty: stocks.apply(lambda x: worker_req.put(('sync',x['代码'],date,1)),axis=1)

        while datetime.now() < h24:
            worker_req(*req.get())

from queue import Queue

if __name__ == '__main__':
    req = Queue()
 
    threading.Thread(target=player,name='玩家',args=[req],daemon=True).start()

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
        if args.date: date = args.date
        if args.days: days = args.days
        
        if args.mode == 'sync':
            df = pd.DataFrame(dict(zip(records.keys(),records.values())))
            if df.shape[0]: print(df.to_string())
            print('-')
        elif args.mode == 'up':
            codes = args.code.split(',')
            # if not df.empty: print(df.to_string())
            print('-')
        elif args.mode == 'play':
            print('-')
        else:
            print(params['status'])
            print('-')
            pass
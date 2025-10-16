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

def feature(code,date,info,interval_seconds = 5):
    transaction_filepath = os.path.join(db_dir,f'{code}-{date}-交易.csv')
    if not os.path.exists(transaction_filepath):
        return pd.DataFrame(columns=['起时','终时','起价','终价','总价','均价','涨幅'])

    交易 = pd.read_csv(transaction_filepath)
    基价 = info['昨收']
    起价 = 基价

    时间,成交价,手数,买卖盘性质 = 交易.iloc[0]['时间'],交易.iloc[0]['成交价'],交易.iloc[0]['手数'],交易.iloc[0]['买卖盘性质']
    起时 = 终时 = 时间
    终价 = 成交价
    总价 = round((成交价 * 手数 * 100) / 1e4,1)
    涨幅 = round((成交价 - 起价) / 基价 * 100,2)
    均价 = round((终价 - 起价) / 2 + 起价,2)
    当前 = 起时,终时,起价,终价,总价,均价,涨幅,买卖盘性质

    rows = list()
    for i,r in 交易.iterrows():
        时间,成交价,手数,买卖盘性质 = r['时间'],r['成交价'],r['手数'],r['买卖盘性质']
        if 买卖盘性质 == '中性盘':
            continue
        
        if 时间.startswith('09:25'):
            起时 = 终时 = 时间
            终价 = 成交价
            总价 = round((成交价 * 手数 * 100) / 1e4,1)
            涨幅 = round((成交价 - 起价) / 基价 * 100,2)
            均价 = round((终价 - 起价) / 2 + 起价,2)
            当前 = 起时,终时,起价,终价,总价,均价,涨幅,买卖盘性质
        else:
            _,终时,_,终价,_,_,_,性质 = 当前

            time1 = datetime.strptime(终时, "%H:%M:%S")
            time2 = datetime.strptime(时间, "%H:%M:%S")
            time_diff = time2 - time1
            incre_diff = round((终价 - 起价) / 基价 * 100,2) - round((成交价 - 起价) / 基价 * 100,2)

            if (买卖盘性质 != 性质 and 成交价 == 终价):
                买卖盘性质 = 性质

            if time_diff.total_seconds() > interval_seconds or (性质 != 买卖盘性质 and 成交价 != 终价):
                涨幅 = round((终价 - 起价) / 基价 * 100,2)
                均价 = round((终价 + 起价) / 2,2)

                if len(rows):
                    最近 = rows[-1]
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
                            总价 = 最近['总价'] + 总价
                            涨幅 = 最近['涨幅'] + 涨幅
                            rows.pop()

                rows.append(pd.Series({'起时':起时,'终时':终时,'起价':起价,'终价':终价,'总价':总价,'均价':均价,'涨幅':round(涨幅,2),'性质':性质}))
                涨幅 = 0
                总价 = 0
                起时 = 时间
                起价 = 终价

            总价 += (成交价 * 手数 * 100)
            终时 = 时间
            终价 = 成交价
            当前 = 起时,终时,起价,终价,总价,均价,涨幅,买卖盘性质
    
    return pd.DataFrame(rows,columns=['起时','终时','起价','终价','总价','均价','涨幅','性质'])


# def measure(code,date,info,freq=10):
#     df = pd.DataFrame(columns=['时间','买手','卖手','买额','卖额','涨跌','价格'])
    
#     transaction_filepath = os.path.join(db_dir,f'{code}-{date}-交易.csv')
#     if not os.path.exists(transaction_filepath):
#         return pd.DataFrame(columns=['起时','终时','起价','终价','总价','均价','涨幅'])

#     交易 = pd.read_csv(transaction_filepath)
#     基价 = info['昨收']
#     起价 = 基价

#     交易 = 交易[(交易['买卖盘性质'] != '中性盘')].copy()
    
#     if 0 == 交易.shape[0]: return df
#     交易 : pd.DataFrame

#     交易['金额'] = 交易['成交价'] * 交易['手数'] * 100
#     交易['_时间'] = pd.to_datetime(交易['时间'])
#     交易.set_index('_时间',inplace=True)
#     date = info[info['item'] == '日期'].iloc[0]['value']
#     价格 = 昨收价 = float(info[info['item'] == '昨收'].iloc[0]['value'])
#     涨跌 = 0
#     window_size = int(60 / freq * 2.5)
#     last_time = '9:25'
#     a = pd.date_range(date + ' 9:25',date + ' 15:00',freq=f'{freq}s',inclusive='right')
#     b = pd.date_range(date + ' 11:30',date + ' 13:00',freq=f'{freq}s',inclusive='right')
#     for i,cur in enumerate(a.difference(b)):
#         cur_time = cur.strftime('%H:%M:%S')
#         买卖盘 = 交易.between_time(last_time,cur_time)
#         买盘 = 买卖盘[买卖盘['买卖盘性质'] == '买盘']
#         卖盘 = 买卖盘[买卖盘['买卖盘性质'] == '卖盘']
#         买手 = 买盘['手数']
#         卖手 = 卖盘['手数']
#         买总手 = round(买手.sum(),1)
#         卖总手 = round(卖手.sum(),1)
#         买盘金额 = 买盘['金额']
#         卖盘金额 = 卖盘['金额']
#         买总额 = round(买盘金额.sum() / 1e7,1)
#         卖总额 = round(卖盘金额.sum() / 1e7,1)
#         成交价 = 买卖盘['成交价']        
        
#         if not 成交价.empty:
#             涨跌 = round((成交价.iloc[-1] / 昨收价 - 1) * 100,2)
#             价格 = 成交价.iloc[-1]
        
#         if i < window_size:
#             df.loc[i] = (date + ' ' + cur_time,买总手/window_size,卖总手/window_size,买总额/window_size,卖总额/window_size,涨跌,价格)
#         else:
#             df.loc[i] = (date + ' ' + cur_time,买总手,卖总手,买总额,卖总额,涨跌,价格)

#         if cur > 交易.index.values[-1]:
#             break

#         last_time = cur_time

#     # def gaussian_weights(n):
#     #     x = np.linspace(-1, 0, n)
#     #     sigma = 0.5
#     #     weights = np.exp(-(x ** 2) / (2 * sigma ** 2))
#     #     return weights
    
#     # weights = gaussian_weights(window_size)
#     # df[['买流量', '卖流量']] = df[['买额', '卖额']].rolling(window=window_size).apply(lambda x: np.sum(x * weights) / np.sum(weights), raw=True).round(1)
#     # df.loc[df.index[:window_size],'买流量'] = np.linspace(0,df.loc[0,'买额'],window_size)
#     # df.loc[df.index[:window_size],'卖流量'] = np.linspace(0,df.loc[0,'卖额'],window_size)
    
#     return df

def measure(交易,freq=10):
    df = pd.DataFrame(columns=['时间','买手','卖手','买额','卖额','涨跌','价格'])
    
    # 基价 = 信息['昨收']
    # 起价 = 基价

    交易 = 交易[(交易['买卖盘性质'] != '中性盘')].copy()
    
    if 0 == 交易.shape[0]: return df
    交易 : pd.DataFrame

    交易['金额'] = 交易['成交价'] * 交易['手数'] * 100
    交易['_时间'] = pd.to_datetime(交易['时间'])
    交易.set_index('_时间',inplace=True)
    # 价格 = 昨收价 = 基价
    涨跌 = 0
    window_size = int(60 / freq * 2.5)
    last_time = '9:25'
    a = pd.date_range(date + ' 9:25',date + ' 15:00',freq=f'{freq}s',inclusive='right')
    b = pd.date_range(date + ' 11:30',date + ' 13:00',freq=f'{freq}s',inclusive='right')
    for i,cur in enumerate(a.difference(b)):
        cur_time = cur.strftime('%H:%M:%S')
        买卖盘 = 交易.between_time(last_time,cur_time)
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

        if cur > 交易.index.values[-1]:
            break

        # last_time = cur_time

    # def gaussian_weights(n):
    #     x = np.linspace(-1, 0, n)
    #     sigma = 0.5
    #     weights = np.exp(-(x ** 2) / (2 * sigma ** 2))
    #     return weights
    
    # weights = gaussian_weights(window_size)
    # df[['买流量', '卖流量']] = df[['买额', '卖额']].rolling(window=window_size).apply(lambda x: np.sum(x * weights) / np.sum(weights), raw=True).round(1)
    # df.loc[df.index[:window_size],'买流量'] = np.linspace(0,df.loc[0,'买额'],window_size)
    # df.loc[df.index[:window_size],'卖流量'] = np.linspace(0,df.loc[0,'卖额'],window_size)
    
    return df

def evaluate(code,date,info):
    pass

def up(worker_req : mp.Queue,worker_res : mp.Queue,*args):
    codes,date,days,cap = args

    stocks = pd.read_csv(os.path.join(db_dir,f'0-{date}-行情.csv'),dtype={'代码':str})
    stocks_seleted = stocks[stocks['代码'].isin(codes)]
    stocks = stocks[(stocks['流通市值'] >= cap[0] * 1e8) & (stocks['流通市值'] <= cap[1] * 1e8)]
    stocks = pd.concat([stocks, stocks_seleted], ignore_index=True).drop_duplicates()    

    end = datetime.strptime(date,'%Y%m%d')
    start = end - timedelta(days)
    dates = [d.strftime('%Y%m%d') for d in pd.date_range(start,end,freq='1D')][1:]

    stocks.apply(lambda r: worker_req.put(('feature',r['代码'],date,r)), axis=1)

    dfs = dict()
    for _ in range(stocks.shape[0]):
        fun,code,date,feature_df = worker_res.get()
        feature_df : pd.DataFrame
        if feature_df is None or feature_df.empty: continue
        feature_df.insert(0,'时间', date + ' ' + feature_df['起时'])
        
        if code in dfs:
            dfs[code].append(feature_df)
        else:
            dfs[code] = [feature_df]
            
    df = pd.DataFrame(columns=['代码','名称','市值','涨幅','评分','态势','参数','权重'])    
    for i,r in stocks.iterrows():
        if r['代码'] not in dfs: continue
        feature_df = pd.concat(dfs[r['代码']], ignore_index=True)
        feature_df['资金规模'] = pd.cut(feature_df['总价'], bins=[0, 100, 500, 1000, 1000000], labels=['小散', '牛散', '游资', '主力'])
        # feature_df['_时间'] = pd.to_datetime(feature_df['时间'])
        # feature_df = feature_df.set_index('_时间').sort_index()
        # 多方分布 = feature_df.groupby('资金规模',observed=True).count()
        if feature_df.empty: continue
        当日终价 = feature_df.tail(1)['终价'].item()
        套牢资金 = feature_df[(feature_df['总价'] > 100 * 1e4) & (feature_df['终价'] > 当日终价)]['总价'].sum().item()
        买方 = feature_df[feature_df['涨幅'] > 0]
        卖方 = feature_df[feature_df['涨幅'] < 0] 
        买方资金 = 买方['总价'].sum().item()
        卖方资金 = 卖方['总价'].sum().item()
        涨代价 = float((买方['总价'] / 买方['涨幅']).median() / 1e4)
        跌代价 = float((卖方['总价'] / -卖方['涨幅']).median() / 1e4)

        bais = 0.0
        涨停资金 = 10 * 1e8
        基准代价 = 1000 * 1e4

        a = (
            round(套牢资金 / 涨停资金,2),
            round((1 - 卖方资金 / 买方资金) if 买方资金 else -1,2),
            round((1 - 涨代价 / 跌代价) if not np.isnan(涨代价) and not np.isnan(跌代价) else -1,2),
        )
        w = (0.1,0.1,0.1)
        score = bais + a[0] * w[0] + a[1] * w[1] + a[2] * w[2]

        # 支撑位策略
        # 趋势上涨策略
        # 下套反弹策略
        df.loc[len(df.index)] = (r['代码'],r['名称'],round(r['流通市值'] / 1e8,2),r['涨跌幅'],round(score,2),'下套反弹',str(a),str(w))
        # 超-下套反弹策略
    return df
    
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

def sync_stock_intraday(code,date):
    filepath = os.path.join(db_dir,f'{code}-{date}-交易.csv')
    if not os.path.exists(filepath) or 0 == os.path.getsize(filepath):
        df = get_stock_intraday(code)
        df.to_csv(filepath,index=False)

def get_stock_intraday(code):
    i = 1
    while i:
        try: df = ak.stock_intraday_em(code)
        except: i += 1
        else: i = 0
    return df

def worker(id,req : mp.Queue,res : mp.Queue):
    while True:
        args = req.get()
        fun = args[0]
        code = args[1]
        date = args[2]
        info = args[3]
        val = eval(f'{fun}(code,date,info)')
        res.put((fun,code,date,val))

def data_syncing_of_stock_intraday(log : list):
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
        log[:] = []
        if now.weekday() < 5 and not df.empty: 
            stocks = stocks[(stocks['流通市值'] >= 80 * 1e8) & (stocks['流通市值'] <= 300 * 1e8)]

            for i,r in stocks.iterrows():
                sync_stock_intraday(r['代码'],date)
                log.append(f'{int(i)+1}/{stocks.shape[0]} {r["代码"]}-{r["名称"]}')

        while datetime.now() < h24:
            time.sleep(60)
            continue

if __name__ == '__main__':
    shared = mp.Manager()
    worker_req = shared.Queue()
    worker_res = shared.Queue()
    log = shared.list()

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
        parser.add_argument('--days',type=int,default=1)
        parser.add_argument('--cap',type=str,default='(80,300)')
        
        args = parser.parse_args(cmd)
        
        if args.mode == 'sync':
            threading.Thread(target=data_syncing_of_stock_intraday,args=[log],name='股票数据同步',daemon=True).start()
        elif args.mode == 'up':
            codes = args.code.split(',')
            df = up(worker_req,worker_res,codes,args.date,args.days,eval(args.cap))
            df = df.sort_values(by='评分').reset_index(drop=True)
            print(df.to_string())
            print(df[df['代码'].isin(codes)].to_string())
        elif args.mode == 'play':
            codes = args.code.split(',')
            play(worker_req,worker_res,codes,args.date,args.days)
        elif args.mode == 'measure':
            codes = args.code.split(',')
            for code in codes:
                交易 = get_stock_intraday(code)
                print(交易,flush=True)
        elif args.mode == 'test':
            pass
        else:
            while len(log):
                print(log.pop(0))
        print('-')
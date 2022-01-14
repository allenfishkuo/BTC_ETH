#import torch
#import torch.nn as nn
import numpy as np
#import new_dataloader
import MDD
from trade_trend import trade_down_slope, trade_up_slope, trade_normal
#import matrix_trading
import os 
import pandas as pd
#import torch
#import torch.utils.data as Data

import matplotlib.pyplot as plt
import time
import sys
import time
from multiprocessing import Pool
from correlation_check import correlation_check
#import FT_compare
path_to_tick = "/Users/kuoweilun/CryptoCurrency_PT-main/test_thousand"
#path_to_tick = "/Users/kuoweilun/CryptoCurrency_PT-main/BTC_ETH_test"

ext_of_compare = "_table.csv"
path_to_profit = "/Users/kuoweilun/CryptoCurrency_PT-main/profit_formation_test_5min_tens/"
path_to_profit = "/Users/kuoweilun/CryptoCurrency_PT-main/profit_BTC_ETH/"

open, loss = 1.5, 10#
trading_cost_threshold = 0.0015
max_hold = 500
trading_cost = 0.0015
capital = 300000000
cost_gate_Train = False
loading_data = False
dtype = {
    'S1' : str,
    'S2' : str,
    'VECMQ' : float,
    'mu': float,
    'Johansen_slope' : float,
    'stdev' : float,
    'model' : int,
    'w1' : float,
    'w2' : float
}

def return_dataframe(table,trade_capital_list,open_list,loss_list,reward_list,open_num_list, trading_history):
    df = pd.DataFrame(columns=['stock1','stock2','trade_capital','open','loss','reward','open_num','trading_history'])
    df['stock1'] = table.S1
    df['stock2'] = table.S2
    df['trade_capital'] = trade_capital_list
    df['open'] = open_list
    df['loss'] = loss_list
    df['reward'] = reward_list
    df['open_num'] = open_num_list
    df['trading history'] = trading_history
    return df

def test_reward(formation_time, year = 2021):
    start_time = time.time()
    path_to_average = "./"+str(time)+"/averageprice/"
    ext_of_average = "_averagePrice_min.csv"
    path_to_minprice = "./"+str(time)+"/minprice/"
    ext_of_minprice = "_min_stock.csv"
    total_reward = 0
    total_num = 0
    total_trade = [0,0,0]
    action_list = []
    action_list2 = []
    check = 0
    path_to_compare = f'/Users/kuoweilun/CryptoCurrency_PT-main/Crypto_Currency_Cointegration/formation_{formation_time}_5min_thousand_no_normality'
    #path_to_compare = f'/Users/kuoweilun//CryptoCurrency_PT-main/Crypto_Currency_Cointegration/BTC_ETH_table'
    datelist = [f.split('_')[0] for f in os.listdir(f'{path_to_compare}/')]
    print(datelist)
    #print(datelist[167:])
    profit_count = 0
    count = 0
    total_normal = 0
    program_file = f'{path_to_profit}/formation_{formation_time}/'
    program_file = f'{path_to_profit}/'
    if not os.path.exists(program_file):
                os.makedirs(program_file)
    for date in sorted(datelist[:]): #決定交易要從何時開始
        print(date)
        open_list = []
        loss_list = []
        trade_capital_list = []
        reward_list = []
        open_num_list = []
        trading_history = []
        negative_pair = {}
        table = pd.read_csv(f'{path_to_compare}/{date}_table.csv', dtype = dtype)
        tickdata = pd.read_csv(f'{path_to_tick}/{date[:4]}-{date[4:6]}-{date[6:8]}_daily_min_price.csv')
        tickdata = tickdata.iloc[:]
        tickdata.index = np.arange(0,len(tickdata),1)  
        num = np.arange(0,len(table),1)
        strategy = {
                    "up_open_time" : open,
                    "down_open_time" : open,
                    "stop_loss_time" : loss,
                    "maxhold" : max_hold,
                    "cost_gate" : trading_cost_threshold,
                    "capital" : capital,
                    "tax_cost" : trading_cost
                }
        #print(date)
        normal_table = table[table["model"]<4]
        count = 0 
        pair_data = []
        open_list = []
        loss_list = []
        for index, row in normal_table[:].iterrows():
            _trade, _profit, _capital, _return, _trading_rule,_history = 0, 0 ,0 ,0,[0,0,0],{}
            s1_tick = tickdata[row["S1"]]
            s2_tick = tickdata[row["S2"]]
            
            tmp_pair = row["S1"]+':'+row["S2"]
            #if correlation_check(formation_time, s1_tick, s2_tick)  :
            _trade, _profit, _capital, _return, _trading_rule, _history = trade_normal(s1_tick, s2_tick, row.to_dict(), strategy, formation_time)
            if _profit < 0 :
                if tmp_pair not in negative_pair:
                    negative_pair[tmp_pair] = 1
                else :
                    negative_pair[tmp_pair] += 1
            total_normal += _profit
            total_trade[0] += _trading_rule[0]
            total_trade[1] += _trading_rule[1]
            total_trade[2] += _trading_rule[2] 
            total_num += _trade
            
            open_list.append(open)
            loss_list.append(loss)
            trade_capital_list.append(_capital)
            reward_list.append(_profit)
            open_num_list.append(_trade)
            table.at[index,"_return"] = _return * 100
            table.at[index,"_profit"] = _profit
            """
            trading_history.append({
                "s1" : row["S1"],
                "s2" : row["S2"],
                "profit" : _profit ,
                "return" : _return ,
                "capital" : _capital ,
                "trade" : _trade,
                "history" : _history
            })
            """
            trading_history.append(_history)
            #print(f'each_profit : {_profit}')
            
        print(np.array(trading_history).shape)
        store_data = return_dataframe(normal_table[:],trade_capital_list,open_list,loss_list,reward_list,open_num_list,trading_history) 
        if not store_data.empty:
        #print(store_data)
            profit_count += sum([p > 0 for p in table["_profit"]])
        if loading_data :
            flag = os.path.isfile(f'{program_file}{date}_profit.csv')
            if not flag :
                store_data.to_csv(f'{program_file}{date}_profit.csv', mode='w',index=False)
        
    print(f'利潤 : {total_normal} and 開倉次數 : {total_num} and 開倉有賺錢的次數/開倉次數 : {profit_count/total_num}')
    print(f'開倉有賺錢次數 : {profit_count}')
    print("正常平倉 停損平倉 強迫平倉 :",total_trade[0],total_trade[1],total_trade[2])
    print("正常平倉率 :",total_trade[0]/total_num)
    print('Time used: {} sec'.format(time.time()-start_time))
    negative_pair ={k: v for k, v in sorted(negative_pair.items(), key=lambda item: item[1])}
    print(negative_pair)
    
    if loading_data :
        reward,return_reward,per_reward,max_cap ,datelist = MDD.reward_calculation(program_file)
        sharp_ratio, per_sharpe_ratio, mdd = MDD.plot_performance_with_dd(program_file, reward,return_reward,per_reward,datelist,total_num,total_trade[0],profit_count/total_num,max_cap )
        print(f'{total_reward:.2f}')
        win_rate = profit_count/total_num * 100
        normal_close_rate = total_trade[0]/total_num * 100
        print(f'{win_rate:.2f}%/{normal_close_rate:.2f}%')
        print(f'{total_trade[0]},{total_trade[1]},{total_trade[2]},{total_num}')
        print(f'{sharp_ratio[0]:.4f}')
        print(f'{per_sharpe_ratio[0]:.4f}')
        profit_per_open = total_reward / total_num
        print(f'{profit_per_open:.4f}/{max_cap:.2f}/{mdd:.2f}')
    return total_normal, profit_count/total_num, total_trade[0]/ total_num
    
if __name__ == '__main__':
    test_reward(120)
    
    
    
    
    
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:06:06 2022

@author: kuoweilun
"""

import torch
import torch.nn as nn
import numpy as np
import new_dataloader
import MDD
from trade_trend import trade_down_slope, trade_up_slope, trade_normal
#import matrix_trading
import os 
import pandas as pd
#import torch
import torch.utils.data as Data
#import datetime
import matplotlib.pyplot as plt
import time
import sys
import time
from multiprocessing import Pool
from correlation_check import correlation_check
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FuncFormatter
from datetime import datetime, timedelta


path_to_tick = "/home/allen/CryptoCurrency_TP"

ext_of_compare = "_table.csv"
path_to_profit = "/home/allen/CryptoCurrency_TP/profit_five_pairs_tradinglog/"
path_to_ground_truth = "/home/allen/CryptoCurrency_TP/BTC_ETH_ground_truth/"
#open, loss = 1.5, 1#
trading_cost_threshold = 0.0004
max_hold = 500
trading_cost = 0.0004
capital = 2000
max_capital = 100000
cost_gate_Train = False
loading_data = True
test_ML = True
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

actions = [[0.75,9],[1.0,7.5],[1.25,10],[1.5,10],[2.0,12.5],[float('inf'),float('inf')]] # ML learning所使用的threshold 透過model來決定不同組的pair需要啥threshold

def color_change(n):
    if n : return "g"
    else : return "r" 

def return_dataframe(reward_list,trading_history): #log 交易紀錄
    df = pd.DataFrame(columns=['reward','trading history'])
    #df['stock1'] = stock
    #df['stock2'] = table.S2
    #df['trade_capital'] = trade_capital_list
    #df['open'] = open_list
    #df['loss'] = loss_list
    df['reward'] = reward_list
    #df['open_num'] = open_num_list
    df['trading history'] = trading_history
    return df

def test_reward(formation_time,jump = 0, year = 2021): #testing的地方
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
    path_to_compare = f'/home/allen/CryptoCurrency_TP/BTC_ETH_table'
    #path_to_compare = f'/home/allen/CryptoCurrency_TP'
    #print(datelist[167:])
    profit_count = 0
    count = 0
    total_normal = 0
    program_file = f'{path_to_profit}/'
    table = pd.read_csv(f'{path_to_compare}/formation_{formation_time}/BTC_ETH_formation_table_p10.csv', dtype = dtype) # 讀cointegration table
    tickdata = pd.read_csv(f'{path_to_tick}/BTC_ETH_combine.csv') #read 5 minpirce data to trade
    #ground_truth = pd.read_csv(f'{path_to_ground_truth}/formation_{formation_time}/BTC_ETH_ground_truth.csv')
    #tickdata = [["BTCUSDT","ETHUSDT"]]
    negative_count = 0
    negative_time_bar = 12
    if not os.path.exists(program_file):
                os.makedirs(program_file)
    if test_ML : # trade by ML
        Net = torch.load("BTC_ETH.pkl")
        Net.eval()
        #print(Net)
        whole_year = new_dataloader.test_data()
        whole_year = torch.FloatTensor(whole_year)
        #print(whole_year)
        torch_dataset_train = Data.TensorDataset(whole_year)
        whole_test = Data.DataLoader(
                dataset=torch_dataset_train,      # torch TensorDataset format
                batch_size = 128,      # mini batch size
                shuffle = False,               
                )
        for step, (batch_x,) in enumerate(whole_test):
            #print(batch_x)
            output = Net(batch_x)               # cnn output
            _, predicted = torch.max(output, 1)
            action_choose = predicted.cpu().numpy()
            action_choose = action_choose.tolist()
            action_list.append(action_choose)
    # action_choose = predicted.cpu().numpy()
        action_list =sum(action_list, [])
        print(len(action_list))
    for date in range(1): #決定交易要從何時開始
        open_list = []
        loss_list = []
        trade_capital_list = []
        reward_list = []
        open_num_list = []
        trading_history = []
        date_list = []
        negative_pair = {}
        tickdata = tickdata.iloc[:]
        tickdata.index = np.arange(0,len(tickdata),1)  
        #table.index = np.arange(0,len(table),1)
        correlation_list = []
        #print(date)
        normal_table = table[table["model"]<4] #model 1-3 為非趨勢項 因此只根據非趨勢定態去做trade
        count = 0 
        pair_data = []
        open_list = []
        loss_list = []
        day_profit = {}
        day_capital = {}
        record_date = {}
        _day_return = []
        max_capital = 100000
        normal_table = normal_table[2000:] #如果trade 03-10 則 [:] ,ML trade 則[2000:] because labelled before 2000
        normal_table.index = np.arange(0,len(normal_table),1)
        for index, row in normal_table[:].iterrows():
            #print(index)
            _trade, _profit, _capital, _return, _trading_rule,_history = 0, 0 ,0 ,0,[0,0,0],{}
            #open, loss  = ground_truth["open"][index], ground_truth["loss"][index]
            if test_ML :
                open, loss = actions[action_list[index]][0],actions[action_list[index]][1]
            else :
                open,loss = 1.5, 10
            #open,loss = 1.5, 10#float("inf")
            #print(open,loss)
            strategy = {
                    "up_open_time" : open, #上開倉門檻
                    "down_open_time" : open, #下開倉門檻
                    "stop_loss_time" : loss, #停損門檻
                    "maxhold" : max_hold, #目前使用資金權重 所以不重要
                    "cost_gate" : trading_cost_threshold, # estimate volatility can bit cost
                    "capital" : capital, #使用資金
                    "tax_cost" : trading_cost 
                }

            del_min = row["form_del_min"]
            #
            # 
            #print(del_min)
            s1_tick = tickdata[row["S1"]][del_min:] #take rolling window 5 min price data
            s2_tick = tickdata[row["S2"]][del_min:]
            tick_date = tickdata["day"][del_min:]
            print(tick_date.iloc[0])
            s1_tick.index =np.arange(0,len(s1_tick),1)
            s2_tick.index =np.arange(0,len(s2_tick),1)
            
            day = del_min // 289 #看這交易是在第幾天 從03-01開始算

            if day not in record_date :
                record_date[day] = 1
                max_capital = 100000 #每天只有10W資金
            else :
                record_date[day] += 1
                
            
                
            if max_capital > 0  :#and correlation_check(formation_time,s1_tick,s2_tick) :
                correlation_list.append(correlation_check(formation_time,s1_tick,s2_tick))
                _trade, _profit, _capital, _return, _trading_rule, _history = trade_normal(s1_tick, s2_tick, row.to_dict(), strategy, formation_time) #trade model 1-3 ,
                #_day_return.append(_profit/2000)
                max_capital -= 2000
                date_list.append(tick_date.iloc[0])
                 
            else :
                continue

            if _profit > 0 :
                profit_count += 1
            
            if day not in day_profit : #log 每天的profit and capital
                day_profit[day] = _profit
                day_capital[day] = 2000
            else :
                day_profit[day] += _profit
                day_capital[day] += 2000
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
            table.at[index,"_return"] = _return * 100 # no use
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
            print(f'each_profit : {_profit}')
        print(date_list)   
        store_data = return_dataframe(reward_list,trading_history)  #紀錄log
        store_data.to_csv(f'{program_file}formation_{formation_time}_tradinglog.csv', mode='w',index=False)
            #if not store_data.empty:
            #profit_count += sum([p > 0 for p in table["_profit"]])
    

    return_reward = []
    reward = []
    capital_list = []
    print(len(day_profit))
    last = list(day_profit.keys())[-1]
    front = list(day_profit.keys())[0]
    for i in range(front,last+1): # to generate the datelist and draw the picture
        if i in day_profit :
            return_reward.append(day_profit[i]/day_capital[i])
            reward.append(day_profit[i])
            capital_list.append(day_capital[i])
        else :
            return_reward.append(0)
            reward.append(0)
            capital_list.append(0)
    oct = datetime.strptime(date_list[0], "%Y-%m-%d")
    date_list = [oct + timedelta(days=x) for x in range(len(return_reward))]
    date_list = [str(i).split(" ")[0] for i in date_list]
    #print(date_list)
    #print(len(date_list))
    """
    if loading_data :
        for i in range(len(day_profit)):
            store_data = return_dataframe(day_profit[0])
            store_data.to_csv(f'{program_file}{day}_profit.csv', mode='w',index=False)
    """   
    print(f'利潤 : {total_normal} and 開倉次數 : {total_num} and 開倉有賺錢的次數/開倉次數 : {profit_count/total_num}')
    print(f'開倉有賺錢次數 : {profit_count}')
    print("正常平倉 停損平倉 強迫平倉 :",total_trade[0],total_trade[1],total_trade[2])
    print("正常平倉率 :",total_trade[0]/total_num) 
    print('Time used: {} sec'.format(time.time()-start_time))
    #negative_pair ={k: v for k, v in sorted(negative_pair.items(), key=lambda item: item[1])}
    #print(negative_pair)
    count = 0
    
    if loading_data : # draw picture
        #reward,return_reward,per_reward,max_cap ,datelist = MDD.reward_calculation(program_file)
        sharp_ratio,sortino_ratio, mdd = MDD.plot_performance_with_dd( reward,return_reward,date_list,total_num,total_trade[0],profit_count/total_num,capital_list,formation_time,jump,correlation_list,_day_return )
        print(f'{total_reward:.2f}')
        win_rate = profit_count/total_num * 100
        normal_close_rate = total_trade[0]/total_num * 100
        print(f'{win_rate:.2f}%/{normal_close_rate:.2f}%')
        print(f'{total_trade[0]},{total_trade[1]},{total_trade[2]},{total_num}')
        print(f'{sharp_ratio[0]:.4f}')
        profit_per_open = total_reward / total_num
        print(f'{profit_per_open:.4f}/{max(capital_list):.2f}/{mdd:.2f}')
    
    #return total_normal, profit_count/total_num, total_trade[0]/ total_num
    return total_normal, win_rate, sharp_ratio, sortino_ratio, mdd , total_trade[0],total_trade[1] + total_trade[2]

    
if __name__ == '__main__':
    test_reward(120)
    
    
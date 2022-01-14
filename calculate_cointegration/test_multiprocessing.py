# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 09:59:18 2021

@author: Allen
"""

#import torch
#import torch.nn as nn
import numpy as np
#import new_dataloader

import trading_period_by_gate_mean_new
#import matrix_trading
import os 
import pandas as pd


import matplotlib.pyplot as plt
import time
import sys
import time
import multiprocessing as mp

ext_of_compare = "_table.csv"

path_to_python ="C:/Users/Allen/pair_trading DL2"

path_to_half = "C:/Users/Allen/pair_trading DL2/2016/2016_half/"
path_to_2017half = "./2017_halfmin/"
path_to_2018half = "./2018_halfmin/"
ext_of_half = "_half_min.csv"
model_name = '201611_only_stock_price_return'
path_to_profit = "D:/Allen/bitcoin_python/profit/"
max_posion = 1000

cost_gate_Train = False
loading_data = True
#test_period = {2016 : [path_to_2016compare],2017 :[path_to_2017compare],2018 : [path_to_2018compare],2019 : [path_to_2019compare],2020 : [path_to_2020compare]}

trading_cost = 0.002
capital = 3000000000


def return_dataframe(table,trade_capital_list,open_list,loss_list,reward_list,open_num_list):
    df = pd.DataFrame(columns=['stock1','stock2','trade_capital','open','loss','reward','open_num'])
    df['stock1'] = table.S1
    df['stock2'] = table.S2
    df['trade_capital'] = trade_capital_list
    df['open'] = open_list
    df['loss'] = loss_list
    df['reward'] = reward_list
    df['open_num'] = open_num_list
    return df

def test_reward():
    start_time = time.time()
    path_to_average = "./"+str(time)+"/averageprice/"
    ext_of_average = "_averagePrice_min.csv"
    path_to_minprice = "./"+str(time)+"/minprice/"
    ext_of_minprice = "_min_stock.csv"
    range_trading_cost_threshold = np.arange(0.0015,0.008,0.0005)
    number_of_label = len(range_trading_cost_threshold)
    total_reward = 0
    total_num = 0
    total_trade = [0,0,0]
    action_list = []
    action_list2 = []
    check = 0
    open_num_list = []
    datelist = [f.split('_')[0] for f in os.listdir('D:/Allen/bitcoin_python/test')]
    #print(datelist)
    #print(datelist[167:])
    profit_count = 0
    count = 0
    #for date in sorted(datelist[:]): #決定交易要從何時開始
    for date in range(1):
    

        table = pd.read_csv('D:/Allen/bitcoin_python/Crypto_Currency_Cointegration/tmp/20211102_table.csv')
        #mindata = pd.read_csv(path_to_average+date+ext_of_average)
        tickdata = pd.read_csv('D:/Allen/bitcoin_python/test/2021-11-02_daily_min_price.csv')
        tickdata = tickdata.iloc[:480]
        tickdata.index = np.arange(0,len(tickdata),1)  
        num = np.arange(0,len(table),1)
        #print(date)
        pair_data = []
        open, loss = 2, 1000#
        trading_cost_threshold = 0.003
        formation_time = 150
        pool = mp.Pool(2)
        open_list = []
        loss_list = []
        trade_capital_list = []
        reward_list = []
        for pair in num: #看table有幾列配對 依序讀入
            profit,opennum,trade_capital,trading = [0], 0, 0, [0,0,0]
            #print("action choose :",action_list[count_test])
            #open, loss = 2, 1000#
            #trading_cost_threshold = 0.003
            if count == 10 :
                break
            if table["model"][pair] == 1 or table["model"][pair] == 2 or table["model"][pair] == 3 :
                print(table["model"][pair])
                pair_data.append((pair,formation_time,table,tickdata,tickdata,open,open,loss,tickdata,max_posion,trading_cost,trading_cost_threshold,capital))
                #profit,opennum,trade_capital,trading  = trading_period_by_gate_mean_new.pairs\
                #( pair ,formation_time,  table , tickdata , tickdata , open ,open, loss ,tickdata,\
                # max_posion , trading_cost, trading_cost_threshold, capital )
                count += 1
                open_list.append(open)
                loss_list.append(loss)
            
                
        
            #print('交易貨幣',table.S1[pair],table.S2[pair])
            #print("利潤 :",profit)
            
            """
            profit_count += sum([p > 0 for p in profit])
                  
            #print("開倉次數 :",opennum)
 
            if opennum == 1 or opennum == 0:
                check += 1
                
            total_reward += sum(profit)
            total_num += opennum
            total_trade[0] += trading[0]
            total_trade[1] += trading[1]
            total_trade[2] += trading[2]
            """
            """
            if loading_data :
                flag = os.path.isfile(path_to_profit+str(date)+'_profit.csv')
                if not flag :
                    df = pd.DataFrame({"stock1":[table.S1[pair]],"stock2":[table.S2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[sum(profit)],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='w',index=False)
                else :
                    df = pd.DataFrame({"stock1":[table.S1[pair]],"stock2":[table.S2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[sum(profit)],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='a', header=False,index=False)  
            """
        print(table.S1)
        res = pool.starmap(trading_period_by_gate_mean_new.pairs,pair_data) 
        print(res)
        for pair in res :
            reward_list.append(pair[0])
            trade_capital_list.append(pair[2])
            open_num_list.append(pair[1])
            total_trade[0] += pair[3][0]
            total_trade[1] += pair[3][1]
            total_trade[2] += pair[3][2]
        profit_count = sum([p > 0 for p in reward_list])
        print(reward_list)
        total_num = sum(open_num_list)
        total_reward = sum(reward_list)
        store_data = return_dataframe(table[:10],trade_capital_list,open_list,loss_list,reward_list,open_num_list) 
        print(store_data)
        if loading_data :
            flag = os.path.isfile(path_to_profit+str(date)+'_profit.csv')
            if not flag :
                store_data.to_csv(path_to_profit+str(date)+'_profit.csv', mode='w',index=False)
    print("total :",check)        
            #print(count_test)
    print("利潤  and 開倉次數 and 開倉有賺錢的次數/開倉次數:",total_reward ,total_num, profit_count/total_num)
    print("開倉有賺錢次數 :",profit_count)
    print("正常平倉 停損平倉 強迫平倉 :",total_trade[0],total_trade[1],total_trade[2])
    print("正常平倉率 :",total_trade[0]/total_num)
    print('Time used: {} sec'.format(time.time()-start_time))
    """
    if loading_data :
        reward,return_reward,per_reward,max_cap ,datelist = MDD.reward_calculation(path_to_profit)
        sharp_ratio, per_sharpe_ratio, mdd = MDD.plot_performance_with_dd(path_to_profit, reward,return_reward,per_reward,datelist,total_num,total_trade[0],profit_count/total_num,max_cap )
        print(f'{total_reward:.2f}')
        win_rate = profit_count/total_num * 100
        normal_close_rate = total_trade[0]/total_num * 100
        print(f'{win_rate:.2f}%/{normal_close_rate:.2f}%')
        print(f'{total_trade[0]},{total_trade[1]},{total_trade[2]},{total_num}')
        print(f'{sharp_ratio[0]:.4f}')
        print(f'{per_sharpe_ratio[0]:.4f}')
        profit_per_open = total_reward / total_num
        print(f'{profit_per_open:.4f}/{max_cap:.2f}/{mdd:.2f}')
    """
if __name__ == '__main__':
    test_reward()
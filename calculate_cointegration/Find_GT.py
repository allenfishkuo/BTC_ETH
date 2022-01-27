import numpy as np
import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
from trade_trend import trade_down_slope, trade_up_slope, trade_normal




path_to_tick = "/home/allen/CryptoCurrency_TP"

trading_cost_threshold = 0.000
max_hold = 500
trading_cost = 0.0004
capital = 2000
max_capital = 100000

reward_list=[]

lower_bound = np.arange(0.5,8,0.05)
upper_bound = np.arange(5,25,1)

def choose_action(lower_bound,upper_bound) :
    action_list=[]
    count = 0
    l , u = 1,0
    while count < 300 :
        l = np.random.choice(lower_bound,1)
        u = np.random.choice(upper_bound,1)        
        if 1.5*l < u :
            w = np.concatenate((l,u),axis = None)
            w = list(w)
            #print(w)
            action_list.append(w)
            count +=1
    return action_list
"""    
action_list = choose_action(lower_bound, upper_bound)
actions = sorted(action_list, key = lambda s: s[0])
actions.append([15,25])
print(actions)
"""
actions = [[0.75,9],[1.0,7.5],[1.25,10],[1.5,10],[2.0,12.5],[float('inf'),float('inf')]]



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
formation_time = 120
save_path = '/home/allen/CryptoCurrency_TP/BTC_ETH_ground_truth'

path_to_compare = f'/home/allen/CryptoCurrency_TP/BTC_ETH_table'
btc_eth_table_GT= pd.DataFrame(columns = ['S1','S2','open','loss','reward','action_choose','form_del_min'])

#all_actions = check_open_loss()
for i in range(1):
        program_file = f'{save_path}/formation_{formation_time}'
        if not os.path.exists(program_file):
                os.makedirs(program_file)
        table = pd.read_csv(f'{path_to_compare}/formation_{formation_time}/BTC_ETH_formation_table.csv', dtype = dtype)
        tickdata = pd.read_csv(f'{path_to_tick}/BTC_ETH_combine.csv')
        #tickdata = tickdata.iloc[166:]
        tickdata.index = np.arange(0,len(tickdata),1)

        num = np.arange(0,len(table),1)
        #gt = gt.ravel()
       # print(gt[0][0])
        tickdata = tickdata.iloc[:]
        tickdata.index = np.arange(0,len(tickdata),1)  
        
        #print(date)
        normal_table = table[table["model"]<4]
        count = 0 
        pair_data = []
        open_list = []
        loss_list = []
        day_profit = {}
        day_capital = {}
        record_date = {}
        max_capital = 100000
        for index, row in normal_table[:].iterrows():
            _trade, _profit, _capital, _return, _trading_rule,_history = 0, 0 ,0 ,0,[0,0,0],{}
           
            action_choose = 0
            del_min = row["form_del_min"]
            s1_tick = tickdata[row["S1"]][del_min:]
            s2_tick = tickdata[row["S2"]][del_min:]
            s1_tick.index =np.arange(0,len(s1_tick),1)
            s2_tick.index =np.arange(0,len(s2_tick),1)
            
            day = del_min // 289
            
            local_profit = -0.00001
            #print(del_min)
            for open, loss in sorted(actions):
                strategy = {
                    "up_open_time" : open,
                    "down_open_time" : open,
                    "stop_loss_time" : loss,
                    "maxhold" : max_hold,
                    "cost_gate" : trading_cost_threshold,
                    "capital" : capital,
                    "tax_cost" : trading_cost
                }
                _trade, _profit, _capital, _return, _trading_rule, _history = trade_normal(s1_tick, s2_tick, row.to_dict(), strategy, formation_time)
                if _profit >= local_profit :
                    open_time = open
                    loss_time  = loss
                    local_profit = _profit
                    action_ = action_choose
                action_choose += 1
                if local_profit <= 0:
                    action_ = 5
                    open_time = actions[5][0]
                    loss_time  = actions[5][1]
            
            df = pd.DataFrame({"S1":[row["S1"]],"S2":[row["S2"]],"open":[open_time],"loss":[loss_time],"reward":[local_profit],"action_choose":[action_],'form_del_min':[del_min]})
            #print(df)
            btc_eth_table_GT = pd.concat([btc_eth_table_GT,df],ignore_index=True)
        btc_eth_table_GT.to_csv( f"{program_file}/BTC_ETH_ground_truth.csv" ,index = False)
                

                
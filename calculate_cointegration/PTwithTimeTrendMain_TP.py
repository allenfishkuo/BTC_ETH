#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:32:16 2022

@author: kuoweilun
"""

import pandas as pd
import numpy as np
#import mt 
import matplotlib.pyplot as plt
import PTwithTimeTrend_AllStock as ptm
import time
import os
import test
form_del_min = 0
#參數
#indataNum=150 #建模期間
Cost= 0.0015     #交易成本
CostS=0.0015   #交易門檻
Os=1.5          #開倉倍數(與喬登對齊)
Fs= 10000       #強迫平倉倍數(無限大=沒有強迫平倉) (與喬登對齊)
Cs = 0
MaxVolume=5     #最大張數限制
OpenDrop=16     #開盤捨棄
Min_c_p= 300    #最小收斂點門檻(大於indataNum=沒有設)(與喬登對齊,無收斂點概念)
Max_t_p=190     #最大開盤時間(190        (與喬登對齊)

##set the date
#years = ['2015','2016','2017','2018']
years = ['2021']
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]

days = ["01","02","03","04","05","06","07","08","09","10",
    "11","12","13","14","15","16","17","18","19","20",
  "21","22","23","24","25","26","27","28","29","30","31"]


#if False:
#    test_csv_name = '_half_min'
#else:
#   test_csv_name = '_averagePrice_min'
#or indataNum in range(100,110,10):z

save_path = '/home/allen/CryptoCurrency_TP/BTC_ETH_table' #save路徑
test_path = '/home/allen/CryptoCurrency_TP/'              #讀取2021-03~ 2021-10 的5分K csv檔
btc_eth_table_combine = pd.DataFrame(columns = ['S1','S2','VECM(q)','mu','Johansen_slope','stdev','model','w1','w2','form_del_min']) # calculate cointegration之後算產生的餐數 w1 w2為共整合係數（資金權重）
for t in range(1,2): 
    indataNum = 120
    for form_del_min in range(0,90000,1):   #form_del_min為rolling window刪去前面多少個5分k
            program_file = f'{save_path}/formation_{indataNum}'
            if not os.path.exists(program_file):
                os.makedirs(program_file)

            try:
                        #print("iin")
                        test_data = pd.read_csv(test_path+'/'+"BTC_ETH_combine.csv",index_col=False)
                        #test_data = pd.read_csv(test_path+"BTC_ETH_combine.csv",index_col=False)
                        test_data = test_data[["BTCUSDT","ETHUSDT"]]
                        #test_data = test_data[["BTCUSDT","ETHUSDT","AVAXUSDT","BNBUSDT","LUNAUSDT"]]
                        #print(test_data)
                        test_data = test_data.iloc[form_del_min:,:]
                        if len(test_data) < 240 :
                            break
                        test_data = test_data.reset_index(drop=True)
                        print(test_data)
                        #test_data = test_data[['2379','6269']]
                        dailytable = ptm.formation_table(test_data,indataNum,CostS,Cost,Os,Fs,MaxVolume,OpenDrop,Min_c_p, Max_t_p)         
                        btc_eth_table = pd.DataFrame(dailytable,columns = ['S1','S2','VECM(q)','mu','Johansen_slope','stdev','model','w1','w2'])
                        
                        if not btc_eth_table.empty :
                            btc_eth_table["form_del_min"] = form_del_min                           
                            btc_eth_table_combine = pd.concat([btc_eth_table_combine,btc_eth_table],ignore_index=True)
                            
            except:
                        
                        continue
    btc_eth_table_combine.to_csv( f"{program_file}/BTC_ETH_formation_table_p10.csv" ,index = False)



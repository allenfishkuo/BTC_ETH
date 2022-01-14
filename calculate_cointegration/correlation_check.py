#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 18:27:55 2022

@author: kuoweilun
"""


import numpy as np
#import new_dataloader
#import MDD
#from trade_trend import trade_down_slope, trade_up_slope, trade_normal
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



def correlation_check(formation_time, s1_tick, s2_tick):

    #path_to_compare = f'/home/allen/CryptoCurrency_PT-main/table_30_5min_formation_greedy/formation_{formation_time}'
    #path_to_compare = f'/Users/kuoweilun/CryptoCurrency_PT-main/Crypto_Currency_Cointegration/formation_{formation_time}_5min_tens/2021'
    #datelist = [f.split('_')[0] for f in os.listdir(f'{path_to_compare}/')]
    #print(datelist)
    #print(datelist[167:])
    """
    #for date in sorted(datelist[:2]): #決定交易要從何時開始
        print(date)
        #table = pd.read_csv(f'{path_to_compare}/{date}_table.csv', dtype = dtype)
        tickdata = pd.read_csv(f'{path_to_tick}/{date[:4]}-{date[4:6]}-{date[6:8]}_daily_min_price.csv')
        tickdata = tickdata[:120]
        tickdata["AVAXUSDT"] , tickdata["LUNAUSDT"] = tickdata["AVAXUSDT"].pct_change() , tickdata["LUNAUSDT"].pct_change()
        plt.scatter(tickdata["AVAXUSDT"] , tickdata["LUNAUSDT"])
        plt.show()
        correlation = tickdata["AVAXUSDT"].corr(tickdata["LUNAUSDT"])
        print("Correlation :", correlation)
    """
    s1_tick , s2_tick = s1_tick[:formation_time], s2_tick[:formation_time]
    s1_tick , s2_tick = s1_tick.pct_change() , s2_tick.pct_change()
    #plt.scatter(s1_tick , s2_tick)
    #plt.show()
    correlation = s1_tick.corr(s2_tick)
    #print("Correlation :", correlation)
    if correlation > 0.5 :
        return True
    else : return False

if __name__ == "__main__":
    correlation_check(1,2,3)
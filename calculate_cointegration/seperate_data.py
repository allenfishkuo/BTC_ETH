# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:28:42 2021

@author: allen
"""

import pandas as pd 
import numpy as np
import datetime as dt
TXF = pd.read_csv("FXF1-Minute-Trade.csv")
print(TXF)
tmp = TXF["Date"][0]
print(tmp)
store_data = pd.DataFrame()
date_time = []
for i in range(len(TXF)):
    
    date = TXF["Date"][i]
    if date not in date_time :
        date_time.append(date)
print(date_time)
    
groups = TXF.groupby("Date")
for date in date_time :
    TXF_daily = groups.get_group(date)
    date_list = date.split("/")
    input_date = dt.datetime(int(date_list[0]),int(date_list[1]),int(date_list[2])).strftime("%Y%m%d")
    print(input_date)
    TXF_daily.to_csv("./FXF_daily_file/{}_minprice.csv".format(input_date),index=False)

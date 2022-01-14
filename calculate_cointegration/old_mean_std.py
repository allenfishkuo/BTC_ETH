# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:34:56 2021

@author: MAI
"""
import numpy as np
import pandas as pd
import os

years = ['2018']

months = ["01","02","03","04","05","06","07","08","09","10",
     "11","12"]
#months = ["01"]
days = ["01","02","03","04","05","06","07","08","09","10",
     "11","12","13","14","15","16","17","18","19","20",
       "21","22","23","24","25","26","27","28","29","30",
     "31"]
#days = ["06","07"]
## set the table path
#tablePath = r'D:\HSINHUA\formation_table_20201202\ptstrendPythonCode\allstock_formation_table_adf\2015'
test_csv_name = '_averagePrice_min'

# construct the table, open&close timing, structural break&stopLoss timing 
for year in years:
    for month in months:
        print("Now import: ",month,"-th month")
        for day in days:
            
            tablePath = r'D:\HSINHUA\formation_table_20201202\ptstrendPythonCode\all_model3_table'
#          allstock_formation_table_adf\{}'.format(year)
            #tablePath = r'D:\HSINHUA\formation_table_20201202\ptstrendPythonCode\trace_test_based'
            #tablePath = r'D:\HSINHUA\formation_table_20201202\ptstrendPythonCode\all_model3_table'
            tick_trigger_dataPath = r'D:\JordanMaiThesis\Thesis\data\tick(secs)\{}'.format(year)
            origin_trigger_dataPath = r'D:\JordanMaiThesis\Thesis\data\min_data\{}\minprice'.format(year)
            test_dataPath = r'D:\JordanMaiThesis\Thesis\data\min_data\{}\averageprice'.format(year)
            try:
                # Read data 
                table = pd.read_csv(os.path.join(tablePath,'{}{}{}_formationtable.csv'.format(year,month,day)))
#                table = np.array(table)
#                table = table[:,1:10]
#                table = pd.DataFrame(table,columns = ['S1','S2','VECM(q)','Johansen_intercept','Johansen_slope','Johansen_std','model','w1','w2'])
                table['S1'].astype('int')
                table['S2'].astype('int')
                # test data: min average data or half min average data
                test_data = pd.read_csv(os.path.join(test_dataPath,'{}{}{}{}.csv'.format(year,month,day,test_csv_name)))
                test_data_iterated = pd.read_csv(os.path.join(test_dataPath,'{}{}{}{}.csv'.format(year,month,day,test_csv_name)))
                test_data = test_data.iloc[16:,:]
                test_data = test_data.reset_index(drop=True)
                test_data_iterated = test_data_iterated.reset_index(drop=True)
            except:
                
                continue
            stock1_name = table.S1.astype('int',copy=False)
            stock1_name = stock1_name.astype('str',copy=False)
            stock2_name = table.S2.astype('int',copy=False)
            stock2_name = stock2_name.astype('str',copy=False)
            test_stock1 = np.array(test_data[stock1_name].T)
            test_stock2 = np.array(test_data[stock2_name].T)
            w1 = np.expand_dims(np.array(table.w1),axis=1)
            w2 = np.expand_dims(np.array(table.w2),axis=1)            
            test_spread = w1 * np.log(test_stock1) + w2 * np.log(test_stock2)
            mu =np.zeros([len(test_spread),1])
            stdev =np.zeros([len(test_spread),1])
            for i in range(len(test_spread)):           
                mu[i,0] = np.mean(test_spread[i,0:150])
                stdev[i,0] = np.std(test_spread[i,0:150])
            table['mu'] = mu
            table['stdev'] =stdev
            save_path = r'D:\HSINHUA\formation_table_20201202\ptstrendPythonCode\all_model3_table_with_old_mean\{}'.format(year)
            table.to_csv( ''.join([ save_path ,'{}{}{}_formationtable.csv'.format(year,month,day) ]))
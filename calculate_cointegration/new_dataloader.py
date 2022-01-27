import numpy as np
import pandas as pd
import os 
from sklearn import preprocessing
import matplotlib.pyplot as pltr
from sklearn.preprocessing import StandardScaler 

min_max_scaler = preprocessing.MinMaxScaler()

#no_half =["2231","8454","6285","2313","2867","1702","3662","1536","9938","2847","6456"]
min_max_scaler = preprocessing.MinMaxScaler()

SS = StandardScaler()
read_coverge_time = False
normalize_spread = False
input_of_three = False
Use_avg = False
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
def read_data(formation_time = 120):
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    path_to_compare = f'/home/allen/CryptoCurrency_TP/BTC_ETH_table'
    path_to_ground_truth = "/home/allen/CryptoCurrency_TP/BTC_ETH_ground_truth/"
    path_to_tick = "/home/allen/CryptoCurrency_TP"


    
    for i in range(1):

            table = pd.read_csv(f'{path_to_compare}/formation_{formation_time}/BTC_ETH_formation_table.csv', dtype = dtype)
            tickdata = pd.read_csv(f'{path_to_tick}/BTC_ETH_combine.csv')
            ground_truth = pd.read_csv(f'{path_to_ground_truth}/formation_{formation_time}/BTC_ETH_ground_truth.csv',usecols=["action_choose"])
            #gt = pd.read_csv(dic_choose[year]+date+ext_of_groundtruth,usecols=["action choose"])
            gt = ground_truth.values
            #print(date)
            #print(count)
            tickdata.index = np.arange(0,len(tickdata),1)  
        #table.index = np.arange(0,len(table),1)
            normal_table = table[table["model"]<4]
            normal_table.index = np.arange(0,len(normal_table),1)
            for index, row in normal_table[:].iterrows():
                del_min = row["form_del_min"]

                s1_tick = tickdata[row["S1"]][del_min:]
                s2_tick = tickdata[row["S2"]][del_min:]
                s1_tick = s1_tick[:formation_time]
                s2_tick = s2_tick[:formation_time]
                spread = row["w1"] * np.log(s1_tick) + row["w2"] * np.log(s2_tick)
                spread = spread.values
                spread = preprocessing.scale(spread)
                #print(spread)
                new_spread = np.zeros((1,512))
                new_spread[0,196:316] = spread        
                number = gt[index]
                train_data.append(new_spread)
                train_label.append(number)                                   
    train_data = np.asarray(train_data)
  
    train_label = np.asarray(train_label)
    train_label = train_label.flatten()
    #print(train_label)
    print(train_label.shape)
    print(train_data.shape)   
  
    print(np.any(np.isnan(train_data)))
    return train_data, train_label, test_data, test_label



def test_data(formation_time = 120):

    whole_year = []
    path_to_compare = f'/home/allen/CryptoCurrency_TP/BTC_ETH_table'
    path_to_ground_truth = "/home/allen/CryptoCurrency_TP/BTC_ETH_ground_truth/"
    path_to_tick = "/home/allen/CryptoCurrency_TP"
    for i in range(1):

            table = pd.read_csv(f'{path_to_compare}/formation_{formation_time}/BTC_ETH_formation_table.csv', dtype = dtype)
            tickdata = pd.read_csv(f'{path_to_tick}/BTC_ETH_combine.csv')
            ground_truth = pd.read_csv(f'{path_to_ground_truth}/formation_{formation_time}/BTC_ETH_ground_truth.csv',usecols=["action_choose"])
            #gt = pd.read_csv(dic_choose[year]+date+ext_of_groundtruth,usecols=["action choose"])
            gt = ground_truth.values
            #print(date)
            #print(count)
            tickdata.index = np.arange(0,len(tickdata),1)  
            normal_table = table[table["model"]<4]
            normal_table.index = np.arange(0,len(normal_table),1)
            for index, row in normal_table[2000:].iterrows():
                del_min = row["form_del_min"]

                s1_tick = tickdata[row["S1"]][del_min:]
                s2_tick = tickdata[row["S2"]][del_min:]
                s1_tick = s1_tick[:formation_time]
                s2_tick = s2_tick[:formation_time]
                spread = row["w1"] * np.log(s1_tick) + row["w2"] * np.log(s2_tick)
                spread = spread.values
                spread = preprocessing.scale(spread)
                #print(spread)
                new_spread = np.zeros((1,512))
                new_spread[0,196:316] = spread                       
                whole_year.append(new_spread )                                
    whole_year = np.asarray(whole_year)
    print("whole_year :",whole_year.shape)

    return whole_year

if __name__ == '__main__':    
    #read_data()
    test_data()
    #val_data()
    #find_threshold_data()
    #read_actions()
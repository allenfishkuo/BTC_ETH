# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:49:47 2020

@author: allen
"""

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
#from Predict_Client import send_request
from scipy.stats import skew
from cost import tax , slip 
from integer import num_weight
from vecm import rank
from MTSA import fore_chow , spread_chow , order_select
#import tensorflow
#from keras.models import load_model
import pandas as pd
import numpy as np
import numba as nb



plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']='sans-serif' 
plt.rcParams['axes.unicode_minus'] = False
# 標準差倍數當作停損門檻(滑價＋交易稅)-------------------------------------------------------------------------------
path_to_image = "./tmp2/"
def pairs(pair, formate_time, table, min_data, tick_data, up_open_time, down_open_time, stop_loss_time, day1, maxi,
          tax_cost, cost_gate, capital,  dump=True, allow_multiple_trades=False):

    trade_capital = 0
    trade_capital_list = list()
    cpA,cpB = 0,0
    trading =[0,0,0]
    '''
    if table.model_type[pair] == 'model1':
        model_name = 'H2'
    elif table.model_type[pair] == 'model2':
        model_name = 'H1*'
    else:
        model_name = 'H1'
    '''
    use_fore_lag5 = False
    use_adf = False
    # 波動太小的配對不開倉
    if up_open_time *table['stdev'][pair] < cost_gate:
        trading_profit = 0
        trade = 0
        local_profit = 0
        local_open_num = 0
        local_rt = 0
        local_std = 0
        local_skew = 0
        local_timetrend = 0
        position = 0
        return local_profit, local_open_num, trade_capital,trading
    min_price = day1
    # min_price = min_price.dropna(axis = 1)
    # min_price.index  = np.arange(0,len(min_price),1)
    # num = np.arange(0,len(table),1)
    t = formate_time  # formate time
    stock1_seq = min_price[str(table.S1[pair])].loc[0:t]
    stock2_seq = min_price[str(table.S2[pair])].loc[0:t]
    # z = ( np.vstack( [stock1_seq , stock2_seq] ).T )
    # p = order_select(z,5)
    local_open_num = []
    local_profit = []
    local_rt = []
    local_std = []
    local_skew = []
    local_timetrend = []
    # for pair in num:
    # spread = table.w1[pair] * np.log(min_data[ table.stock1[pair] ]) + table.w2[pair] * np.log(min_data[ table.stock2[pair] ])
    spread = table.w1[pair] * np.log(tick_data[str(table.S1[pair])]) + table.w2[pair] * np.log(
        tick_data[str(table.S2[pair])])
    #print("spread length :",len(spread))
    up_open = table['mu'][pair] + table['stdev'][pair] * up_open_time  # 上開倉門檻
    down_open = table['mu'][pair] - table['stdev'][pair] * down_open_time  # 下開倉門檻
    stop_loss = table['stdev'][pair] * stop_loss_time  # 停損門檻
    close = table['mu'][pair]  # 平倉(均值)
    #M = round(1 / table.zcr[pair])  # 平均持有s時間
    trade = 0  # 計算開倉次數
    break_point = 0  # 計算累積斷裂點
    # break_CNN = 0
    # discount = 1
    position = 0  # 持倉狀態，1:多倉，0:無倉，-1:空倉，-2：強制平倉
    pos = [0]
    stock1_profit = []
    stock2_profit = []
    num = 0
    threshold_time = len(spread) - 2
    """
    if allow_multiple_trades:
        threshold_time = len(spread) - 2
    else:
        threshold_time = 40
    """
    for i in range(t, len(spread) - 2):
        stock1_seq = min_price[str(table.S1[pair])].loc[0:t + i]
        stock2_seq = min_price[str(table.S2[pair])].loc[0:t + i]
        # z = ( np.vstack( [stock1_seq , stock2_seq] ).T )
        # r = rank( pd.DataFrame(z) , model_name , p )
        # if position == 0 and len(spread) - i > M:  # 之前無開倉
        if position == 0 and i != len(spread) - 3 and i < threshold_time:  # 之前無開倉
        #if position == 0 and i != len(spread) - 3 :
            if  spread[i] < (close + stop_loss) and  spread[i] > up_open :  # 碰到上開倉門檻且小於上停損門檻
                # 資金權重轉股票張數，並整數化
                w1, w2 = num_weight(table.w1[pair], table.w2[pair], tick_data[str(table.S1[pair])][i],
                                    tick_data[str(table.S2[pair])][i], maxi, capital)
                print(str(table.S1[pair]),str(table.S2[pair]),w1,w2,tick_data[str(table.S1[pair])][(i)],tick_data[str(table.S2[pair])][(i)])

                spread1 = w1 * np.log(stock1_seq) + w2 * np.log(stock2_seq)
                if use_adf and adfuller(spread1, regression='c')[1] > 0.05:  # spread平穩才開倉
                    position = 0
                    stock1_payoff = 0
                    stock2_payoff = 0
                else:
                    position = -1
                    #stock1_payoff = w1 * slip(tick_data[str(table.S1[pair])][(i)], table.w1[pair])
                    #stock2_payoff = w2 * slip(tick_data[str(table.S2[pair])][(i)], table.w2[pair])
                    stock1_payoff = w1 * tick_data[str(table.S1[pair])][(i)]
                    stock2_payoff = w2 * tick_data[str(table.S2[pair])][(i)]
                    #stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                    stock1_payoff, stock2_payoff = tax(stock1_payoff,tax_cost), tax(stock2_payoff,tax_cost) 
                    cpA,cpB = stock1_payoff, stock2_payoff
                    if cpA > 0 and cpB > 0:
                        trade_capital = abs(cpA)+abs(cpB)
                    elif cpA > 0 and cpB < 0 :
                        trade_capital = abs(cpA)+0.9*abs(cpB)
                    elif cpA < 0 and cpB > 0 :
                        trade_capital = 0.9*abs(cpA)+abs(cpB)
                    elif cpA < 0 and cpB < 0 :
                        trade_capital = 0.9*abs(cpA)+0.9*abs(cpB)
                    tmp = 0
                    if w1 > 0:
                        tmp += 0.9 * tick_data[str(table.S1[pair])][i] * w1
                    else:
                        tmp += abs(tick_data[str(table.S1[pair])][i] * w1)
                    if w2 > 0:
                        tmp += 0.9 * tick_data[str(table.S2[pair])][i] * w2
                    else:
                        tmp += abs(tick_data[str(table.S2[pair])][i] * w2)
                    # print(table.stock1[pair], table.stock2[pair], trade_capital, tmp)
                    trade_capital_list.append(trade_capital)
                    # down_open = table.mu[pair] - table.stdev[pair] * close_time
                    trade = trade + 1
            elif spread[i] > (close - stop_loss) and spread[i] < down_open:  # 碰到下開倉門檻且大於下停損門檻
                # 資金權重轉股票張數，並整數化
                w1, w2 = num_weight(table.w1[pair], table.w2[pair], tick_data[str(table.S1[pair])][(i)],
                                    tick_data[str(table.S2[pair])][(i)], maxi, capital)
                # print(str(table.stock1[pair]), str(table.stock2[pair]),w1,w2,tick_data[str(table.stock1[pair])][(i)],tick_data[str(table.stock2[pair])][(i)])

                spread1 = w1 * np.log(stock1_seq) + w2 * np.log(stock2_seq)
                if use_adf and adfuller(spread1, regression='c')[1] > 0.05:  # spread平穩才開倉
                    position = 0
                    stock1_payoff = 0
                    stock2_payoff = 0
                else:
                    position = 1
                    #stock1_payoff = -w1 * slip(tick_data[str(table.S1[pair])][(i)], -table.w1[pair])
                    #stock2_payoff = -w2 * slip(tick_data[str(table.S2[pair])][(i)], -table.w2[pair])
                    stock1_payoff = -w1 * tick_data[str(table.S1[pair])][(i)]
                    stock2_payoff = -w2 * tick_data[str(table.S2[pair])][(i)]
                    #stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                    stock1_payoff, stock2_payoff = tax(stock1_payoff,tax_cost), tax(stock2_payoff,tax_cost) 
                    cpA,cpB = stock1_payoff,stock2_payoff
                    if cpA > 0 and cpB > 0:
                        trade_capital = abs(cpA)+abs(cpB)
                    elif cpA > 0 and cpB < 0 :
                        trade_capital = abs(cpA)+0.9*abs(cpB)
                    elif cpA < 0 and cpB > 0 :
                        trade_capital = 0.9*abs(cpA)+abs(cpB)
                    elif cpA < 0 and cpB < 0 :
                        trade_capital = 0.9*abs(cpA)+0.9*abs(cpB)
                    tmp = 0
                    if w1 > 0:
                        tmp += tick_data[str(table.S1[pair])][i] * w1
                    else:
                        tmp += 0.9 * abs(tick_data[str(table.S1[pair])][i] * w1)
                    if w2 > 0:
                        tmp += tick_data[str(table.S2[pair])][i] * w2
                    else:
                        tmp += 0.9 * abs(tick_data[str(table.S2[pair])][i] * w2)
                    # print(table.stock1[pair], table.stock2[pair], trade_capital, tmp)
                    trade_capital_list.append(trade_capital)
                    # up_open = table.mu[pair] + table.stdev[pair] * close_time
                    trade = trade + 1
            else:
                position = 0
                stock1_payoff = 0
                stock2_payoff = 0
        elif position == -1:  # 之前有開空倉，平空倉
            spread1 = table.w1[pair] * np.log(stock1_seq) + table.w2[pair] * np.log(stock2_seq)
            #num = fore_chow(min_price[str(table.stock1[pair])].loc[0:t], min_price[str(table.stock2[pair])].loc[0:t], stock1_seq,
                            #stock2_seq, table.model_type[pair])  #做forlag5
            if use_fore_lag5:
                if num == 0:
                    break_point = 0
                else:  # num == 1
                    break_point = break_point + num
            '''   
            temp=np.array(spread1[i:t+i]).reshape(1,200,1)
            num = model.predict_classes(temp)
            if num == 0 or num == 4:
                break_CNN = 0
            else: # num == 1
                num = 1
                break_CNN = break_CNN + num
            '''
            if i == (len(spread) - 3):  # 回測結束，強制平倉
                position = -4
                #stock1_payoff = -w1 * slip(tick_data[str(table.S1[pair])][i], -table.w1[pair])
                # = -w2 * slip(tick_data[str(table.S2[pair])][i], -table.w2[pair])
                stock1_payoff = -w1 * tick_data[str(table.S1[pair])][i]
                stock2_payoff = -w2 * tick_data[str(table.S2[pair])][i]
                #stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                stock1_payoff, stock2_payoff = tax(stock1_payoff,tax_cost), tax(stock2_payoff,tax_cost) 
                trading[2] += 1
            elif (spread[i] - close) < 0:  # 空倉碰到下開倉門檻即平倉
                position = 666  # 平倉
                #stock1_payoff = -w1 * slip(tick_data[str(table.S1[pair])][i], -table.w1[pair])
                #stock2_payoff = -w2 * slip(tick_data[str(table.S2[pair])][i], -table.w2[pair])
                stock1_payoff = -w1 * tick_data[str(table.S1[pair])][i]
                stock2_payoff = -w2 * tick_data[str(table.S2[pair])][i]
                #stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                stock1_payoff, stock2_payoff = tax(stock1_payoff,tax_cost), tax(stock2_payoff,tax_cost) 
                # print(str(table.stock1[pair]),str(table.stock2[pair]),w1,w2,tick_data[str(table.stock1[pair])][(i)],tick_data[str(table.stock2[pair])][(i)])

                trading[0]+=1
                # down_open = table.mu[pair] - table.stdev[pair] * open_time
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif spread[i] > (close + stop_loss):  # 空倉碰到上停損門檻即平倉停損
                position = -2  # 碰到停損門檻，強制平倉
                #stock1_payoff = -w1 * slip(tick_data[str(table.S1[pair])][(i)], -table.w1[pair])
                #stock2_payoff = -w2 * slip(tick_data[str(table.S2[pair])][(i)], -table.w2[pair])
                stock1_payoff = -w1 * tick_data[str(table.S1[pair])][(i)]
                stock2_payoff = -w2 * tick_data[str(table.S2[pair])][(i)]
                #stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                stock1_payoff, stock2_payoff = tax(stock1_payoff,tax_cost), tax(stock2_payoff,tax_cost) 
                trading[1]+=1
                # 每次交易報酬做累加(最後除以交易次數做平均)

            elif break_point == 5:
                position = -3  # 結構性斷裂，強制平倉
                #stock1_payoff = -w1 * slip(tick_data[str(table.S1[pair])][(i + 1)], -table.w1[pair])
                #stock2_payoff = -w2 * slip(tick_data[str(table.S2[pair])][(i + 1)], -table.w2[pair])
                stock1_payoff = -w1 * tick_data[str(table.S1[pair])][(i + 1)]
                stock2_payoff = -w2 * tick_data[str(table.S2[pair])][(i + 1)]
                stock1_payoff, stock2_payoff = tax(stock1_payoff,tax_cost), tax(stock2_payoff,tax_cost) 
                #stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2]+=1

            else:
                position = -1
                stock1_payoff = 0
                stock2_payoff = 0
        elif position == 1:  # 之前有開多倉，平多倉
            spread1 = table.w1[pair] * np.log(stock1_seq) + table.w2[pair] * np.log(stock2_seq)
            #num = fore_chow(min_price[str(table.stock1[pair])].loc[0:t], min_price[str(table.stock2[pair])].loc[0:t], stock1_seq,
                           # stock2_seq, table.model_type[pair]) #做forlag5
            if use_fore_lag5:
                if num == 0:
                    break_point = 0
                else:  # num == 1
                    break_point = break_point + num
            '''
            temp=np.array(spread1[i:t+i]).reshape(1,200,1)
            num = model.predict_classes(temp)
            if num == 0 or num == 4:
                break_CNN = 0
            else: # num == 1
                num = 1
                break_CNN = break_CNN + num
            '''
            if i == (len(spread) - 3):  # 回測結束，強制平倉
                position = -4
                stock1_payoff = w1 * tick_data[str(table.S1[pair])][i]
                stock2_payoff = w2 * tick_data[str(table.S2[pair])][i]
                #stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                stock1_payoff, stock2_payoff = tax(stock1_payoff,tax_cost), tax(stock2_payoff,tax_cost) 
                trading[2] += 1
                # print(str(table.stock1[pair]), str(table.stock2[pair]), w1, w2, tick_data[str(table.stock1[pair])][(i)],
                #      tick_data[str(table.stock2[pair])][(i)])
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif (spread[i] - close) > 0:
                position = 666  # 平倉
                stock1_payoff = w1 * tick_data[str(table.S1[pair])][(i)]
                stock2_payoff = w2 * tick_data[str(table.S2[pair])][(i)]
                stock1_payoff, stock2_payoff = tax(stock1_payoff,tax_cost), tax(stock2_payoff,tax_cost) 
                #stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                # print(str(table.stock1[pair]),str(table.stock2[pair]),w1,w2,tick_data[str(table.stock1[pair])][(i)],tick_data[str(table.stock2[pair])][(i)])

                trading[0]+=1
                # up_open = table.mu[pair] + table.stdev[pair] * open_time
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif spread[i] < (close - stop_loss):
                position = -2  # 碰到停損門檻，強制平倉
                stock1_payoff = w1 * tick_data[str(table.S1[pair])][(i)]
                stock2_payoff = w2 * tick_data[str(table.S2[pair])][(i)]
                #stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                stock1_payoff, stock2_payoff = tax(stock1_payoff,tax_cost), tax(stock2_payoff,tax_cost) 
                trading[1]+=1
   
            elif break_point == 5:
                position = -3  # 結構性斷裂，強制平倉
                stock1_payoff = w1 * tick_data[str(table.S1[pair])][(i)]
                stock2_payoff = w2 * tick_data[str(table.S2[pair])][(i)]
                stock1_payoff, stock2_payoff = tax(stock1_payoff,tax_cost), tax(stock2_payoff,tax_cost) 
                #stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2] += 1

            else:
                position = 1
                stock1_payoff = 0
                stock2_payoff = 0
        else:
            # -4: 強迫平倉 -3: 結構性斷裂平倉(for lag 5) -2:停損 666:正常平倉
            if position == -2 or position == -3 or position == -4 or position == 666:
                stock1_payoff = 0
                stock2_payoff = 0
            else:
                position = 0  # 剩下時間少於預期開倉時間，則不開倉，避免損失
                stock1_payoff = 0
                stock2_payoff = 0
        pos.append(position)
        stock1_profit.append(stock1_payoff)
        stock2_profit.append(stock2_payoff)
        if allow_multiple_trades and position in [-2, -3, -4, 666]:
            position = 0
    trading_profit = [s1_profit + s2_profit for s1_profit, s2_profit in zip(stock1_profit, stock2_profit)]
    trading_profit = [x for x in trading_profit if x!=0]
    local_profit = []
    for i in range(0, len(trading_profit), 2):
        local_profit.append(trading_profit[i] + trading_profit[i+1])
    local_open_num = trade
    # local_profit = trading_profit
    '''
    if trading_profit != 0 and position == 666:
        position = 666
    '''
    if len(trade_capital_list) > 0:
        trade_capital = max(trade_capital_list)
    else:
        trade_capital = 0
    # -4: 強迫平倉 -3: 結構性斷裂平倉(for lag 5) -2:停損 666:正常平倉
    if True :# (position == -2 or position == -3 or position == -4) and local_profit <= -10 :
        if dump and sum(trading) > 0:
            plot_spread( table.S1[pair], table.S2[pair], spread, up_open, down_open, stop_loss,
                        close, local_profit, pos, position, up_open_time, down_open_time, stop_loss_time)
    return   sum(local_profit), local_open_num, trade_capital, trading
            
            
    # table.stock1 , table.stock2 , local_profit , local_open_num , local_rt , local_std , local_skew , local_timetrend
    # #, 0


def plot_spread( stock1, stock2, spread, up_open, down_open, stop_loss, close, local_profit, pos, position,
                up_open_time, down_open_time, stop_loss_time):
    #print(type(up_open_time),type(stock1))
    plt.figure(figsize=(20, 10))
    plt.plot(spread)
    plt.vlines(150,down_open,up_open)
    plt.hlines(up_open, 0, len(spread) - 1, 'b')
    plt.hlines(down_open, 0, len(spread) - 1, 'b')
    #plt.hlines(close + stop_loss, 0, len(spread) - 1, 'r')
    #plt.hlines(close - stop_loss, 0, len(spread) - 1, 'r')
    plt.hlines(close, 0, len(spread) - 1, 'g')
    print(spread)
    spread = spread[150:]
    spread = spread.reset_index(drop = True)
    print(spread)
    for x in range(1, len(pos)):
        if pos[x] != pos[x - 1] and pos[x] != 0:
            plt.scatter(x-1+150, spread[x-1], color='', edgecolors='r', marker='o')
    plt.title( ' s1:' + str(stock1) + ' s2:' + str(stock2) + ' up open threshold:' + str(up_open_time) + ' down open threshold:'
              + str(down_open_time) + ' stop threshold:' + str(stop_loss_time))
    if position == 666:
        plt.xlabel('profit:' + str(local_profit) + ' normal close profit')
    elif position == -2:
        plt.xlabel('profit:' + str(local_profit) + ' stop close profit')
    elif position == -3:
        plt.xlabel('profit:' + str(local_profit) + ' fore_lag5')
    elif position == -4:
        plt.xlabel('profit:' + str(local_profit) + ' times up，forced close profit')
    else:
        plt.xlabel('Total open count:' + str(len(local_profit)) + ' ' + ', total profit: ' + str(sum(local_profit)))
    plt.savefig(path_to_image+ '_' + str(stock1) + '_' + str(stock2) + '_' + str(
        up_open_time) + '_' + str(down_open_time) + '_' + str(stop_loss_time) + '.png')
    plt.close('all')
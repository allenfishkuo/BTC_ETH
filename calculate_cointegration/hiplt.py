from matplotlib.pyplot import show
import pandas as pd
import numpy as np
import hiplot as hip
import test_hole_year
def draw_hiplot():
    
    formation_period_list = []
    PNL_list = []
    win_rate_list = []
    Sharpe_ratio_list = []
    Sortin_ratio_list = []
    MDD_list = []
    win_trade_list = []
    loss_trade_list = []
    for test_period in range(100,205,5):
        PNL, winrate ,sharp_ratio,sortino_ratio, mdd ,win_trade, loss_trade = test_hole_year.test_reward(test_period)
        sharp_ratio = sharp_ratio.values.tolist()[0]
        sortino_ratio = sortino_ratio.values.tolist()[0]
        formation_period_list.append(test_period)
        PNL_list.append(PNL)
        win_rate_list.append(winrate)
        Sharpe_ratio_list.append(sharp_ratio)
        Sortin_ratio_list.append(sortino_ratio)
        MDD_list.append(mdd)
        win_trade_list.append(win_trade)
        loss_trade_list.append(loss_trade)

    show_hiplot = pd.DataFrame(columns = ["formation period","PNL","Win rate","Sharpe ratio","Sortino ratio","MDD","# of win trades","# of loss trades"])
    show_hiplot["formation period"] = formation_period_list
    show_hiplot["PNL"] = PNL_list
    show_hiplot["Win rate"] = win_rate_list
    show_hiplot["Sharpe ratio"] = Sharpe_ratio_list
    show_hiplot["Sortino ratio"] = Sortin_ratio_list
    show_hiplot["MDD"] = MDD_list
    show_hiplot["# of win trades"] = win_trade_list
    show_hiplot["# of loss trades"] = loss_trade_list
    show_hiplot.to_csv("BTC_ETH_hiplot.csv",index = False)
    
    #iris_hiplot = hip.Experiment.from_csv('BTC_ETH_hiplot.csv')
    #iris_hiplot.display()
    

if __name__ == "__main__":
    draw_hiplot()
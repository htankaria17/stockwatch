import yfinance as yf
import pandas as pd
import sys
import csv
import datetime

#try:
FileWrite=0

#stock_name = "XPP.L"
stock_name = "NESTLEIND.NS"
stock_obj = yf.Ticker(stock_name)


# get stock info
stock_info= stock_obj.info
stock_sustain = stock_obj.get_sustainability()
stock_splits=stock_obj.get_splits()
# get historical market data, here max is 5 years.
stock_history = stock_obj.history(period="max")

try:
    if(FileWrite):
        with open('C:/Users/harsh/OneDrive/Documents/stockwatch/data/'+stock_name+'info_'+str(datetime.datetime.now()).replace(":","_").replace(" ","_")+'.csv', 'w') as f:
            for key in stock_info.keys():
                f.write("%s,%s\n"%(key,stock_info[key]))
        with open('C:/Users/harsh/OneDrive/Documents/stockwatch/data/'+stock_name+'history_'+str(datetime.datetime.now()).replace(":","_").replace(" ","_")+'.csv', 'w') as f:
            for key in stock_history.keys():
                f.write("%s,%s\n"%(key,stock_history[key]))
        print("info exported")       
        #stock_info_df.to_csv('C:/Users/harsh/OneDrive/Documents/stockwatch/NESTLIND_info.csv')
except:
    print(sys.exc_info())


def f_clean():
    pass

f_clean()
#except:
#   print("Stock not found")

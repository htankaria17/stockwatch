import yfinance as yf
import pandas as pd
import sys
import csv
import datetime

#try:
FileWrite=1

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


"""
returns:
{
 'quoteType': 'EQUITY',
 'quoteSourceName': 'Nasdaq Real Time Price',
 'currency': 'USD',
 'shortName': 'Microsoft Corporation',
 'exchangeTimezoneName': 'America/New_York',
  ...
 'symbol': 'MSFT'
}
"""



"""
returns:
              Open    High    Low    Close      Volume  Dividends  Splits
Date
1986-03-13    0.06    0.07    0.06    0.07  1031788800        0.0     0.0
1986-03-14    0.07    0.07    0.07    0.07   308160000        0.0     0.0
...
2019-11-12  146.28  147.57  146.06  147.07    18641600        0.0     0.0
2019-11-13  146.74  147.46  146.30  147.31    16295622        0.0     0.0
"""
#except:
#   print("Stock not found")

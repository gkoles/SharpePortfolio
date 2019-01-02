## Computing Volatility

# Load the required modules and packages
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import datetime
from alpha_vantage.timeseries import TimeSeries

# AlphaVantage API
ts = TimeSeries(key='NCA3330GHPL99542', output_format='pandas', indexing_type='date ')

########################################################################
# FUNCTIONS BELOW
########################################################################

# Dily returns
def daily_ret(df):

    daily_ret = df.pct_change()

    return daily_ret


# Sharpe Ratio
def rolling_sharpe(returns, days=252):

    #add rolling ratio
    rolling_returns = returns.rolling(days)

    rolling_sharpe = np.sqrt(days) * ((
        rolling_returns.mean() ) / rolling_returns.std()
        )

    return rolling_sharpe

# read data from a ticker
def readFinanceData(ticker, start, end):
    data, meta_data = ts.get_daily_adjusted(symbol = ticker , outputsize='full')

    close = data['4. close']

    return close

########################################################################
# FINANCIAL DATA BELOW
########################################################################

start_date = (datetime.date.today() - datetime.timedelta(days=5*365))
end_date = datetime.date.today()

# Pull stock data
stock_ticker = 'SPY'
stock_data = readFinanceData(stock_ticker, start_date, end_date)

# Pull gold data
gold_ticker = 'GLD'
gold_data = readFinanceData(gold_ticker, start_date, end_date)

# Pull bond data
bond_ticker = 'TLT'
bond_data = readFinanceData(bond_ticker, start_date, end_date)

# Risk free return
rf_data = readFinanceData('SHY',start_date, end_date)

aggregated_df = pd.DataFrame(data=[stock_data, gold_data, bond_data, rf_data]).T

aggregated_df.index.names = ['date']
aggregated_df.index = pd.to_datetime(aggregated_df.index)

aggregated_df.columns = ['SPY', 'GLD', 'TLT', 'SHY']
aggregated_df = aggregated_df.dropna()

last_5 = aggregated_df[start_date:]

########################################################################
# SHARPE CALC BELOW
########################################################################

daily_ret = daily_ret(last_5)

sharpe = rolling_sharpe(daily_ret, 252)

sharpe = sharpe.dropna()

print (sharpe.tail())


########################################################################
# PLOT BELOW
########################################################################

fig, ax = plt.subplots(figsize=(20, 10))

sharpe['TLT'].plot(style='-', lw=3, label='TLT')\
        .axhline(y = 0, color = "black", lw = 3)

sharpe['GLD'].plot(style='-', lw=3, color='indianred', label='GLD')\
        .axhline(y = 0, color = "black", lw = 3)

sharpe['SPY'].plot(style='-', lw=3, color='blue', label='SPY')\
        .axhline(y = 0, color = "black", lw = 3)

plt.ylabel('Sharpe ratio')
plt.legend(loc='best')
plt.title('Rolling Sharpe ratio (12-month)')
fig.tight_layout()
plt.show()

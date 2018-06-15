# import warnings
# warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
# %matplotlib inline

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 100

def load_data():
    from_date = '2017.09.15'
    to_date = '2017.09.30'
    start_time = '03:00'
    end_time = '15:00'

    sym = '`USDINR'
    site = "`LOH"

    # load data from csv file (stratpy not available)
    prices_raw = pd.read_csv("./Prices_raw.csv")

    # parse timestamps correctly
    for t in [u'date' , u'ebsMarketUpdateTime', u'feedHandlerPublishTime', u'feedHandlerReceiveTime', u'eventCaptureTime']:
        prices_raw[t] = pd.to_datetime(prices_raw[t])
    
    return prices_raw

def clean_data(prices_raw):
    prices = prices_raw[['date','bid','ask','bid2','ask2','bid3','ask3','bidSize1','askSize1','bidSize2','askSize2','bidSize3','askSize3','paid', 'given']]

    prices['bid'] = prices['bid'].replace(0,np.NaN)
    prices['ask'] = prices['ask'].replace(0,np.NaN)
    prices['bid2'] = prices['bid2'].replace(0,np.NaN)
    prices['ask2'] = prices['ask2'].replace(0,np.NaN)

    prices['paid'] = prices['paid'].replace(0,np.NaN)
    prices['given'] = prices['given'].replace(0,np.NaN)
    prices['mid'] = 0.5*(prices['bid'] + prices['ask'])
    prices.index = prices_raw.feedHandlerReceiveTime

    return prices 
    # columns: feedHandlerReceiveTime | bid, ask, bid2, ask2, bid3, ask3, bidSize1, askSize1, bidSize2, askSize2, bidSize3, askSize3, paid, given, mid

def whiten_data(prices):
    mean_price = np.round(prices['mid'].mean(), 4)
    block_size = 1e6

    # prices[['bid', 'ask', 'bid2', 'ask2', 'bid3', 'ask3', 'mid']] -= mean_price
    # prices[['bidSize1', 'askSize1', 'bidSize2', 'askSize2', 'bidSize3', 'askSize3']] /= block_size
    return prices

########### Features engineering ########
def features_engineering(prices):
    # spread 
    prices['spread'] = prices['ask'] - prices['bid']

    # book pressure feature
    prices['bp'] = prices['mid'] - (prices['bidSize1']*prices['bid'] + prices['askSize1']*prices['ask'])/(prices['bidSize1']+prices['askSize1'])
    prices['bp_with2'] = prices['mid'] - (prices['bidSize1']*prices['bid'] + prices['askSize1']*prices['ask']
                                                     + prices['askSize2']*prices['ask2'] + prices['askSize3']*prices['ask3']
                                                     + prices['bidSize2']*prices['bid2'] + prices['bidSize3']*prices['bid3'])/(prices['bidSize1']+prices['askSize1']+prices['bidSize2']+prices['askSize2']+prices['bidSize3']+prices['askSize3'])

    # trade Features, print,tradeSeq,lastPaid,lastGiven,bidToPaid,bidToGiven,midToPaid ...
    atomicTrades = prices[['paid','given']].loc[(prices['paid']>1) | (prices['given']>1)] 
    atomicTrades.loc[atomicTrades['paid'] < 1, 'paid' ] = np.NaN
    atomicTrades.loc[atomicTrades['given'] < 1, 'given' ] = np.NaN
    atomicTrades = atomicTrades.replace(0,np.NaN)
    prices['paid'] = atomicTrades['paid']
    prices['given'] = atomicTrades['given']
    prices['print'] = np.where((prices['paid']>1) | (prices['given']>1),1,0)
    prices['tradeSeq'] = prices['print'].cumsum()
    prices['lastPaid'] = prices['paid'].ffill()
    prices['lastGiven'] = prices['given'].ffill()
    prices.drop('paid',1,inplace=True)
    prices.drop('given',1,inplace=True)
    prices['midToPaid'] = prices['mid'] - prices['lastPaid']
    prices['midToGiven'] = prices['mid'] - prices['lastGiven']
    prices['bidToPaid'] = prices['bid'] - prices['lastPaid']
    prices['bidToGiven'] = prices['bid'] - prices['lastGiven']
    prices['askToPaid'] = prices['ask'] - prices['lastPaid']
    prices['askToGiven'] = prices['ask'] - prices['lastGiven']

    #timestamp
    prices['weekday'] = prices.index.weekday
    # prices['daytimestamp'] = prices['time'].dt.timestamp

    # TODO: timestamp telling how advanced we are in the day, ranging from 0 to 1 

    # volatility
    vol_lookbacks = [1e3, 1e5, 1e7, 1e9]
    for lookback in vol_lookbacks:
        prices['mid_vol_%d_ms' % lookback] = prices['mid'].rolling('%dms' % lookback).std()
        prices['mid_vol_%d_ms' % lookback].ffill()

    # column name over which we build moving averages
    columns = ['bid','ask','bid2','ask2','bidSize1','askSize1','bidSize2','askSize2','mid','spread', 'bp', 'bp_with2']
    columns = ['bid','ask','bidSize1','askSize1','mid','spread','bp','bp_with2','lastPaid','lastGiven']
    
    #moving averages over last n rows
    row_intervals = [1, 5, 10, 20, 80, 320, 1280]
    for window in row_intervals:
        for feature in columns:
            prices['%s_ma_%d_row' % (feature, window)] = prices[feature].rolling(window, min_periods=1).mean()
            prices['mid_vol_%d_ms' % lookback].ffill()

    # moving averages over last n milliseconds
    time_intervals = [20, 80, 320, 1000, 4000, 16000] 
    for time_window in time_intervals:
        for feature in columns:
            prices['%s_ma_%d_ms' % (feature, time_window)] = prices[feature].rolling('%ds' % time_window, min_periods=1).mean()
            prices['%s_ma_%d_ms' % (feature, time_window)]
    
    # columns over which we'll build delta /deltadelta signals
    for col in ['spread', 'bid', 'ask']:
        columns.remove(col)
    ma_row_columns = ['%s_ma_%d_row' % (feature, window) for feature in columns for window in row_intervals]
    ma_ms_columns = ['%s_ma_%d_ms' % (feature, time_window) for feature in columns for time_window in time_intervals]
    ma_columns = ma_row_columns + ma_ms_columns

    # columns to differentiate once
    ma_columns += ['mid']
    delta_column_names = ['delta_' + col for col in ma_columns]
    prices[delta_column_names] = prices[ma_columns] - prices[ma_columns].shift(1)
    # columns to differentiate twice
    delta_delta_column_names = ['delta_' + col for col in delta_column_names]
    prices[delta_delta_column_names] = prices[delta_column_names] - prices[delta_column_names].shift(1)

    # drop first two rows since they're nan for the delta_delta
    prices = prices.iloc[2:] 

    prices['mid_diff_interval'] = (prices['delta_mid'] != 0).cumsum()

    # drop some features
    LLL = []
    for l in LLL:
        prices.drop(l,1,inplace=True)

    old_n_rows = prices.shape[0]
    prices.dropna(inplace=True)
    print 'Dropped %d out of %d rows containing NaNs' % (old_n_rows - prices.shape[0], old_n_rows)
    old_n_rows = prices.shape[0]
    prices = prices[(np.abs(stats.zscore(prices['delta_mid'])) < 5)]
    print 'Dropped %d out of %d rows with extreme z-score' % (old_n_rows - prices.shape[0], old_n_rows)

    ######### create feature to learn, ie next move (not to be used as covariates!)
    prices['midDiff'] = prices['mid'].diff()
    prices['nextMidDiff'] = prices['midDiff'].shift(-1)
    prices['nextMidVariation'] = prices['nextMidDiff'].replace(to_replace=0, method='bfill')
    # drop nans again (there may be new nan's in nextMidVariation?)
    old_n_rows = prices.shape[0]
    prices.dropna(inplace=True)
    print 'Dropped %d out of %d rows containing NaNs' % (old_n_rows - prices.shape[0], old_n_rows)

    mid_look_ahead = prices[['nextMidVariation']]
    # drop variables which should not be used as covariates
    prices.drop(['midDiff'], 1, inplace = True)
    prices.drop(['nextMidDiff'], 1, inplace = True)
    prices.drop(['nextMidVariation'], 1, inplace = True)

    return prices, mid_look_ahead

def split_test_train(prices, dep_var):
    features = prices.columns
    OUT = (prices.date == '2017.09.29') | (prices.date == '2017.09.28') 
    OUT = OUT | (prices.date == '2017.09.27') 
    IN = ~OUT

    X_train = np.array(prices[IN][features].values)
    y_train = np.array(dep_var[IN]['nextMidVariation'].values)
    X_test = np.array(prices[OUT][features].values)
    y_test = np.array(dep_var[OUT]['nextMidVariation'].values)

    y_train[y_train<0] = -1
    y_train[y_train>0] = 1
    y_test[y_test<0] = -1
    y_test[y_test>0] = 1

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    prices_raw = load_data()
    prices = clean_data(prices_raw)

    prices, dep_var = features_engineering(prices)
    prices = whiten_data(prices)
    split_test_train(prices, dep_var)


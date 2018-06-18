import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
from sklearn import linear_model, svm, kernel_ridge, kernel_approximation

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 100

def load_data():
    # load data from csv file (stratpy not available)
    prices_raw = pd.read_csv("./Prices_raw.csv")

    # parse timestamps correctly
    for t in [u'date' , u'ebsMarketUpdateTime', u'feedHandlerPublishTime', u'feedHandlerReceiveTime', u'eventCaptureTime']:
        prices_raw[t] = pd.to_datetime(prices_raw[t])
    
    return prices_raw

def clean_data(prices_raw):
    prices = prices_raw[['date','bid','ask','bid2','ask2','bid3','ask3','bidSize1','askSize1','bidSize2','askSize2','bidSize3','askSize3']]

    prices['bid'] = prices['bid'].replace(0,np.NaN)
    prices['ask'] = prices['ask'].replace(0,np.NaN)
    prices['bid2'] = prices['bid2'].replace(0,np.NaN)
    prices['ask2'] = prices['ask2'].replace(0,np.NaN)

    # prices['paid'] = prices['paid'].replace(0,np.NaN)
    # prices['given'] = prices['given'].replace(0,np.NaN)
    prices['mid'] = 0.5*(prices['bid'] + prices['ask'])
    prices.index = prices_raw.feedHandlerReceiveTime

    return prices 
    # columns: feedHandlerReceiveTime | bid, ask, bid2, ask2, bid3, ask3, bidSize1, askSize1, bidSize2, askSize2, bidSize3, askSize3, paid, given, mid

def whiten_data(prices):
    mean_price = np.round(prices['mid'].mean(), 4)
    block_size = 1e6

    prices[['bid', 'ask', 'bid2', 'ask2', 'bid3', 'ask3', 'mid']] -= mean_price
    prices[['bidSize1', 'askSize1', 'bidSize2', 'askSize2', 'bidSize3', 'askSize3']] /= block_size
    
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
    
    
    prices['weekday'] = prices.index.weekday
    
    # volatility
    vol_lookbacks = [1e3, 1e5, 1e7, 1e9]
    for lookback in vol_lookbacks:
        prices['mid_vol_%d_ms' % lookback] = prices['mid'].rolling('%dms' % lookback).std()
        prices['mid_vol_%d_ms' % lookback].ffill()

    # column name over which we build moving averages
    columns = ['bidSize1','askSize1','mid','bp_with2']
    
    #moving averages over last n rows
    row_intervals = [2, 5, 10, 320, 1280]
    for window in row_intervals:
        for feature in columns:
            prices['%s_ma_%d_row' % (feature, window)] = prices[feature].rolling(window, min_periods=1).mean()
            prices['mid_vol_%d_ms' % lookback].ffill()

    # moving averages over last n milliseconds
    time_intervals = [20, 80, 400, 1000, 16000] 
    for time_window in time_intervals:
        for feature in columns:
            prices['%s_ma_%d_ms' % (feature, time_window)] = prices[feature].rolling('%ds' % time_window, min_periods=1).mean()
            prices['%s_ma_%d_ms' % (feature, time_window)]
    
    # columns over which we'll build delta /deltadelta signals
    for col in ['spread', 'bid', 'ask']:
        if col in columns:
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
    # prices.drop(['midDiff'], 1, inplace = True) # does not need to be dropped since observed at time t
    prices.drop(['nextMidDiff'], 1, inplace = True)
    prices.drop(['nextMidVariation'], 1, inplace = True)

    return prices, mid_look_ahead

def split_test_train(prices, dep_var):
    # IN = row indices for train, OUT = for test
    OUT = (prices.date == '2017.09.29') | (prices.date == '2017.09.28') 
    OUT = OUT | (prices.date == '2017.09.27') 
    IN = ~OUT

    X_train = prices[IN]
    X_train.drop(['date'], 1, inplace = True) # drop the date in order to keep a multivariate input
    cols = list(X_train.columns)
    X_train = np.array(X_train.values)
    y_train = np.array(dep_var[IN]['nextMidVariation'].values)
    
    X_test = prices[OUT]
    X_test.drop(['date'], 1, inplace = True)
    X_test = np.array(X_test.values)
    y_test = np.array(dep_var[OUT]['nextMidVariation'].values)

    y_train[y_train<0] = -1
    y_train[y_train>0] = 1
    y_test[y_test<0] = -1
    y_test[y_test>0] = 1

    return X_train, y_train, np.array(dep_var[IN]['nextMidVariation'].values), X_test, y_test, np.array(dep_var[OUT]['nextMidVariation'].values), cols

def classif_correct_rate(estim, truth):
    return 1.0 - np.linalg.norm(np.sign(estim) - truth, ord = 1) / (2 * estim.shape[0])

def print_statistics(y_train, y_test):
    print('train: %d instances\t+1: %.3f, -1: %.3f' % (y_train.shape[0], np.sum(y_train[y_train > 0]) / y_train.shape[0], np.sum(y_train[y_train < 0]) / y_train.shape[0]))
    print('test:  %d instances\t+1: %.3f, -1: %.3f' % (y_test.shape[0],  np.sum(y_test[y_test > 0]) / y_test.shape[0],    np.sum(y_test[y_test < 0]) / y_test.shape[0]))

if __name__ == '__main__':
    prices_raw = load_data()
    prices = clean_data(prices_raw)
    prices = whiten_data(prices)
    prices, dep_var = features_engineering(prices)
    X_train, y_train, y_train_value, X_test, y_test, y_test_value, cols = split_test_train(prices, dep_var)
    
    print_statistics(y_train, y_test)

    # First estimator: assume that after an up day there comes a down day
    y_MR_train = -X_train[:, cols.index('midDiff')]
    y_MR_test = -X_test[:, cols.index('midDiff')]

    lin_ridge_reg = linear_model.Ridge(alpha = 0.2, max_iter = 1e5, normalize = True, tol = 1e-8)
    lin_ridge_reg.fit(X_train, y_train_value)
    y_lin_ridge_reg_test = lin_ridge_reg.predict(X_test)
    y_lin_ridge_reg_train = lin_ridge_reg.predict(X_train)

    rbf_feature = kernel_approximation.RBFSampler(gamma = 1e-11, n_components = 1000, random_state = 0)
    PhiX_train = rbf_feature.fit_transform(X_train)
    PhiX_test = rbf_feature.fit_transform(X_test)
    RFF_lin_ridge_reg = linear_model.Ridge(alpha = 1e-2, max_iter = 1e5, normalize = True, tol = 1e-8)
    RFF_lin_ridge_reg.fit(PhiX_train, y_train)
    y_RFF_train = RFF_lin_ridge_reg.predict(PhiX_train)
    y_RFF_test = RFF_lin_ridge_reg.predict(PhiX_test)

    y_ensemble_test =  np.sign(y_RFF_test) + np.sign(y_lin_ridge_reg_test)
    y_ensemble_train = np.sign(y_RFF_train) + np.sign(y_lin_ridge_reg_train)

    y_ensemble_train[y_ensemble_train == 0] = np.sign(y_MR_train[y_ensemble_train == 0])
    y_ensemble_test[y_ensemble_test == 0] = np.sign(y_MR_test[y_ensemble_test == 0])

    classif_rates = {}
    classif_rates['Linear ridge reg.'] = [classif_correct_rate(y_lin_ridge_reg_test, y_test), classif_correct_rate(y_lin_ridge_reg_train, y_train)]
    classif_rates['RFF ridge reg.\t'] = [classif_correct_rate(y_RFF_test, y_test), classif_correct_rate(y_RFF_train, y_train)]
    classif_rates['Mean reverting\t'] = [classif_correct_rate(y_MR_test, y_test), classif_correct_rate(y_MR_train, y_train)]
    classif_rates['Ensemble\t'] = [classif_correct_rate(y_ensemble_test, y_test), classif_correct_rate(y_ensemble_train, y_train)]
    
    print 'Classif. rate\t\ttrain\t\ttest'
    for algo_name in classif_rates:
        print('%s\t%.3f\t\t%.3f' % (algo_name, classif_rates[algo_name][1], classif_rates[algo_name][0]))
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy import stats
#import matplotlib.pyplot as plt
#import seaborn as sns

#sns.set(style="whitegrid", color_codes=True)
import matplotlib

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 100

from_date = '2017.09.15'
to_date = '2017.09.30'
start_time = '03:00'
end_time = '15:00'

sym = '`USDINR'
site = "`LOH"

# load data from csv file (stratpy not available)
prices_raw = pd.read_csv("./Prices_raw.csv")

# parse timestamps correctly
for t in [u'date', u'ebsMarketUpdateTime', u'feedHandlerPublishTime', u'feedHandlerReceiveTime', u'eventCaptureTime']:
	prices_raw[t] = pd.to_datetime(prices_raw[t])

#print("First three rows of raw data")
#print(prices_raw.head(3))

#T = [u'date' , u'time' , u'ebsMarketUpdateTime', u'feedHandlerPublishTime', u'feedHandlerReceiveTime', u'eventCaptureTime']
#timeDeltas = prices_raw[T]
#timeDeltas['marketLatency'] = timeDeltas['feedHandlerReceiveTime'] - timeDeltas['ebsMarketUpdateTime']
#timeDeltas['processLatency'] = timeDeltas['feedHandlerPublishTime'] - timeDeltas['feedHandlerReceiveTime']

#timeDeltas[['marketLatency','processLatency',]].describe().T[['mean','std','max']]

N=[u'index', u'bid', u'ask', u'paid', u'given', u'bid2', u'bid3', u'bidSize1',\
   u'bidSize2', u'bidSize3', u'ask2', u'ask3', u'askSize1',u'askSize2', u'askSize3']

prices_raw[N].describe().T

prices = prices_raw[['date','bid','ask','bid2','ask2','bidSize1','askSize1','bidSize2','askSize2','paid', 'given']]

prices['bid'] = prices['bid'].replace(0,np.NaN)
prices['ask'] = prices['ask'].replace(0,np.NaN)
prices['bid2'] = prices['bid2'].replace(0,np.NaN)
prices['ask2'] = prices['ask2'].replace(0,np.NaN)

prices['paid'] = prices['paid'].replace(0,np.NaN)
prices['given'] = prices['given'].replace(0,np.NaN)
prices['mid'] =  prices['ask']
prices['mid'] = 0.5*(prices['bid'] + prices['mid'])

prices.index = prices_raw.feedHandlerReceiveTime

prices=prices.drop_duplicates()

#fig, axes = plt.subplots(nrows=2,ncols=2, figsize=(20,10))
#prices.groupby('date')['mid'].count().plot(ax = axes[0,0],title = sym+' \n number of orderbook snapshot updates per day', style = 'o-')
#prices.groupby('date')['paid','given'].count().plot(ax = axes[0,1],title = sym+' \n number of trades updates per day', style = 'o-')
#prices.groupby('date')['mid'].mean().plot(ax = axes[1,0],title = 'daily average mid price per day', style = 'o-')
#prices.groupby('date')[['bidSize1','askSize1','bidSize2','askSize2']].mean().plot(ax = axes[1,1],title = ' daily average liquidity in first 2 levels of book', style = 'o-')
#plt.show()

#?
#date = '2017.09.15'
prices['date'] = pd.to_datetime(prices.date)
#prices[prices['date']==date][['mid']].plot(figsize=(15,5), title=sym+ ' mid timeserie '+date)
#plt.show()

columns = ['bid','ask','bid2','ask2','bidSize1','askSize1','bidSize2','askSize2','mid']
prices_delta = prices[columns] - prices[columns].shift(1)
prices_delta.rename(columns = {'mid':'deltaMid','bid':'deltaBid','ask':'deltaAsk','bidSize1':'deltaBidSize1','askSize1':'deltaAskSize1',
                              'bidSize2':'deltaBidSize2','askSize2':'deltaAskSize2'}, inplace=True)

# add back old prices, and a midDiff for learning later
LL = ['mid','bid','ask','bidSize1','bidSize2','askSize1','askSize2']
prices_delta[LL] = prices[LL]
prices_delta['midDiffInterval'] = (prices_delta['deltaMid'] != 0).cumsum()

# drop some features
LLL = ['bid2','ask2','bidSize2','askSize2','deltaBidSize2','deltaAskSize2']
for l in LLL:
    prices_delta.drop(l,1,inplace=True)

# time feature (on feedHandlerRecieve), date,time ...
prices_delta['date'] = prices.date
prices_delta['time'] = prices.index

# trade Features, print,tradeSeq,lastPaid,lastGiven,bidToPaid,bidToGiven,midToPaid ...
atomicTrades = prices[['paid','given']].loc[(prices['paid']>1) | (prices['given']>1)]
atomicTrades.loc[atomicTrades['paid'] <1, 'paid' ] = np.NaN
atomicTrades.loc[atomicTrades['given'] <1, 'given' ] = np.NaN
atomicTrades = atomicTrades.replace(0,np.NaN)
prices_delta['paid'] = atomicTrades['paid']
prices_delta['given'] = atomicTrades['given']
prices_delta['print'] = np.where((prices_delta['paid']>1) | (prices_delta['given']>1),1,0)
prices_delta['tradeSeq'] = prices_delta['print'].cumsum()
prices_delta['lastPaid'] = prices_delta['paid'].ffill()
prices_delta['lastGiven'] = prices_delta['given'].ffill()
prices_delta.drop('paid',1,inplace=True)
prices_delta.drop('given',1,inplace=True)
prices_delta['midToPaid'] = prices_delta['mid'] - prices_delta['lastPaid']
prices_delta['midToGiven'] = prices_delta['mid'] - prices_delta['lastGiven']
prices_delta['bidToPaid'] = prices_delta['bid'] - prices_delta['lastPaid']
prices_delta['bidToGiven'] = prices_delta['bid'] - prices_delta['lastGiven']
prices_delta['askToPaid'] = prices_delta['ask'] - prices_delta['lastPaid']
prices_delta['askToGiven'] = prices_delta['ask'] - prices_delta['lastGiven']

# book preasure feature
prices['book_pressure'] = prices['mid'] - (prices['bidSize1']*prices['bid'] + prices['askSize1']*prices['ask'])/(prices['bidSize1']+prices['askSize1'])
prices_delta['book_pressure'] = prices['mid'] - (prices['bidSize1']*prices['bid'] + prices['askSize1']*prices['ask'])/(prices['bidSize1']+prices['askSize1'])

# spread feature
prices_delta['spread'] = prices_delta['ask'] - prices_delta['bid']

# create feature to learn, ie next move (not to be used as covariates!)
prices_delta['midDiff'] = prices_delta['mid'].diff()
prices_delta['nextMidDiff'] = prices_delta['midDiff'].shift(-1)
prices_delta['nextMidVariation'] = prices_delta['nextMidDiff'].replace(to_replace=0, method='bfill')

prices_delta.dropna(inplace=True)

prices_delta = prices_delta.replace(0,np.NaN)

#import functions as func
#func.formatdf(prices_delta.describe().transpose())

prices_delta = prices_delta.replace(np.NaN,0)
prices_delta_clean = prices_delta[(np.abs(stats.zscore(prices_delta['deltaMid'])) < 5)]
prices_delta_clean = prices_delta_clean.replace(0,np.NaN)
prices_delta_clean = prices_delta_clean.replace(np.NaN,0)

features = ['deltaBid','deltaAsk','deltaMid','midToPaid','midToGiven','bidSize1','askSize1','bidToPaid','askToGiven','bidToGiven','askToPaid', 'book_pressure','spread']

OUT = (prices_delta_clean.date == '2017.09.29') | (prices_delta_clean.date == '2017.09.28')
OUT = OUT | (prices_delta_clean.date == '2017.09.27')
IN = ~OUT

X_train = np.array(prices_delta_clean[IN][features].values)
y_train = np.array(prices_delta_clean[IN]['nextMidVariation'].values)
X_test = np.array(prices_delta_clean[OUT][features].values)
y_test = np.array(prices_delta_clean[OUT]['nextMidVariation'].values)



y_train[y_train<0] = -1
y_train[y_train>0] = 1
y_test[y_test<0] = -1
y_test[y_test>0] = 1


from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import keras
from base_nn import *

np.random.seed(437)

clf = LogisticRegression()

clf.fit(sparse.csr_matrix(X_train), y_train)
predictions = clf.predict(sparse.csr_matrix(X_test))


print(metrics.accuracy_score(y_test, predictions))
print(metrics.classification_report(y_test, predictions))


#nn shenanigans
import build_time_data

lookback = 20
lstm_train = build_time_data.construct_time_series_inputs(X_train,lookback=lookback)
lstm_test = build_time_data.construct_time_series_inputs(X_test,lookback=lookback)

print (lstm_train[0:5])

nn_y_train = []
for y in y_train:
	if y == 0:
		nn_y_train.append([1,0])
	else:
		nn_y_train.append([0,1])

nn_y_train = np.array(nn_y_train)
nn_y_test = []

for y in y_test:
	if y == 0:
		nn_y_test.append([1,0])
	else:
		nn_y_test.append([0,1])

nn_y_test = np.array(nn_y_test)

lstm_shape = (lookback+1,len(X_train[0]))
lstm_model = build_lstm_model(lstm_shape)

print("Training Model")
train_pred_model(lstm_model,lstm_train[lookback:],nn_y_train[lookback:],lstm_test[lookback:],nn_y_test[lookback:],batch_size=1,epochs=5,verbose=1)

evaluate_model(lstm_model,lstm_test[lookback:],nn_y_test[lookback:])
'''

basic_nn_model = build_pred_model(input_shape=len(X_train[0]))

trained_nn_model = train_pred_model(basic_nn_model,X_train,nn_y_train,X_test,nn_y_test,epochs=50,verbose=1)

predictions = trained_nn_model.predict(X_test)

print predictions[0:5]
print nn_y_test[0:5]

evaluate_model(trained_nn_model,X_test,nn_y_test)

print(metrics.accuracy_score(nn_y_test, predictions))
print(metrics.classification_report(nn_y_test, predictions))
'''

#features_importance = pd.DataFrame(index = features, data={'features importance':clf.feature_importances_})
#features_importance.plot(kind='bar', figsize=(15,5))
#plt.show()
import pandas as pd
import numpy as np

def orderbook_features(df):
    """
    builds a clean dataset of orderbook features to use in a supervised learning model
    :param df
    """
    prices = df[['date','feedHandlerReceiveTime','bid','ask','bid2','ask2','bidSize1','askSize1','bidSize2','askSize2',
                    'paid', 'given']]

    y = pd.to_datetime(prices['feedHandlerReceiveTime'],format="%Y-%m-%d %H:%M:%S.%f")

    prices.set_index(y, inplace = True)
    prices['bid'] = prices['bid'].replace(0,np.NaN)
    prices['ask'] = prices['ask'].replace(0,np.NaN)
    prices['bid2'] = prices['bid2'].replace(0,np.NaN)
    prices['ask2'] = prices['ask2'].replace(0,np.NaN)

    prices['mid'] = 0.5*(prices['bid']+prices['ask'])
    prices['paid'] = prices['paid'].replace(0,np.NaN)
    prices['given'] = prices['given'].replace(0,np.NaN)

    columns = ['bid','ask','bid2','ask2','bidSize1','askSize1','bidSize2','askSize2','mid']
    
    features = prices[columns] - prices[columns].shift(1)
    features.rename(columns = {'mid':'deltaMid','bid':'deltaBid','ask':'deltaAsk','bidSize1':'deltaBidSize1','askSize1':'deltaAskSize1',
                                  'bidSize2':'deltaBidSize2','askSize2':'deltaAskSize2'}, inplace=True)
    features['mid'] = prices['mid']
    features['bid'] = prices['bid']
    features['ask'] = prices['ask']
    features['bidSize1'] = prices['bidSize1']
    features['askSize1'] = prices['askSize1']
    features['bidSize2'] = prices['bidSize2']
    features['askSize2'] = prices['askSize2']
    features['midDiffInterval'] = (features['deltaMid'] != 0).cumsum()
    features['time'] = prices.index

    atomicTrades = prices[['paid','given']].loc[(prices['paid']>1) | (prices['given']>1)]
    atomicTrades.loc[atomicTrades['paid'] <1, 'paid' ] = np.NaN
    atomicTrades.loc[atomicTrades['given'] <1, 'given' ] = np.NaN

    atomicTrades = atomicTrades.replace(0,np.NaN)

    features['paid'] = atomicTrades['paid']
    features['given'] = atomicTrades['given']

    features['print'] = np.where((features['paid']>1) | (features['given']>1),1,0)
    features['tradeSeq'] = features['print'].cumsum()

    features['lastPaid'] = features['paid'].ffill()
    features['lastGiven'] = features['given'].ffill()

    features.drop('paid',1,inplace=True)
    features.drop('given',1,inplace=True)

    features['midToPaid'] = features['mid'] - features['lastPaid']
    features['midToGiven'] = features['mid'] - features['lastGiven']

    features['bidToPaid'] = features['bid'] - features['lastPaid']
    features['bidToGiven'] = features['bid'] - features['lastGiven']
    features['askToPaid'] = features['ask'] - features['lastPaid']
    features['askToGiven'] = features['ask'] - features['lastGiven']

    features['book_pressure'] = prices['mid'] - (prices['bidSize1']*prices['bid'] + prices['askSize1']*prices['ask'])/(prices['bidSize1']+prices['askSize1'])

    features['midDiff'] = features['mid'].diff()
    features['nextMidDiff'] = features['midDiff'].shift(-1)
    features['nextMidVariation'] = features['nextMidDiff'].replace(to_replace=0, method='bfill')

    features['spread'] = features['ask'] - features['bid']
    features.dropna(inplace=True)
    
    return features


def prettynumberformat(x):
    """
       rounds to a number of decimals and seperates thousands with a comma
       :param x: the number to format
       :return: formatted number, for example 1,234,567 for 1234567.89 or 1,234.5 for 1234.567
    """
    groupdigits = lambda x, precision = 0: ('{:,.%sf}' % precision).format(x)
    if type(x) == float:
        if abs(x) < 100:
            return groupdigits(x, 2)
        elif abs(x) < 1000:
            return groupdigits(x, 1)
        else:
            return groupdigits(x, 0)
    elif type(x) == int:
        return groupdigits(x, 0)
    else:
        return x
    
def formatdf(df):
    """
    applies prettynumberformat to the columns of a pandas dataframe
    :param df: the pandas dataframe
    :return: returns the dataframe with formatted values
    """
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].map(prettynumberformat)
    
    return df

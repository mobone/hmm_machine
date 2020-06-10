import yfinance
import matplotlib
import matplotlib.pyplot as plt
from ta_indicators import get_ta
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd

def plot(df, name=None, show=False):
    df.loc[df['state']==0, 'color'] = 'firebrick'
    df.loc[df['state']==1, 'color'] = 'yellowgreen'
    df.loc[df['state']==2, 'color'] = 'forestgreen'
    df.loc[df['state']==3, 'color'] = 'darkslategray'

    df = df.dropna()
    df.plot.scatter(x='date',
                    y='close',
                    c='color',
                    )
                
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    
    if show == False and name is not None: 
        plt.savefig('./plots/%s.png' % name)
    else:
        plt.show()

    plt.close(fig)


def get_data(symbol, period):
    history = yfinance.Ticker(symbol).history(period=period, auto_adjust=False).reset_index()
    history = get_ta(history, volume=True, pattern=False)
    history.columns = map(str.lower, history.columns)
    history['return'] = history['close'].pct_change(1)    
    history['next_return'] = history['return'].shift(-1)
    history = history.dropna().reset_index(drop=True)
    
    train = history.loc[history['date']<'2015-01-01']
    test = history.loc[history['date']>'2015-01-01']
    return train, test


def run_feature_importances(train, n_total_features=20):
    test_cols = list(train.columns.drop(['date','open', 'high', 'low', 'close', 'return', 'next_return']))
    # get features
    clf = ExtraTreesRegressor(n_estimators=150, random_state=42)
    clf = clf.fit(train[test_cols], train['return'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    df = df.sort_values(by='importances')
    
    feature_choices = list(df['feature'].tail(n_total_features).values)

    preset_features = ['aroon_up', 'aroon_down', 'aroonosc','correl', 'mom', 'beta', 'rsi', 'bop', 
                        'ultimate_oscillator', 'bbands_upper', 'bbands_middle', 'bbands_lower', 
                        'bbands_upper_p', 'bbands_middle_p', 'bbands_lower_p', 'stochf_fastk', 'stochf_fastd', 'stochrsi_fastk', 'stochrsi_fastd' ]

    feature_choices = list(set( feature_choices + preset_features ))
    print('number of feature choices')
    print(len(feature_choices))
    
    top_starting_features = list(df.sort_values(by='importances').tail(10)['feature'].values)[::-1]
    return feature_choices, top_starting_features


def write_files(tickers, name, test):
    for symbol in tickers:
        if symbol is None:
            continue
        history = yfinance.Ticker(symbol).history(period='10y', auto_adjust=False).reset_index()
        filename = symbol+'_'+name
        print('writing file ', filename)
        history.to_csv(filename)



def get_results(tickers, name):
    #print(self.X_test.groupby(by=['predicted'])['next_return'].mean())
    
    #print(self.X_test.groupby(by=['predicted'])['next_return'].count())
    from strategy import setup_strategy
    #setup_strategy(files, name, strategy)
    
    files = {}
    files['instrument_count'] = 3
    files['instruments'] = tickers
    
    index_num = 1
    for ticker in tickers:
        files['instrument_%s_filename' % index_num] = '%s_%s' % (ticker, name)
        index_num = index_num + 1
    
    files['states'] = 'states_%s' % name
    files['with_none'] = False
    
    result = setup_strategy(files, name).T

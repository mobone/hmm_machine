import yfinance
from ta_indicators import get_ta
from hmmlearn.hmm import GaussianHMM, GMMHMM, MultinomialHMM
from mlxtend.feature_extraction import PrincipalComponentAnalysis
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import namegenerator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class generate_model:
    def __init__(self, features, n_subsets, n_components, name):
        self.lookback = 150

        train = pd.read_csv('./train.csv')
        test = pd.read_csv('./test.csv')
        
        train['date'] = pd.to_datetime(train['date'])
        test['date'] = pd.to_datetime(test['date'])

        trained_pipelines = self.get_trained_pipelines(train, features, n_subsets, n_components)
        test, models_used, num_models_used = self.run_pipelines(trained_pipelines, test, features)

        return test, models_used, num_models_used
        #print(trained_pipelines)

    def get_trained_pipelines(self, train, features, n_subsets, n_components):
        train_dfs = np.array_split(train, n_subsets)
        int_name = 0
        pipelines = []
        
        for train_subset in train_dfs:
            try:
                pipe_pca = make_pipeline(StandardScaler(),
                            PrincipalComponentAnalysis(n_components=n_components),
                            GMMHMM(n_components=n_components, covariance_type='full', n_iter=150, random_state=7),
                            )
                pipe_pca.fit(train_subset[ features ])

                train['state'] = pipe_pca.predict(train[ features ])
                results = pd.DataFrame(train.groupby(by=['state'])['return'].mean().sort_values())
                results['new_state'] = list(range(n_components))
                results.columns = ['mean', 'new_state']
                results = results.reset_index()
                results['name'] = int_name
                int_name = int_name + 1
                
                pipelines.append( [pipe_pca, results] )
                
            except Exception as e:
                #print('make trained pipelines exception', e)
                pass
        
        return pipelines

    def run_pipelines(self, pipelines, test, features):
        test.loc[:, 'state'] = None
        test.loc[:, 'model_used'] = 'None'
        for i in range(self.lookback, len(test)+1):
            this_test = test.iloc[ i - self.lookback : i]
            today = this_test[-1:]
            max_score = -np.inf
            for pipeline, train_results, in pipelines:
                
                try:
                    test_score = np.exp( pipeline.score( this_test[ features ]) / len(this_test) ) * 100
                    
                    if test_score>max_score:
                        state = pipeline.predict( this_test[ features ] )[-1:][0]
                        
                        state = int(train_results[train_results['state']==state]['new_state'])
                        
                        
                        
                        test.loc[today.index, 'state'] = state
                        
                        test.loc[today.index, 'model_used'] = train_results['name'].values[0]
                        max_score = test_score
                except Exception as e:
                    print('this exception', e)
                    continue
            
        print(test)
        test = test.dropna(subset=['state'])
        models_used = str(test['model_used'].unique())
        num_models_used = len(test['model_used'].unique())
        print(num_models_used)
        #states_plot = plot(test, name=name, show=True)
        return test, models_used, num_models_used

def generate_model_wrapper(features, n_subsets, n_components, name):
    test, models_used, num_models_used = generate_model(features, n_subsets, n_components, name)

    return test, models_used, num_models_used


"""
train, test = get_data('QQQ', period='15y')
print(train)
print(test)

train.to_csv('./train.csv', index=False)
test.to_csv('./test.csv', index=False)

name = namegenerator.gen()
test, models_used, num_models_used = generate_model(['mom', 'rsi', 'return'], 10, 3, name)
print(test)
test.to_csv('test.csv')
"""
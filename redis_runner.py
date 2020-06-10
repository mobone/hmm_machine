from hmmlearn.hmm import GaussianHMM, GMMHMM, MultinomialHMM
from mlxtend.feature_extraction import PrincipalComponentAnalysis
import yfinance
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import namegenerator
import numpy as np
from time import sleep
import warnings
from multiprocessing import Pool, cpu_count
from utils import get_data, plot, run_feature_importances, write_files, get_results

from rq.registry import FinishedJobRegistry
import io
from itertools import product
import sqlite3
from itertools import combinations
from random import shuffle, randint
from rq import Queue, Connection
from redis import Redis
from rq.job import Job
from hmm_class import generate_model_wrapper
import time
import hashlib
import traceback
# TODO: Try using hmm state as input for rfc
class run_machine():
    def __init__(self, params, feature_choices):
        self.redis_host = '192.168.1.128'
        print('sleeping')
        print(params)
        self.thread_id = randint(0,30)
        sleep(self.thread_id)
        print('starting')
        self.params  = params
        starting_feature, n_subsets, n_components = self.params
        self.starting_feature = starting_feature
        self.feature_choices = feature_choices
        self.n_subsets = n_subsets
        self.n_components = n_components
        self.lookback = 150

        self.failed = False

        self.features = ['return', starting_feature]
        self.conn = sqlite3.connect('results.db')

        #TODO: cover cases where all states are none

        self.run()

    def lookup_result(self, feature_hash):
        previous_result = None
        try:
            #print('looking up', feature_hash)
            sql = 'select * from results where feature_hash=="%s"' % feature_hash
            previous_result = pd.read_sql(sql, self.conn)
            if previous_result.empty:
                previous_result = None
        except Exception as e:
            pass

        return previous_result


    def run(self):
        while len(self.features)<21:
            self.results = pd.DataFrame()

            self.jobs = []
            self.create_jobs()
            self.get_redis_results()

            if self.failed == True:
                break


    def create_jobs(self):
        with Redis( host=self.redis_host ) as redis_con:
            #q = Queue(connection=redis_con)
            q = Queue(connection=redis_con, is_async=False)
            for new_feature in self.feature_choices:
                if new_feature in self.features:
                    continue

                test_features = self.features + [new_feature]
                #print(self.params, test_features)

                feature_hash = str(self.params)+'_'+str(test_features)
                #print(feature_hash)
                
                feature_hash = hashlib.md5(feature_hash.encode()).hexdigest()

                previous_result = self.lookup_result(feature_hash)

                if previous_result is None:
                    name = namegenerator.gen()

                    job_name = name+'__'+str(test_features)+'__'+feature_hash
                    
                    print('creating job', job_name)
                    # TODO: possibly use different queues for each simulation run
                    job_args = (test_features, 
                                self.n_subsets, 
                                self.n_components, 
                                #self.lookback, 
                                #self.with_rfc, 
                                #self.include_covid,
                                name, )
                    job = q.enqueue(generate_model_wrapper, args = job_args, job_timeout='6h',  result_ttl=3600 )
                    #TODO: change to nan and check for isnan in redis results
                    self.results = self.results.append( [ [str(test_features),  str(self.params), feature_hash, job.id, -np.inf, -np.inf, job.get_status()] ] )
                    job_dict = {'job_id': job.id,
                                'generated_name': name,
                                'test_features': test_features,
                                'feature_hash': feature_hash,
                                'job_status': job.get_status(),
                                'backtest_status': None
                                }
                    self.jobs.append( job_dict )
                    print('job_created')
                else:
                    #print(previous_result)
                    sharpe_ratio = float(previous_result['sharpe_ratio'])
                    cum_returns = float(previous_result['cum_returns'])
                    #print('found result', job_name, sharpe_ratio, cum_returns)
                    self.results = self.results.append( [ [str(test_features), str(self.params), feature_hash, 'previous', sharpe_ratio, cum_returns, 'previous'] ] )
            self.results.columns = ['features', 'params', 'feature_hash', 'job_id', 'sharpe_ratio', 'cum_returns', 'job_status']
            self.results = self.results.reset_index(drop=True)


    def get_redis_results(self):
        best_sharpe_ratio = -np.inf
        while True:
            try:
                with Redis( host=self.redis_host ) as redis_con:
                    #q = Queue(connection=redis_con)
                    #registry = FinishedJobRegistry(queue=q)
                    for job_dict in self.jobs:
                        #name, features, feature_hash = job_name.split('__')
                        features = job_dict['test_features']
                        job_id = job_dict['job_id']
                        feature_hash = job_dict['feature_hash']
                        name = job_dict['generated_name']

                        job = Job.fetch(job_id, connection = redis_con)

                        job_status = job.get_status()
                        self.results.loc[self.results['features']==str(features), 'job_status'] = job_status
                        job_dict['job_status'] = job_status
                        #print(job_id, job_status, type(job_status))
                        if job_status == 'started' or job_status == 'queued':
                            continue
                        
                        # if backtest has already been completed, continue
                        if job_dict['backtest_status'] != None:
                            continue

                        try:
                            test_with_states, models_used, num_models_used = job.result
                        except Exception as e:
                            print(e)
                            continue

                        self.get_backtest(test_with_states, n_components, name)

                        #print('running backtest', features, type(features))

                        # old 
                        #backtest_results = get_backtest(test_with_states, feature_hash, features, self.params, models_used, num_models_used, name=name, show_plot=False)
                        
                        #print(self.thread_id)
                        #print(backtest_results)
                        #print()
                        job_dict['backtest_status'] = 1

                        sharpe_ratio = float(backtest_results['sharpe_ratio'])
                        cum_returns = float(backtest_results['cum_returns'])

                        self.results.loc[self.results['features']==str(features), 'sharpe_ratio'] = sharpe_ratio
                        self.results.loc[self.results['features']==str(features), 'cum_returns'] = cum_returns
                        #self.results.loc[self.results['features']==str(features), 'job_status'] = job_status
                        
                        # TODO: fix bug here, values[0]
                        #print(self.results)
                        
                        # remove job
                        #if job_status == 'finished':
                        #    registry.remove(job_id)
                        # todo: store failed jobs
                        if 'win_rate' in backtest_results.columns:
                            backtest_results.to_sql('results', self.conn, if_exists='append')
                        if sharpe_ratio > best_sharpe_ratio:
                            if sharpe_ratio > 1:
                                plot(test_with_states, name=name, show=False)
                            best_sharpe_ratio = sharpe_ratio
                            best_features = features
            
            except Exception as e:
                print('EXCEPTION')
                print(e)
                traceback.print_exc()
                if 'no such job' in str(e):
                    self.jobs.remove(job_dict)
                    self.results = self.results[self.results['feature_hash']!=feature_hash]
                    print('removed job and result row')
                sleep(15)
                continue
            print()
            print('thread_id: ',self.thread_id)
            print(self.results[self.results['sharpe_ratio']>.4])
            print()
            num_queued = len(self.results[self.results['job_status']=='queued'])
            num_started = len(self.results[self.results['job_status']=='started'])
            if (num_queued + num_started)>0:
                print('waiting for', num_queued, num_started)
                print('\n','===========','\n')
                sleep(5)
                if num_started == 0:
                    sleep(60)
            else:
                break
            
        best_features = self.results[ self.results['sharpe_ratio'] == self.results['sharpe_ratio'].max() ]['features'].values[0]
        print('found best features', best_features)
        self.features = eval(best_features)

        # sharpe ratio not good enough, just quit
        if len(self.results[self.results['sharpe_ratio']>.4])==0:
            self.failed = True

    def get_backtest( test_with_states, n_components, name ):
        if n_components == 2:
            ticker_groups = [('QID', 'QQQ'), ('QID', 'QLD'), ('QID', 'TQQQ'), (None, 'QQQ'), (None, 'QLD'), (None, 'TQQQ'), ('QQQ', 'QLD'), ('QQQ', 'TQQQ'), ('QLD', 'TQQQ')]
        elif n_components == 3:
            ticker_groups = [('QID', None, 'QQQ'), ('QID', None, 'QLD'), ('QID', None, 'TQQQ'), ('QID', 'QQQ', 'QLD'), ('QID', 'QQQ', 'TQQQ'), ('QID', 'QLD', 'TQQQ'), (None, 'QQQ', 'QLD'), (None, 'QQQ', 'TQQQ'), (None, 'QLD', 'TQQQ'), ('QQQ', 'QLD', 'TQQQ')]
        elif n_components == 4:
            ticker_groups = [('QID', None, 'QQQ', 'QLD'), ('QID', None, 'QQQ', 'TQQQ'), ('QID', None, 'QLD', 'TQQQ'), ('QID', 'QQQ', 'QLD', 'TQQQ'), (None, 'QQQ', 'QLD', 'TQQQ')]
        elif n_components == 5:
            ticker_groups = [('QID', None, 'QQQ', 'QLD', 'TQQQ')]

        for tickers in ticker_groups:
            write_files(tickers, name, test_with_states)
            get_results(tickers, name)

def runner_method(params_with_features):
    params, feature_choices = params_with_features

    run_machine(params, feature_choices)


if __name__ == '__main__':
    
    train, test = get_data('QQQ', period='15y')

    train.to_csv('./train.csv', index=False)
    test.to_csv('./test.csv', index=False)
    
    feature_choices, top_starting_features = run_feature_importances(train, n_total_features=45)
    
    n_subsets = [3,5,10,15,20]
    # todo: test and work on n_components 3
    n_components = [4]
    #lookback = [50,100,150,200]
    #with_rfc = [True, False]
    #include_covid = [True, False]

    params = list(product( top_starting_features, n_subsets, n_components ))

    shuffle(params)
    
    params_with_features = []
    for param in params:
        params_with_features.append([param, feature_choices])

    #params = ['mom', feature_choices, 5, 4, 200, True ]
    #run_machine( params )

    #p = Pool(6)
    #p.map(runner_method, params_with_features)

    runner_method(params_with_features[0])
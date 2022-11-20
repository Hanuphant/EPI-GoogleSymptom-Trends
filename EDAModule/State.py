import os
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import numpy as np
from scipy.signal import correlate, correlation_lags
from statsmodels.tsa.stattools import grangercausalitytests
from typing import List, Tuple, Dict, Union, Optional
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor

class State:
    def __init__(self, name):
        super(State, self).__init__()
        self.name = name
        self.df, self.symptoms = self.load_data()
        self.add_CDC_data()

    def load_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load the data from the pickle file
        Returns:
            (pd.DataFrame, list) - the dataframe and the list of symptoms
        """
        f = open(f"./datasets/weekly/{self.name}/dataset.pkl", "rb")
        df = pkl.load(f)
        symptoms = [col for col in df.columns if 'symptom' in col]
        return (df, symptoms)

    def add_CDC_data(self)->None:
        """
        Adds the new_case and new_deaths from the parsed CDC files
        Returns:
            None
        """
        f = open(f"./datasets/weekly/{self.name}/CDCdataset.pkl", "rb")
        cdcdf = pkl.load(f)

        self.df['date'] = pd.to_datetime(self.df['date'])

        # Merge te datasets on date
        self.df = pd.merge(self.df, cdcdf, on='date', how='left')

        self.df['date'] = pd.to_datetime(self.df['date'])

    def correlation_analysis(self) -> pd.DataFrame:
        """
        Performs cross correlation analysis as well as granger causality tests
        Returns:
            pd.DataFrame - the dataframe with the results
        """

        # Create the dataframe
        results = pd.DataFrame({'symptom':[],'grangerCausalityPVal':[], 'mmcorrelation':[], 'time_lag':[], 'abscorrelation':[]})

        # For each symptom get the granger causality and cross correlation
        for symptom in self.symptoms:
            grangerdat = self.df[['new_cases', symptom]]

            # Standize the data
            grangerdatstd = (grangerdat - grangerdat.mean()) / grangerdat.std()

            # Scale the data
            grangerdatmm = (grangerdat - grangerdat.min()) / (grangerdat.max() - grangerdat.min())

            # Scipy cross correlation
            corrstd = correlate(grangerdatstd[symptom], grangerdatstd['new_case'])
            lags = correlation_lags(len(grangerdat[symptom]), len(grangerdat['new_case']))
            corr = correlate(grangerdatmm[symptom], grangerdatmm['new_case'])

            # Get the time lag
            time_lag = lags[np.argmax(corrstd)]

            # Granger Test
            gran = grangercausalitytests(grangerdat[['new_case', symptom]], maxlag=2, verbose=False)

            results = results.append({'symptom': symptom, 'grangerCausalityPVal': gran[1][0]['ssr_ftest'][1],
                                            'mmcorrelation': np.max(corr), 'time_lag': time_lag,
                                            'abscorrelation': np.max(corrstd)}, ignore_index=True)

        try:
            os.makedirs(f"./datasets/features/{self.name}/")
        except FileExistsError:
            pass
        with open(f"./datasets/features/{self.name}/correlation_features.pkl", "wb") as f:
            pkl.dump(results, f)
        return results

    def get_RFECV_features(self, random_state  : int = 789) -> pd.Series | List[str]:
        """
        Get the features selected by RFECV
        Arguments:
            random_state (int) - the random state to use
        Returns:
            list - the list of features
        """

        # Declare random state
        rng = np.random.RandomState(random_state)

        # Get the features
        features = self.symptoms

        # Get the target
        target = 'new_cases'

        # RFECVto get the total number of features
        rfecv = RFECV(estimator=RandomForestRegressor(random_state=random_state), cv=5, scoring='neg_mean_squared_error', n_jobs = -1)
        rfecv.fit(self.df[features], self.df[target])

        # Optimal number of features
        print("Optimal number of features : %d" % rfecv.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure(figsize=(20, 10))
        plt.xlabel('Number of features selected')
        plt.ylabel('Cross validation score (nb of correct classifications)')
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.show()

        # Get the features
        rfsyms = self.df[features].columns[rfecv.support_]

        return rfsyms

    def implement_darts(self, random_state : int = 789) -> pd.Series | List[str]:
        """
        Get the features selected by DARTS
        Arguments:
            random_state (int) - the random state to use
        Returns:
            prediction (pd.Series) - forecasted values
        """

        try:
            from darts import TimeSeries
        except ModuleNotFoundError:
            subprocess.run('pip install darts'.split(' '))
            from darts import TimeSeries
        from darts.models.forecasting.nbeats import NBEATSModel
        from darts.utils.likelihood_models import QuantileRegression
        from darts.metrics import mape

        # Declare random state
        rng = np.random.RandomState(random_state)

        # Get the features
        features = self.symptoms

        # Get the target
        target = 'new_cases'

        # Create time series object
        tsy = TimeSeries.from_dataframe(self.df, 'date', target)
        trainy = tsy[10:-50]





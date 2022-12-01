import os
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import numpy as np
from scipy.signal import correlate, correlation_lags
from statsmodels.tsa.stattools import grangercausalitytests
from typing import List, Tuple, Dict, Union, Optional, Set
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px


class State:
    def __init__(self, name: str) -> None:
        super(State, self).__init__()
        self.name = name
        self.initialize()


    def initialize(self) -> None:
        """
        Initialize the state
        Returns:
            None
        """
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
        self.df = self.df.merge(cdcdf, on='date', how='inner')

        self.df['date'] = pd.to_datetime(self.df['date'])

    def correlation_analysis(self, maxlag: int = 2) -> None:
        """
        Performs cross correlation analysis as well as granger causality tests

        Can be optimied by using pandas parallelization

        Returns:
            None
        """

        # Create the dataframe
        self.results = pd.DataFrame({'symptom': [], 'grangerCausalityPVal': [], 'mmcorrelation': [], 'time_lag': [], 'abscorrelation': [], 'ccfauc': [], 'ccf_lag1': [], 'grangerCasualityFtest': []})

        # For each symptom get the granger causality and cross correlation
        for symptom in self.symptoms:
            grangerdat = self.df[['new_case', symptom]]

            # Remove rows which are NA
            grangerdat = grangerdat.dropna()            

            # Standize the data
            grangerdatstd = (grangerdat - grangerdat.mean()) / grangerdat.std()

            # Scale the data
            grangerdatmm = (grangerdat - grangerdat.min()) / (grangerdat.max() - grangerdat.min())

            # Scipy cross correlation
            corrstd = correlate(grangerdatstd['new_case'], grangerdatstd[symptom])
            lags = correlation_lags(len(grangerdat['new_case']), len(grangerdat[symptom]))
            corr = correlate(grangerdatmm['new_case'], grangerdatmm[symptom])

            # Get the time lag
            time_lag = lags[np.argmax(corrstd)]

            # ccf_lag1
            ccf_lag1 = corr[np.where(lags == 1)[0][0]] - corr[np.where(lags == -1)[0][0]]

            # ccf-auc
            ccf_auc = np.trapz(corr[corr.shape[0]//2-maxlag:corr.shape[0]//2+maxlag+1], dx=0.01)

            # Granger Test
            gran = grangercausalitytests(grangerdat[['new_case', symptom]], maxlag=maxlag, verbose=False)

            # Get the lowest p-value in the gran test

            self.results = self.results.append({'symptom': symptom, 'grangerCasualityFtest': gran[maxlag-1][0]['ssr_ftest'][0] ,'grangerCausalityPVal': gran[maxlag-1][0]['ssr_ftest'][1],
                                            'mmcorrelation': np.max(corr), 'time_lag': time_lag,
                                            'abscorrelation': np.max(corrstd), 'ccfauc':ccf_auc, 'ccf_lag1': ccf_lag1},  ignore_index=True)

        try:
            os.makedirs(f"./datasets/features/{self.name}/")
        except FileExistsError:
            pass
        with open(f"./datasets/features/{self.name}/correlation_features.pkl", "wb") as f:
            pkl.dump(self.results, f)

    def get_gransyms(self, number_of_symptoms : int = 100, as_set : bool = False) -> Set[str] or pd.Series:
        """
        Get the symptoms with the highest granger causality
        Returns:
            gransyms (List[str]) - the symptoms with the highest granger causality
        """
        try:
            self.results
        except AttributeError:
            self.correlation_analysis()
        if as_set:
            return set(self.results.sort_values(by='grangerCausalityPVal', ascending=True).head(number_of_symptoms)['symptom'].values)
        else:
            return self.results.sort_values(by='grangerCausalityPVal', ascending=True).head(number_of_symptoms)['symptom'].values

    def get_corrsyms(self, number_of_symptoms : int = 100, as_set : bool = False) -> Set[str] or pd.Series:
        """
        Get the symptoms with the highest cross correlation
        Returns:
            corrsyms (List[str]) - the symptoms with the highest cross correlation
        """
        try:
            self.results
        except AttributeError:
            self.correlation_analysis()

        if as_set:
            return set(self.results.sort_values(by='abscorrelation', ascending=False).head(number_of_symptoms)['symptom'].values)
        else:
            return self.results.sort_values(by='abscorrelation', ascending=False).head(number_of_symptoms)['symptom'].values

    def get_RFECV_features(self, random_state : int = 789) -> pd.Series or List[str]:
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
        self.rfecv = RFECV(estimator=RandomForestRegressor(random_state=random_state), cv=5, scoring='neg_mean_squared_error', n_jobs = -1)
        self.rfecv.fit(self.df[features], self.df[target])

        # Optimal number of features
        print("Optimal number of features : %d" % self.rfecv.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure(figsize=(20, 10))
        plt.xlabel('Number of features selected')
        plt.ylabel('Cross validation score (nb of correct classifications)')
        plt.plot(range(1, len(self.rfecv.cv_results_['mean_test_score']) + 1), self.rfecv.cv_results_['mean_test_score'])
        plt.show()

        # Get the features
        self.rfsyms = self.df[features].columns[self.rfecv.support_]

    def design_features(self, feature_counts : int = None) -> None:
        """
        Based on the rfecv results we design the features
        Returns:
            None
        """

        try:
            self.results
        except AttributeError:
            self.correlation_analysis()

        # Get the feature_count set
        if feature_counts is None:
            try:
                feature_counts = self.rfecv.n_features_
            except AttributeError:
                self.get_RFECV_features()
                feature_counts = self.rfecv.n_features_

        # Get the features
        # Extract top 20 symptoms with highest absolute correlation
        self.corrsyms = self.get_corrsyms(number_of_symptoms=feature_counts)
        self.mmcorrsyms = self.results.sort_values(by='mmcorrelation', ascending=False).head(feature_counts)['symptom'].values

        # Extract top 20 symptoms with highest granger causality
        self.gransyms = self.get_gransyms(feature_counts)

        # Extract top 20 symptoms with highest leading time lag
        self.results['magdelay'] = self.results['time_lag'].abs()
        self.lagsyms = self.results.sort_values(by='magdelay', ascending=True).head(feature_counts)['symptom'].values

    def implement_darts(self, random_state : int = 789, input_chunk_length : int =  20, output_chunk_length : int = 5, feature_counts : int = None) -> pd.Series or List[str]:
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

        # Get the feature_count set
        self.design_features(feature_counts)

        def feature_selection_eval(features):

            if 'univariate' in features:
                trainy = tsy[:-60]
                testy = tsy[-60:]

                model = NBEATSModel(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_epochs=100, random_state=random_state, force_reset=True)
                model.fit(trainy, val_series=testy, epochs=100)
                predicted = model.predict(10)

                del model

                return np.array(predicted.values()), mape(predicted, testy)

            else:
                tsx = TimeSeries.from_dataframe(self.df, 'date', features)

                # Get the training and testing data
                trainy = tsy[10:-50]
                trainx = tsx[10:-50]
                testy = tsy[-50:]
                testx = tsx[-50:]

                # Get the model
                model = NBEATSModel(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_epochs=100, random_state=random_state, force_reset=True)
                model.fit(trainy, past_covariates=trainx, val_series=testy, val_past_covariates=testx, epochs=100)
                # model.fit(trainy, past_covariates=trainx, epochs = 100)
                predicted = model.predict(10, past_covariates=trainx)

                del model

                return np.array(predicted.values()), mape(predicted, testy)

        plt.figure(figsize=(20, 10))
        plt.plot(np.array(tsy.values()), label='actual')
        meths = {'pedicted_all': self.symptoms, 'predicted_granger': self.gransyms, 'predicted_mmcorrsyms': self.mmcorrsyms,
                 'predicted_ccf': self.corrsyms, 'predicted_rfe': self.rfsyms, 'leadingtime': self.lagsyms,
                 'univariate': ['univariate']}
        for key in meths.keys():
            pred, mapeval = feature_selection_eval(meths[key])
            print(mapeval)
            plt.plot(np.concatenate((np.array(trainy.values()), pred), axis=0), label=key)
        plt.xlabel('Weeks')
        plt.ylabel('New Cases')
        plt.savefig(f'./datasets/weekly/{self.name}/forecasting_feature_selection.png', dpi=300)
        plt.legend()
        plt.show()

    def find_1_step_neighbors(self) -> List:
        """
        Find the neighbors of the state
        Returns:
            neighbors (List) - the neighbors of the state
        """

        # Load the data
        neighbors = pkl.load(open('datasets/neighbors.pkl', 'rb'))

        # Get the neighbors
        return neighbors[self.name]

    def find_2_step_neighbors(self) -> Set:
        """
        Find the neighbors of the state
        Returns:
            neighbors (List) - the neighbors of the state
        """

        return set([neighbor for neighbor in self.find_1_step_neighbors() for neighbor in State(neighbor).find_1_step_neighbors()])

    def choropleth_maps(self, number_of_symptoms : int = 100) -> None:
        """
        Create a choropleth map
        Arguments:
            type (str) - the type of map to create

        Returns:
            None
        """
        try:
            self.results
        except AttributeError:
            self.correlation_analysis()

        neighbors = self.find_1_step_neighbors()
        neighborsofneighbors = list(self.find_2_step_neighbors().union(neighbors))

        choropleth_df = pd.DataFrame(neighborsofneighbors, columns=['state'])
        types = ['granger', 'correlation']
        for t in types:
            if t == 'granger':
                choropleth_df[t] = choropleth_df['state'].apply(lambda x:  list(self.get_gransyms(number_of_symptoms=number_of_symptoms, as_set=True) & State(x).get_gransyms(number_of_symptoms=number_of_symptoms, as_set=True)))
                choropleth_df[t+'_count'] = choropleth_df[t].apply(lambda x: len(x))

            if t == 'correlation':
                choropleth_df[t] = choropleth_df['state'].apply(lambda x:  list(self.get_corrsyms(number_of_symptoms=number_of_symptoms, as_set=True) & State(x).get_corrsyms(number_of_symptoms=number_of_symptoms, as_set=True)))
                choropleth_df[t+'_count'] = choropleth_df[t].apply(lambda x: len(x))

            fig = px.choropleth(choropleth_df, locations="state", locationmode="USA-states", color=t+'_count', scope="usa", hover_name="state", color_continuous_scale=px.colors.sequential.Blues)
            fig.update_layout(title_text=t)
            fig.write_image(f'./datasets/weekly/{self.name}/shared_{t}_symptoms.png')









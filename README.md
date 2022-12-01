# Suitability of Google Symptoms Trends data to analyze COVID pandemic: A geo-spatial case study

Packages required:
- DARTS (https://unit8co.github.io/darts/README.html)
- PYPOTS (https://github.com/WenjieDu/PyPOTS)
- Numpy
- Pandas
- Plotly
- Matplotlib
- Torch
- Sklearn
- Pickle
- Umap
- Statsmodel
- Prophet
- Kaleido
- Joblib
- Seaborn
- Scipy
- tqdm
- datetime
- random

How to use : 

- SAITS_imputation_forecasting_granger.ipynb : This file can be used for imputation using SAITS and forecasting using NBEATS and TCN for regions of high quality and low quality data. 
- Clustering.ipynb : Clustering of correlated symptoms is done here for all regions
- EDA.ipynb : Primary Exploratory Data Analysis and Recursive feature elimination is done here
- CDC.ipynb : CDC data collection, resampling to weekly data and analysis is done here
- EDA module : Contains helper functions for EDA.ipynb, CDC.ipynb and Clustering.ipynb

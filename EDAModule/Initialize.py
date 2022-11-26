import os
from glob import glob
import pickle as pkl
import numpy as np
import pandas as pd


def initialize():
    kaggle = os.path.exists("/kaggle/input")
    if kaggle:
        files = glob("../input/google-symptom-trends-as-of-october-1st-2022/202?_country_weekly_202?_US_weekly_symptoms_dataset.csv")
    else:
        files = glob("datasets/202?_country_weekly_202?_US_weekly_symptoms_dataset.csv")


    dfs = [pd.read_csv(file) for file in sorted(files)]
    df = pd.concat(dfs, ignore_index=True)
    del dfs

    # Data Stratification based on regions
    regions = df["sub_region_1_code"].unique()
    regions = np.delete(regions, 0)
    dfs = [df[df["sub_region_1_code"] == region].drop(columns=['sub_region_2', 'sub_region_2_code']) for region in
           regions]

    # Store the weekly dataframes to a pickle seperate pickle files
    for i, region in enumerate(regions):
        try:
            os.makedirs(f"./datasets/weekly/{region[3:]}")
        except FileExistsError:
            pass
        with open(f"./datasets/weekly/{region[3:]}/dataset.pkl", "wb") as f:
            pkl.dump(dfs[i], f)

    del dfs

    if kaggle:
        df = pd.read_csv("../input/cdc-covid-tracker-dataset-for-us/United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv")
    else:
        df = pd.read_csv("./datasets/United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv")

    dfs = []
    # Stratify the data by state
    for region in df['state'].unique():
        statedf = df[df['state'] == region]
        statedf['date'] = pd.to_datetime(statedf['submission_date'])
        statedf = statedf.drop(['submission_date'], axis=1)
        statedf = statedf.resample('W-MON', on='date').sum()
        statedf = statedf.reset_index()
        # Get full
        statedf['state'] = region
        try:
            pkl.dump(statedf, open(f"./datasets/weekly/{region}/CDCdataset.pkl", "wb"))
        except FileNotFoundError:
            pass
        dfs.append(statedf)

    del dfs
    return regions
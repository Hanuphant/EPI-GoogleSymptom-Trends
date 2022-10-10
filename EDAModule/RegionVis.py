#!/home/shrey/miniconda3/envs/cse8803e/bin/python

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import multiprocessing as mp
import pickle as pkl
import os

def generalRegionVisualiztion(regiondf, savepath):
    """
    This function takes in a dataframe with the columns as the symptoms and the rows as the dates.
    It then plots the data and saves the plot in the savepath.
    """

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # Plotting

    # Plot missing data heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(regiondf.isnull(), cbar=False, cmap = 'viridis')
    plt.savefig(savepath + "missing_data_heatmap.png")
    plt.close()

    # Plot the distribution heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(regiondf, cmap="viridis", cbar=True)
    plt.savefig(savepath + "distribution_heatmap.png")
    plt.close()

    # Plot the initial correlation heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(regiondf.corr(), cmap="viridis", cbar= True)
    plt.savefig(savepath + "initial_correlation_heatmap.png")
    plt.close()

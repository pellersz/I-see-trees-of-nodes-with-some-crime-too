# Implementation of the random forest (classification) ML algorithm 
# Evaluated on crime statistics at https://archive.ics.uci.edu/dataset/183/communities+and+crime

#from random_forest import RandomForest 
import pickle as pl
import math
import numpy as np
import pandas as pd
from pandas.core.window.common import flex_binary_moment
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from functools import cache
from ucimlrepo import fetch_ucirepo
from random import shuffle

#TODO: multiply continuous with 0.999

def _handle_missing_values(df):
    for col in df.columns:
        df[col].replace('?', np.nan, inplace=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].dropna().mean(), inplace=True)
    return df


def discretize_target(y, outcome_count):
    for i in range(outcome_count):
        y[i] = math.floor(y[i] * outcome_count);

    return y


def setup_data(
        file_name = "cached_obj",
        non_predictive_columns = ['state', 'county', 'community', 'communityname', 'fold'],
        outcome_count = 2
    ):
    try:
        with open(file_name, "rb") as file:
            print("Found cached data")
            return pl.load(file)
    except FileNotFoundError:
        pass #I don't quite know if this is considered better or not

    print("No cached data\nRetrieving dataset from the repository")
    fetched_data = fetch_ucirepo(id=183) 

    #original = fetched_data.data.original
    #original = original.drop(columns=non_predictive_columns)
    #original = _handle_missing_values(original)

    X = fetched_data.data.features
    X = X.drop(columns=non_predictive_columns)
    X = _handle_missing_values(X)

    y = discretize_target(fetched_data.data.targets["ViolentCrimesPerPop"], outcome_count)

    print(y, type(y))

    res = (X, y)

    pl.dump(res, open(file_name, "wb"))
    return res 


def divide_data(X, y, training_percentage):
    final_index = int(len(X) * training_percentage)

    permute = np.random.permutation(X.index)
    X = X.reindex(permute)
    y = y.reindex(permute)

    return (X[:final_index], y[:final_index], X[final_index:], y[final_index:])


def evaluate_model(forest, X, y):
    return True


def main(outcome_count = 2, training_percentage = 0.8, tree_count = 100, data_per_tree = 100, max_height = 20):
    print("Getting data")
    X, y = setup_data(outcome_count=outcome_count) 
    
    print("Creating training and test datasets")
    X_tr, y_tr, X_te, y_te = divide_data(X, y, training_percentage)
    
    #print("Starting training")
    #forest = RandomForest(X_tr, y_tr, X.data.features, tree_count, data_per_tree, max_height, outcome_count)

    print("Evaluating model")
    #percent = evaluate_model(forest, X_te, y_te)
    #print(f"Percent of correct guesses: {percent}%")



if __name__ == "__main__":
    main()


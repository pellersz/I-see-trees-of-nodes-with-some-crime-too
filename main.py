# Implementation of the random forest (classification) ML algorithm 
# Evaluated on crime statistics at https://archive.ics.uci.edu/dataset/183/communities+and+crime

from random_forest import RandomForest 
from rando_forest import RandoForest
import pickle as pl
import numpy as np
import pandas as pd
import sklearn as sk
from ucimlrepo import fetch_ucirepo
from time import perf_counter

def _handle_missing_values(df):
    for col in df.columns:
        df[col].replace('?', np.nan, inplace=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].dropna().mean(), inplace=True)
    return df


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
        print("No cached data\nRetrieving dataset from the repository")
        fetched_data = fetch_ucirepo(id=183) 

        X = fetched_data.data.features
        X = X.drop(columns=non_predictive_columns)
        X = _handle_missing_values(X)

        y = fetched_data.data.targets.ViolentCrimesPerPop
        y *= outcome_count * 0.9999
        y = np.floor(y)
        y = y.astype(np.int64)

        res = (X, y)

        pl.dump(res, open(file_name, "wb"))
        return res 


def evaluate_model(forest, X, y):
    count = 0
    for i in range(len(X)):
        if type(forest) == RandoForest:
            count += (forest.evaluate(X.iloc[i]) == y[i])
        else:
            count += (forest.predict(X.reindex([X.index[i]])) == y[i])

    return count / len(X)


def tree_count_search(X_tr, y_tr, X_te, y_te, min_tree_count = 1, max_tree_count = 100, data_per_tree = 150):
    for tree_count in range(min_tree_count, max_tree_count + 1):
        test_accuracy = 0;
        train_accuracy = 0;
        time_used = 0;
        for _ in range(2):
            for method in ("gini", "gain"):
                start = perf_counter()
                forest = RandoForest(X_tr, y_tr, X_tr.keys(), tree_count, data_per_tree, 20, method, 2)
                time_used += perf_counter() - start
                test_accuracy += evaluate_model(forest, X_te, y_te.iloc)
                train_accuracy += evaluate_model(forest, X_tr, y_tr.iloc)
                
        print(f"test accuracy = {test_accuracy / 4}, train accuracy = {train_accuracy / 4}, in {time_used / 4} seconds with parameters: tree_count = {tree_count}, data_per_tree = {data_per_tree}")


def hyper_parameter_search(X_tr, y_tr, X_te, y_te, min_tree_count = 1, max_tree_count = 15, min_data_per_tree = 10, max_data_per_tree = 150):
    best_test = 0
    best_train = 0

    for s in range(min_tree_count + min_data_per_tree, max_data_per_tree + max_tree_count + 1):
        for tree_count in range(min_tree_count, max_tree_count + 1):
            for method in ("gini", "gain"):
                data_per_tree = s - tree_count
                if data_per_tree > max_data_per_tree:
                    continue
                if data_per_tree < min_data_per_tree:
                    break

                start = perf_counter()
                forest = RandoForest(X_tr, y_tr, X_tr.keys(), tree_count, data_per_tree, 20, method, 2)
                time_used = perf_counter() - start
                evaluation_test = evaluate_model(forest, X_te, y_te.iloc)
                evaluation_train = evaluate_model(forest, X_tr, y_tr.iloc)
                
                if evaluation_test > best_test:
                    best_test = evaluation_test
                    yes_test = "*"
                elif evaluation_test == best_test:
                    yes_test = ":"
                else:
                    yes_test = ""

                if evaluation_train > best_train:
                    best_train = evaluation_train
                    yes_train = "+"
                elif evaluation_train == best_train:
                    yes_train = "-"
                else:
                    yes_train = ""

                if yes_test != "" or yes_train != "":
                    print(f"{yes_test}{yes_train} test accuracy = {evaluation_test}, train accuracy = {evaluation_train}, in {time_used} seconds with parameters: tree_count = {tree_count}, data_per_tree = {data_per_tree}, method = {method}")
            

def eval_model(forest_type, division, X, y, outcome_count = 2, tree_count = 100, data_per_tree = 100, max_height = 20):
    chunk_size = len(X) / division
    time_used = 0 
    print(f"test started for {forest_type}")

    accuracy_te = 0
    precision_te = 0
    recall_te = 0
    specificity_te = 0
    accuracy_tr = 0
    precision_tr = 0
    recall_tr = 0
    specificity_tr = 0

    for i in range(division):
        X_tr = X.iloc[lambda x: (i * chunk_size > x.index) | ((i + 1) * chunk_size <= x.index)]
        y_tr = y.iloc[lambda x: (i * chunk_size > x.index) | ((i + 1) * chunk_size <= x.index)] 
        X_te = X.iloc[lambda x: (i * chunk_size <= x.index) & ((i + 1) * chunk_size > x.index)]
        y_te = y.iloc[lambda x: (i * chunk_size <= x.index) & ((i + 1) * chunk_size > x.index)]

        start = perf_counter()
        if forest_type == RandoForest:
            forest = RandoForest(X_tr, y_tr, X.keys(), tree_count, data_per_tree, max_height, "gini", outcome_count)
            evaluate = forest.evaluate
 
        else:
            forest = sk.ensemble.RandomForestClassifier(n_estimators=tree_count)
            forest.fit(X_tr, y_tr)
            evaluate = forest.predict

        time_used += perf_counter() - start

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(X_te)):
            if forest_type == RandoForest: 
                pred_label = evaluate(X_te.iloc[j])
            else:
                pred_label = evaluate(X_te.reindex([X_te.index[j]]))

            true_label = y_te.iloc[j]

            if pred_label == 1 and true_label == 1:
                tp += 1
            elif pred_label == 1:
                fp += 1
            elif true_label == 1:
                fn += 1
            else:
                tn += 1
        accuracy_te += (tp + tn) / len(X_te)
        precision_te += tp / (tp + fp)
        recall_te += tp / (tp + fn)
        specificity_te += tn / (tn + fp)


        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(X_tr)):
            if forest_type == RandoForest: 
                pred_label = evaluate(X_tr.iloc[j])
            else:
                pred_label = evaluate(X_tr.reindex([X_tr.index[j]]))

            true_label = y_tr.iloc[j]

            if pred_label == 1 and true_label == 1:
                tp += 1
            elif pred_label == 1:
                fp += 1
            elif true_label == 1:
                fn += 1
            else:
                tn += 1
        accuracy_tr += (tp + tn) / len(X_tr)
        precision_tr += tp / (tp + fp)
        recall_tr += tp / (tp + fn)
        specificity_tr += tn / (tn + fp)

        print(f"finished round {i + 1}")

    print(f"Test finished for {forest_type} in time {time_used} seconds")
    
    print(f"avg. test accuracy: {accuracy_te / division}")
    print(f"avg. test precision: {precision_te / division}")
    print(f"avg. test recall: {recall_te/ division}")
    print(f"avg. test specificity {specificity_te / division}")
    print(f"avg. test f-score: {2 / ((1 / (precision_te / division)) + (1 / (recall_te / division)))}")

    print(f"avg. training accuracy: {accuracy_tr / division}")
    print(f"avg. training precision: {precision_tr / division}")
    print(f"avg. training recall: {recall_tr / division}")
    print(f"avg. training specificity {specificity_tr / division}")
    print(f"avg. training f-score: {2 / ((1 / (precision_tr / division)) + (1 / (recall_tr / division)))}")

    print("")
        

def main(outcome_count = 2, division = 3):
    print("Getting data")
    X, y = setup_data(outcome_count=outcome_count) 
    print("")

    permute = np.random.permutation(X.index)
    X = X.reindex(permute)
    X = X.reset_index(drop=True)
    y = y.reindex(permute)
    y = y.reset_index(drop=True)

    #final_index = int(len(X) * 0.8)
    #tree_count_search(X[:final_index], y[:final_index], X[final_index:], y[final_index:])
    #hyper_parameter_search(X[:final_index], y[:final_index], X[final_index:], y[final_index:])
    
    RandomForest(X, y, X.keys(), 10, 100, 20, "gain", 2)

    eval_model(RandoForest, division, X, y, outcome_count=outcome_count)
    eval_model(sk.ensemble.RandomForestClassifier, division, X, y)

if __name__ == "__main__":
    main()


# Random forest (classification) testing
# Evaluated on crime statistics at https://archive.ics.uci.edu/dataset/183/communities+and+crime

from random_forest import RandomForest
from rando_forest import RandoForest
import pickle as pl
import numpy as np
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import sklearn as sk
from ucimlrepo import fetch_ucirepo
from time import perf_counter
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


number_of_explainablity_tests = 100


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
        if type(forest) == RandoForest or type(forest) == RandomForest:
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


def hyper_parameter_search(X_tr, y_tr, X_te, y_te, min_tree_count = 1, max_tree_count = 100, min_data_per_tree = 30, max_data_per_tree = 150):
    vals = [[[0.0 for _ in range(11)] for _ in range(min_data_per_tree, max_data_per_tree + 1)] for _ in range(min_tree_count, max_tree_count + 1)]
    full_count = (max_data_per_tree + 1 - min_data_per_tree) * (max_tree_count + 1 - min_tree_count) 
    curr_count = 0

    for s in range(min_tree_count + min_data_per_tree, max_data_per_tree + max_tree_count + 1):
        for tree_count in range(min_tree_count, max_tree_count + 1):
            for method in ("gini",):
                data_per_tree = s - tree_count
                if data_per_tree > max_data_per_tree:
                    continue
                if data_per_tree < min_data_per_tree:
                    break

                curr_count += 1

                start = perf_counter()
                forest = RandomForest(X_tr, y_tr, X_tr.keys(), tree_count, data_per_tree, 20, method, 2)
                time_used = perf_counter() - start
               
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for j in range(len(X_te)):
                    pred_label = forest.evaluate(X_te.iloc[j])
                    true_label = y_te.iloc[j]

                    if pred_label == 1 and true_label == 1:
                        tp += 1
                    elif pred_label == 1:
                        fp += 1
                    elif true_label == 1:
                        fn += 1
                    else:
                        tn += 1

                vals[tree_count - min_tree_count][data_per_tree - min_data_per_tree][0] = (tp + tn) / len(X_te)
                vals[tree_count - min_tree_count][data_per_tree - min_data_per_tree][1] = (0 if tp == 0 else tp / (tp + fp))
                vals[tree_count - min_tree_count][data_per_tree - min_data_per_tree][2] = (0 if tp == 0 else tp / (tp + fn))
                vals[tree_count - min_tree_count][data_per_tree - min_data_per_tree][3] = (0 if tn == 0 else tn / (tn + fp))
                vals[tree_count - min_tree_count][data_per_tree - min_data_per_tree][4] = 2 * tp / (2*tp + fp + fn)

                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for j in range(len(X_tr)):
                    pred_label = forest.evaluate(X_tr.iloc[j])
                    true_label = y_tr.iloc[j]

                    if pred_label == 1 and true_label == 1:
                        tp += 1
                    elif pred_label == 1:
                        fp += 1
                    elif true_label == 1:
                        fn += 1
                    else:
                        tn += 1

                vals[tree_count - min_tree_count][data_per_tree - min_data_per_tree][5] = (tp + tn) / len(X_tr)
                vals[tree_count - min_tree_count][data_per_tree - min_data_per_tree][6] = (0 if tp == 0 else tp / (tp + fp))
                vals[tree_count - min_tree_count][data_per_tree - min_data_per_tree][7] = (0 if tp == 0 else tp / (tp + fn))
                vals[tree_count - min_tree_count][data_per_tree - min_data_per_tree][8] = (0 if tn == 0 else tn / (tn + fp))
                vals[tree_count - min_tree_count][data_per_tree - min_data_per_tree][9] = 2 * tp / (2*tp + fp + fn)

                vals[tree_count - min_tree_count][data_per_tree - min_data_per_tree][10] = time_used

        print(curr_count / full_count)     

    pl.dump(vals, open("cached_res", "wb"))


def eval_model(forest_type, division, X, y, outcome_count = 2, tree_count = 100, data_per_tree = 100, max_height = 20):
    global number_of_explainablity_tests

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

    data = {}

    for i in range(division):
        X_tr = X.iloc[lambda x: (i * chunk_size > x.index) | ((i + 1) * chunk_size <= x.index)]
        y_tr = y.iloc[lambda x: (i * chunk_size > x.index) | ((i + 1) * chunk_size <= x.index)] 
        X_te = X.iloc[lambda x: (i * chunk_size <= x.index) & ((i + 1) * chunk_size > x.index)]
        y_te = y.iloc[lambda x: (i * chunk_size <= x.index) & ((i + 1) * chunk_size > x.index)]

        start = perf_counter()
        if forest_type == RandoForest or forest_type == RandomForest:
            forest = forest_type(X_tr, y_tr, X.keys(), tree_count, data_per_tree, max_height, "gini", outcome_count)
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
            if forest_type == RandoForest or forest_type == RandomForest: 
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
        precision_te += (0 if tp == 0 else tp / (tp + fp))
        recall_te += (0 if tp == 0 else tp / (tp + fn))
        specificity_te += (0 if tn == 0 else tn / (tn + fp))


        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(X_tr)):
            if forest_type == RandoForest or forest_type == RandomForest: 
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
        precision_tr += (0 if tp == 0 else tp / (tp + fp))
        recall_tr += (0 if tp == 0 else tp / (tp + fn))
        specificity_tr += (0 if tn == 0 else tn / (tn + fp))

        data = explain_this(X_tr, y_tr, X_te, y_te, data)   

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

    data = [(key, data[key] / (division * number_of_explainablity_tests)) for key in data]
    data = sorted(data, key=lambda elem: elem[1])
    data = [data[:10], data[-10:]]
    data[1].reverse()

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].barh(range(10), [elem[1] for elem in data[0]])
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels([elem[0] for elem in data[0]])
    axes[0].set_xlabel("Mean negative importance")
    axes[0].set_title(f"Top 10 Features by Mean negative importance")
    axes[0].invert_yaxis()

    axes[1].barh(range(10), [elem[1] for elem in data[1]])
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([elem[0] for elem in data[1]])
    axes[1].set_xlabel("Mean positive importance")
    axes[1].set_title(f"Top 10 Features by Mean positive importance")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()

    print("")


def make_some_plots():
    stats = pl.load(open("cached_res", "rb"))
    stat_names = [
                    "test accuracy", 
                    "test precision", 
                    "test recall", 
                    "test specificity", 
                    "test f-score", 
                    "training accuracy", 
                    "training precision", 
                    "training recall",
                    "training specificity",
                    "training f-score",
                    "training to converge"
                ]

    for i in range(len(stat_names)):
        if i == 5:
            layer = np.array([[a[i] / 4 for a in b] for b in stats])
        else:
            layer = np.array([[a[i] for a in b] for b in stats])

        plt.imshow(layer, cmap='viridis', extent=(30.0, 150.0, 100.0, 1.0))
        plt.xlabel("data amount per tree")
        plt.ylabel("number of trees")
        plt.colorbar(label=stat_names[i])

        plt.show()


def explain_this(X_tr, y_tr, X_te, y_te, data = {}):
    global number_of_explainablity_tests
    forest = RandomForest(X_tr, y_tr, X_tr.keys(), 100, 100, 20, "gini", 2)
    explainer = LimeTabularExplainer(training_data=X_tr.to_numpy(), training_labels=y_tr.to_numpy(), feature_names=X_tr.keys(), class_names=("low", "high"), discretize_continuous=False)

    for i in range(number_of_explainablity_tests):
        expl = explainer.explain_instance(X_te.to_numpy()[i], forest.predict_proba, num_features=20)
        for name, measure in expl.as_list():
            if name not in data:
                data[name] = 0
            data[name] += measure
    return data


def main(outcome_count = 2, division = 10):
    print("Getting data")
    X, y = setup_data(outcome_count=outcome_count) 
    print("")

    permute = np.random.permutation(X.index)
    X = X.reindex(permute)
    X = X.reset_index(drop=True)
    y = y.reindex(permute)
    y = y.reset_index(drop=True)

    make_some_plots()

    eval_model(RandomForest, division, X, y, tree_count=10, data_per_tree=100, max_height=20, outcome_count=outcome_count)
    eval_model(sk.ensemble.RandomForestClassifier, division, X, y, tree_count=10, outcome_count=outcome_count)
    eval_model(RandomForest, division, X, y, tree_count=100, outcome_count=outcome_count)
    eval_model(sk.ensemble.RandomForestClassifier, division, X, y, tree_count=100, outcome_count=outcome_count)
    
    final_index = int(len(X) * 0.8)
    tree_count_search(X[:final_index], y[:final_index], X[final_index:], y[final_index:])
    hyper_parameter_search(X[:final_index], y[:final_index], X[final_index:], y[final_index:])


if __name__ == "__main__":
    main()


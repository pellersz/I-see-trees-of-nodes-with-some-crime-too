import math
from time import sleep
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from functools import cache
from ucimlrepo import fetch_ucirepo
from random import randrange
import threading

class DecisionTree:
    def calculate_gain(self, X, y, is_categorical, category_count, number_of_labels):
        n = len(X)
        res = -10000.0

        for i in range(category_count):
            for j in range(number_of_labels + 1):
                self.training_helper[i][j] = 0

        if is_categorical:
            #print("categorical?")
            for i in range(n):
                self.training_helper[X.iloc[i]][y[i]] += 1
                self.training_helper[X.iloc[i]][number_of_labels] += 1
    
            for i in range(category_count):
                if self.training_helper[i][number_of_labels] == 0:
                    continue

                child_entropy = 0.0 
                for j in range(number_of_labels):
                    if self.training_helper[i][j] == 0:
                        continue

                    p = float(self.training_helper[i][j]) / self.training_helper[i][number_of_labels]
                    child_entropy -= p * math.log(p)
                
                res -= (self.training_helper[i][number_of_labels] / n) * child_entropy
            
            return (res, 0)

        self.training_helper[1][number_of_labels] = n
        #print("populating helper")
        for i in range(n):
            self.training_helper[1][y[i]] += 1
        #print("getting column")

        #print(feature)
        #print("\n", "original", X)
        #print("\n", "iriginal", X.iloc)
        #print("\n", "sorted", X.sort_values())
        column = X.sort_values().iloc

        #print("got it")
        best_divider = 0.0
        for i in range(1, n):
            #print(f"calculating split {i}")
            self.training_helper[1][y[i - 1]] -= 1
            self.training_helper[1][number_of_labels] -= 1
            self.training_helper[0][y[i - 1]] += 1
            self.training_helper[0][number_of_labels] += 1
            
            #print("deciding")
            if column[i - 1] != column[i]:
                #print("we are if")
                new_res = 0.0
                divider = (column[i - 1] + column[i]) / 2
                for ii in range(category_count):
                    #print(f"now {ii}")
                    child_entropy = 0.0
                    for j in range(number_of_labels):
                        if self.training_helper[ii][j] == 0:
                            continue

                        p = self.training_helper[ii][j] / self.training_helper[ii][number_of_labels]
                        child_entropy -= p * math.log(p)
                
                    new_res -= (self.training_helper[ii][number_of_labels] / n) * child_entropy
                    
                if res < new_res:
                    res = new_res
                    best_divider = divider
                    #print(res)
        
        return (res, best_divider)


    def __init__(self, X, y, features, height_left, number_of_labels):
        print(len(X), 20 - height_left)

        n = len(X)
        last_same = 0
        max_category_count = 0
        max_ind = 0

        yloc = y.iloc

        #print("building tree")

        self.decision = -1
        self.child_count = 0
        self.divider = -1.0

        if height_left <= 0:
            count = number_of_labels * [0];

            for i in range(n):
                count[yloc[i]] += 1

            for i in range(1, number_of_labels):
                if count[i] > count[max_ind]:
                    max_ind = i

            self.decision = max_ind
            
            return

        #print("lastsames")
        while last_same < n and yloc[last_same] == yloc[0]:
            last_same += 1

        #print("donesames")
        if last_same == n:
            self.decision = yloc[0] 
            return

        #print("calculating max category count")
        for i in range(len(features)):
            if max_category_count < features[i][2]:
                max_category_count = features[i][2]

        self.training_helper = [[0 for _ in range(number_of_labels + 1)] for _ in range(max_category_count)]

        best_feature_ind = -1
        best_gain = -float("inf")
        
        #print(f"max categorical: {max_category_count}, other max: {number_of_labels + 1}")

        for i in range(len(features)):
            curr_feature = features[i]
            #print(f"calculating gain {i} for feature {curr_feature[0]}")
            #print("vell", X[curr_feature[0]])
            #print("unvell", X[curr_feature[0]])
            #csopi = X[curr_feature[0]].copy()
            gain, divider_for_feature = self.calculate_gain(X[curr_feature[0]], yloc, curr_feature[1], curr_feature[2], number_of_labels)
            if gain > best_gain:
                best_gain = gain
                best_feature_ind = i
                self.divider = divider_for_feature

        #print("done")

        best_feature = features[best_feature_ind]
        #print("best", best_feature, best_feature[0])
        self.best_feature = best_feature[0]
        #print("was")
        self.is_categorical = best_feature[1]
        #print("3")
        self.child_count = best_feature[2]
        #print("4")
        self.children = []
        #print("5")
        features.pop(best_feature_ind)

        #print("izi")
    
        column = X[self.best_feature].iloc
        #print("kolumbo", X[self.best_feature], column)
        #print(column[0])
        
        if self.is_categorical:
            indexes = []
            for i in range(self.child_count):
                indexes.append([])

            for i in range(n):
                indexes[column[i]].append(i)

            for i in range(self.child_count):
                if len(indexes[i]) != 0:
                    new_child = DecisionTree(X.reindex(indexes[i]), y.reindex(indexes[i]), features.copy(), height_left - 1, number_of_labels)
                    self.children[i] = new_child
                else:
                    self.children[i] = None

        indexes = [[], []]
        
        #print("idontknow")
        Xindex = X.index
        for i in range(n):
            indexes[int(column[i] < self.divider)].append(Xindex[i])
        #print("yes")

        #print(X)
        #print(indexes[0])
        #print(X.reindex(indexes[0]))
        #print(y)
        #print(y.reindex(indexes[0]))
        #print(features.copy())
        new_child = DecisionTree(X.reindex(indexes[0]), y.reindex(indexes[0]), features.copy(), height_left - 1, number_of_labels)
        #print("nyes")
        self.children.append(new_child)
        new_child = DecisionTree(X.reindex(indexes[1]), y.reindex(indexes[1]), features.copy(), height_left - 1, number_of_labels)
        self.children.append(new_child)

        #print("that done too")
        #print("finished")

    def evaluate(self, x):
        if self.decision != -1:
            return self.decision
        
        if self.is_categorical:#might need an index 0
            return self.children[x[self.best_feature]].evaluate(x)
        
        return self.children[0].evaluate(x) if x[self.best_feature] < self.divider else self.children[1].evaluate(x)


def do_it(X, y, features, max_height, number_of_labels, idx, dest):
    dest[idx] = DecisionTree(X, y, features, max_height, number_of_labels)

class RandoForest:
    def __init__(self, X, y, feature_names, tree_count, data_per_tree, max_height, number_of_labels):
        #cdef DecisionTree new_tree
       
        self.number_of_labels = number_of_labels
        self.tree_count = tree_count
        self.trees = tree_count * [None]
        self.count = number_of_labels * [0]
        features = []

        for i in range(len(feature_names)):
            if np.issubdtype(type(X[feature_names[i]].iloc[0]), np.integer):
                categories = set()
                for category in X[feature_names[i]]:
                    categories.add(category)
                features.append((feature_names[i], True, len(categories)))
            elif np.issubdtype(type(X[feature_names[i]].iloc[0]), np.floating):
                features.append((feature_names[i], False, 2))
            else:
                pass
        
        threads = []
        for i in range(tree_count): 
            print(f"tree {i}")
            X_sample = X.sample(data_per_tree, replace=True)
            y_sample = y.reindex_like(X_sample)
            X_sample = X_sample.reset_index(drop=True)
            y_sample = y_sample.reset_index(drop=True)
            #new_tree = DecisionTree(X_sample, y_sample, features, max_height, number_of_labels)
            #self.trees.append(new_tree)
            t = threading.Thread(target=do_it, args=(X_sample, y_sample, features.copy(), max_height, number_of_labels, i, self.trees))
            threads.append(t)

        for t in threads:
            t.start()
        
        # Wait for all threads to finish
        count = 0
        for t in threads:
            t.join()
            print("finished ", count)
            count += 1
          

    def evaluate(self, x):
        max_ind = 0
        for i in range(self.number_of_labels):
            self.count[i] = 0

        for i in range(self.tree_count):
            self.count[self.trees[i].evaluate(x)] += 1

        for i in range(self.number_of_labels): 
            if self.count[i] > self.count[max_ind]:
                max_ind = i

        return max_ind



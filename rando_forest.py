import math
import numpy as np
import pandas as pd
import threading

class _DecisionTree:
    def _calculate_goodness(self, X, y, method, is_categorical, category_count, number_of_labels):
        n = len(X)
        res = -10000.0

        for i in range(category_count):
            for j in range(number_of_labels + 1):
                self.training_helper[i][j] = 0

        if is_categorical:
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

                    p = self.training_helper[i][j] / self.training_helper[i][number_of_labels]
                    child_entropy -= method(p)
                
                res -= (self.training_helper[i][number_of_labels] / n) * child_entropy
            
            return (res, 0)

        X = X.sort_values()
        y = y.reindex_like(X).iloc
        column = X.iloc

        self.training_helper[1][number_of_labels] = n
        for i in range(n):
            self.training_helper[1][y[i]] += 1

        best_divider = 0.0
        for i in range(1, n):
            self.training_helper[1][y[i - 1]] -= 1
            self.training_helper[1][number_of_labels] -= 1
            self.training_helper[0][y[i - 1]] += 1
            self.training_helper[0][number_of_labels] += 1
            
            if column[i - 1] != column[i]:
                new_res = 0.0
                divider = (column[i - 1] + column[i]) / 2
                for ii in range(category_count):
                    child_entropy = 0.0
                    for j in range(number_of_labels):
                        if self.training_helper[ii][j] == 0:
                            continue

                        p = self.training_helper[ii][j] / self.training_helper[ii][number_of_labels]
                        child_entropy -= method(p)
                
                    new_res -= (self.training_helper[ii][number_of_labels] / n) * child_entropy
                    
                if res <= new_res:
                    res = new_res
                    best_divider = divider
        
        return (res, best_divider)


    def __init__(self, X, y, features, height_left, method, number_of_labels):
        self.decision = -1
        self.child_count = 0
        self.divider = -1.0

        n = len(X)

        yloc = y.iloc

        if height_left <= 0:
            max_ind = 0
            count = number_of_labels * [0];

            for label in yloc:
                count[label] += 1

            for i in range(1, number_of_labels):
                if count[i] > count[max_ind]:
                    max_ind = i

            self.decision = max_ind
            
            return

        last_same = 0
        while last_same < n and yloc[last_same] == yloc[0]:
            last_same += 1

        if last_same == n:
            self.decision = yloc[0] 
            return

        max_category_count = 0
        for i in range(len(features)):
            if max_category_count < features[i][2]:
                max_category_count = features[i][2]

        self.training_helper = [[0 for _ in range(number_of_labels + 1)] for _ in range(max_category_count)]

        best_feature_ind = -1
        best_gain = -float("inf")

        for i in range(len(features)):
            curr_feature = features[i]
            gain, divider_for_feature = self._calculate_goodness(X[curr_feature[0]], y, method, curr_feature[1],curr_feature[2], number_of_labels)
            if gain > best_gain:
                best_gain = gain
                best_feature_ind = i
                self.divider = divider_for_feature

        best_feature = features[best_feature_ind]
        self.best_feature, self.is_categorical, self.child_count = best_feature
        self.children = self.child_count * [None]
    
        if self.is_categorical:
            for i in range(self.child_count):
                sub_X = X[X[self.best_feature] == i]
                if len(sub_X) != 0:
                    sub_y = y.reindex_like(sub_X)
                    new_child = _DecisionTree(sub_X, sub_y, features.copy(), height_left - 1, method, number_of_labels)
                    self.children[i] = new_child
                else:
                    self.children[i] = None

            return

        sub_X = X[X[self.best_feature] < self.divider]
        sub_y = y.reindex_like(sub_X)
        new_child = _DecisionTree(sub_X, sub_y, features.copy(), height_left - 1, method, number_of_labels)
        self.children[0] = new_child

        sub_X = X[X[self.best_feature] >= self.divider]
        sub_y = y.reindex_like(sub_X)
        new_child = _DecisionTree(sub_X, sub_y, features.copy(), height_left - 1, method, number_of_labels)
        self.children[1] = new_child


    def evaluate(self, x):
        if self.decision != -1:
            return self.decision
        
        if self.is_categorical: #might need an index 0
            return self.children[x[self.best_feature]].evaluate(x)
        
        return self.children[1].evaluate(x) if x[self.best_feature] > self.divider else self.children[0].evaluate(x)


class RandoForest:
    def _do_it(self, X, y, features, max_height, method, number_of_labels, idx):
        self.trees[idx] = _DecisionTree(X, y, features, max_height, method, number_of_labels)

    def __init__(self, X, y, feature_names, tree_count, data_per_tree, max_height, method, number_of_labels):
        self.number_of_labels = number_of_labels
        self.trees = tree_count * [None]
        self.count = number_of_labels * [0]
        features = []

        if method == "gini":
            method = lambda p: p * p 
        else:
            method = lambda p: p * math.log(p)

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
            X_sample = X.sample(data_per_tree, replace=True)
            y_sample = y.reindex_like(X_sample)
            X_sample = X_sample.reset_index(drop=True)
            y_sample = y_sample.reset_index(drop=True)
            t = threading.Thread(target=self._do_it, args=(X_sample, y_sample, features.copy(), max_height, method, number_of_labels, i))
            threads.append(t)

        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
          

    def evaluate(self, x):
        max_ind = 0
        for i in range(self.number_of_labels):
            self.count[i] = 0

        for tree in self.trees:
            self.count[tree.evaluate(x)] += 1

        for i in range(self.number_of_labels): 
            if self.count[i] > self.count[max_ind]:
                max_ind = i

        return max_ind


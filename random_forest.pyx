import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from functools import cache
from ucimlrepo import fetch_ucirepo
from random import randrange
import cython
from cython.mem import PyMem_Malloc, PyMem_Free


cdef class DecisionTree:
    cdef int decision
   
    cdef int is_categorical
    cdef int child_count
    cdef DecisionTree* children
    cdef int** training_helper
    
    cdef float divder
    #TODO: get rid of this
    cdef float tmpdivider


    cdef float calculate_gain(self, X, y, feature, is_categorical, int category_count, int number_of_labels):
        cdef int n = len(X)
        cdef float res = 0
        cdef float child_entropy
        cdef float p
        cdef float new_res
        cdef float divider

        for i in range(category_count):
            for j in range(number_of_labels + 1):
                self.training_helper[i][j] = 0

        if is_categorical: 
            for i in range(n):
                self.training_helper[X[feature][i]][y[i]] += 1
                self.training_helper[X[feature][i]][number_of_labels] += 1
    
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
            
            return res

        else:
            self.training_helper[1][number_of_labels] = n
            for i in range(n):
                self.training_helper[1][y[i]] += 1

            column = X[feature].copy().sort_values()

            for i in range(1, n - 1):
                count[1][y[i - 1]] -= 1
                count[1][number_of_labels] -= 1
                count[0][y[i - 1]] += 1
                count[0][number_of_labels] += 1
    
                new_res = 0.0

                if column[i - 1] != column[i]:
                    divider = (column[i - 1] + column[i]) / 2
                    for ii in range(category_count):
                        child_entropy = 0.0
                        for j in range(number_of_labels):
                            if self.training_helper[ii][j] == 0:
                                continue

                            p = float(self.training_helper[ii][j]) / self.training_helper[ii][number_of_labels]
                            child_entropy -= p * math.log(p)
                    
                        new_res -= (self.training_helper[ii][number_of_labels] / n) * child_entropy

                    if res < new_res:
                        res = new_res 
                        self.tmpdivider = divder
            
            return res


    def __cinit__(self, X, y, features, int height_left, int number_of_labels):
        cdef int n = len(X)
        cdef int* count 
        cdef int last_same = 0
        cdef int max_ind
        cdef int best_feature_ind
        cdef float gain 
        cdef float best_gain

        self.decision = -1
        self.children = NULL
        self.training_helper = NULL
        self.divder = -1.0

        if height_left <= 0:
            count = <int*> PyMem_Malloc(number_of_labels * sizeof(int));
            for i in range(number_of_labels):
                count[i] = 0

            for i in range(n):
                count[y[i]] += 1

            for i in range(1, number_of_labels):
                if count[i] > count[max_ind]:
                    max_ind = i
            self.decision = i
            
            PyMem_Free(count)
            return

        while last_same < n and y[last_same] == y[0]:
            last_same += 1

        if last_same == n:
            self.decision = y[0] 
            return

        self.training_helper = <int**> PyMem_Malloc(category_count * sizeof(int*))
        for i in range(category_count):
            self.training_helper[i] = <int*> PyMem_Malloc((number_of_labels + 1) * sizeof(int))

        best_feature_ind = -1
        best_gain = -float("inf")
        
        for i in range(len(features)):
            curr_feature = features[i]
            gain = calculate_gain(X, y, curr_feature[0], curr_features[1], curr_features[2], number_of_labels)
            if gain > best_gain:
                best_gain = gain
                best_feature_ind = i
                if not curr_feature[0]:
                    self.divder = self.tmpdivider

        best_feature = features[best_feature_ind]
        self.best_feature = best_feature[0]
        self.is_categorical = best_feature[1]
        self.child_count = best_feature[2]
        self.children = <DecisionTree*> PyMem_Malloc(self.child_count * sizeof(DecisionTree))
        features.pop(best_feature_ind)
    
        if self.is_categorical:
            indexes = []
            for i in range(category_count):
                indexes.append([])

            column = X[self.best_feature]
            for i in range(n):
                indexes[column[i]].append(i)

            for i in range(category_count):
                if len(indexes[i]) != 0:
                    self.children[i] = DecisionTree(X.reindex(indexes[i]), y.reindex(indexes[i]), features.copy(), height_left - 1, number_of_labels)
                else:
                    self.children[i] = NULL

        for i in range(category_count):
            PyMem_Free(self.training_helper)
        PyMem_Free(self.training_helper)


    cdef int evaluate(x):
        if decision != -1:
            return self.decision
        
        if is_categorical:#might need an index 0
            return children[x[self.best_feature]].evaluate(x)
        
        return children[0].evaluate(x) if x[self.best_feature] < self.divider else children[1].evaluate(x)


    def __dealloc__(self):
        PyMem_Free(self.children)
            

cdef class RandomForest:
    cdef int tree_count
    cdef DecisionTree* trees
    cdef int* count

    def __cinit__(self, X, y, features_names, int tree_count, int data_per_tree, int max_height, int number_of_labels):
        self.tree_count = tree_count
        self.trees = <DecisionTree*> PyMem_Malloc(tree_count * sizeof(DecisionTree))
        self.count = <int*> PyMem_Malloc(number_of_labels * sizeof(int))
        features = []
        for i in range(len(features_names)):
            if type(X[feature_names[i][0]]) == int:
                categories = set()
                for category in X[feature_names[i]]:
                    categories.add(category)
                features.append((feature_names[i], True, len(categories)))
            elif type(X[feature_names[i][0]]) == float:
                features.append((feature_names[i], False, 2))
            else:
                pass
        
        for i in range(tree_count):        
            sample = [randrange(len(X)) for _ in randge(data_per_tree)]
            X_sample = X.reindex(sample)
            y_sample = y.reindex(sample)
            self.trees[i] = DecisionTree(X, y, features, max_height, number_of_labels)
          

    cdef int evaluate(self, x):
        cdef int max_ind = 0
        for i in range(self.number_of_labels):
            count[i] = 0

        for i in range(self.tree_count):
            count[trees[i].evaluate(x)] += 1

        for i in range(self.number_of_labels): 
            if count[i] > count[max_ind]:
                max_ind = i

        return max_ind


    def __dealloc__(self):
        PyMem_Free(self.trees)
        PyMem_Free(self.counter)


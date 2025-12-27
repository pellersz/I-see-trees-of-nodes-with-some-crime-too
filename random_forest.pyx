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
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.ref cimport Py_INCREF, Py_DECREF, PyObject


cdef class DecisionTree:
    cdef int decision
   
    cdef int is_categorical
    cdef int child_count
    cdef PyObject** children
    cdef int** training_helper
    
    cdef float divider
    #TODO: get rid of this
    cdef float tmpdivider


    cdef float calculate_gain(self, X, y, feature, is_categorical, int category_count, int number_of_labels):
        cdef int n = len(X)
        cdef float res = 0
        cdef float child_entropy
        cdef float p
        cdef float new_res
        cdef float divider

        print("emptying helper")
        for i in range(category_count):
            for j in range(number_of_labels + 1):
                self.training_helper[i][j] = 0
        print("emptied")

        if is_categorical:
            print("categorical?")
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

        print("not categorical")
        self.training_helper[1][number_of_labels] = n
        print("populating helper")
        for i in range(n):
            print(f"at 1 {y[i]}")
            self.training_helper[1][y[i]] += 1
        print("getting column")

        column = X[feature].copy().sort_values()

        for i in range(1, n - 1):
            print(f"calculating split {i}")
            self.training_helper[1][y[i - 1]] -= 1
            self.training_helper[1][number_of_labels] -= 1
            self.training_helper[0][y[i - 1]] += 1
            self.training_helper[0][number_of_labels] += 1
    
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
                    self.tmpdivider = divider
        
        return res


    def __cinit__(self, X, y, features, int height_left, int number_of_labels):
        cdef int n = len(X)
        cdef int* count 
        cdef int last_same = 0
        cdef int max_ind
        cdef int max_category_count = 0
        cdef int best_feature_ind
        cdef float gain 
        cdef float best_gain
        cdef DecisionTree new_child

        print("building tree")

        self.decision = -1
        self.children = NULL
        self.child_count = 0
        self.training_helper = NULL
        self.divider = -1.0

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

        print("lastsames")
        print(y, y.iloc[0])
        while last_same < n and y[last_same] == y[0]:
            print(last_same, y[last_same])
            last_same += 1

        print("donesames")
        if last_same == n:
            self.decision = y[0] 
            return

        print("calculating max category count")
        for i in range(len(features)):
            if max_category_count < features[i][2]:
                max_category_count = features[i][2]

        self.training_helper = <int**> PyMem_Malloc(max_category_count * sizeof(int*))
        for i in range(max_category_count):
            self.training_helper[i] = <int*> PyMem_Malloc((number_of_labels + 1) * sizeof(int))

        best_feature_ind = -1
        best_gain = -float("inf")
        
        print(f"max categorical: {max_category_count}, other max: {number_of_labels + 1}")

        for i in range(len(features)):
            print(f"calculating gain {i}")
            curr_feature = features[i]
            gain = self.calculate_gain(X, y, curr_feature[0], curr_feature[1], curr_feature[2], number_of_labels)
            if gain > best_gain:
                best_gain = gain
                best_feature_ind = i
                if not curr_feature[0]:
                    self.divider = self.tmpdivider

        best_feature = features[best_feature_ind]
        self.best_feature = best_feature[0]
        self.is_categorical = best_feature[1]
        self.child_count = best_feature[2]
        self.children = <PyObject**> PyMem_Malloc(self.child_count * sizeof(PyObject*))
        features.pop(best_feature_ind)
    
        if self.is_categorical:
            indexes = []
            for i in range(self.child_count):
                indexes.append([])

            column = X[self.best_feature]
            for i in range(n):
                indexes[column[i]].append(i)

            for i in range(self.child_count):
                if len(indexes[i]) != 0:
                    new_child = DecisionTree(X.reindex(indexes[i]), y.reindex(indexes[i]), features.copy(), height_left - 1, number_of_labels)
                    Py_INCREF(new_child)
                    self.children[i] = <PyObject*> new_child
                else:
                    self.children[i] = NULL

        for i in range(max_category_count):
            PyMem_Free(self.training_helper)
        PyMem_Free(self.training_helper)


    cdef int evaluate(self, x):
        if self.decision != -1:
            return self.decision
        
        if self.is_categorical:#might need an index 0
            return (<DecisionTree> self.children[x[self.best_feature]]).evaluate(x)
        
        return (<DecisionTree> self.children[0]).evaluate(x) if x[self.best_feature] < self.divider else (<DecisionTree> self.children[1]).evaluate(x)


    def __dealloc__(self):
        for i in range(self.child_count):
            if self.children[i] != NULL:
                Py_DECREF(<object> self.children[i])
        PyMem_Free(self.children)
            

cdef class RandomForest:
    cdef int tree_count
    cdef PyObject** trees
    cdef int* count

    def __cinit__(self, X, y, feature_names, int tree_count, int data_per_tree, int max_height, int number_of_labels):
        cdef DecisionTree new_tree
       
        self.tree_count = tree_count
        self.trees = <PyObject**> PyMem_Malloc(tree_count * sizeof(PyObject*))
        self.count = <int*> PyMem_Malloc(number_of_labels * sizeof(int))
        features = []

        for i in range(len(feature_names)):
            #print(np.issubdtype(type(X[feature_names[i]].iloc[0]), np.floating))
            if np.issubdtype(type(X[feature_names[i]].iloc[0]), np.integer):
                categories = set()
                for category in X[feature_names[i]]:
                    categories.add(category)
                features.append((feature_names[i], True, len(categories)))
            elif np.issubdtype(type(X[feature_names[i]].iloc[0]), np.floating):
                features.append((feature_names[i], False, 2))
            else:
                pass
        
        for i in range(tree_count): 
            print(f"tree {i}")
            print(len(features))
            print(features[0])
            sample = [randrange(len(X)) for _ in range(data_per_tree)]
            X_sample = X.reindex(sample)
            y_sample = y.reindex(sample)
            new_tree = DecisionTree(X, y, features, max_height, number_of_labels)
            Py_INCREF(new_tree)
            self.trees[i] = <PyObject*> new_tree
          

    cdef int evaluate(self, x):
        cdef int max_ind = 0
        for i in range(self.number_of_labels):
            self.count[i] = 0

        for i in range(self.tree_count):
            self.count[(<DecisionTree> self.trees[i]).evaluate(x)] += 1

        for i in range(self.number_of_labels): 
            if self.count[i] > self.count[max_ind]:
                max_ind = i

        return max_ind


    def __dealloc__(self):
        PyMem_Free(self.count)
        for i in range(self.tree_count):
            Py_DECREF(<object> self.trees[i])
        PyMem_Free(self.trees)



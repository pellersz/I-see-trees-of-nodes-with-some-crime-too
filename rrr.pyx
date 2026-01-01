# cython boundscheck=False

import math
import numpy as np
import threading

import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.ref cimport Py_INCREF, Py_DECREF, PyObject
cimport numpy as cnp
cnp.import_array()
from numpy cimport ndarray

cdef class _DecisionTree:
    cdef int decision
   
    cdef object best_feature
    cdef int is_categorical
    cdef int child_count
    cdef PyObject** children
    cdef int** training_helper
    
    cdef float divider
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef (float, float) _calculate_goodness(self, cnp.ndarray X, cnp.ndarray y, method, int is_categorical, int category_count, int number_of_labels):
        cdef int n = len(X)
        cdef float res = -10000.0
        cdef float child_entropy
        cdef float p
        cdef float new_res
        cdef float divider = -1
        cdef float best_divider = 0.0
        cdef cnp.ndarray indices

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

                    p = float(self.training_helper[i][j]) / self.training_helper[i][number_of_labels]
                    child_entropy -= method(p)
                
                res -= (self.training_helper[i][number_of_labels] / n) * child_entropy
            
            return (res, 0.0)

        self.training_helper[1][number_of_labels] = n
        for i in range(n):
            self.training_helper[1][y[i]] += 1

        indices = np.argsort(X)
        X = X[indices]
        y = y[indices]

        for i in range(1, n):
            self.training_helper[1][y[i - 1]] -= 1
            self.training_helper[1][number_of_labels] -= 1
            self.training_helper[0][y[i - 1]] += 1
            self.training_helper[0][number_of_labels] += 1
     
            if X[i - 1] != X[i]:
                new_res = 0.0
                divider = (X[i - 1] + X[i]) / 2
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


    def __cinit__(self, cnp.ndarray X, cnp.ndarray y, features, int height_left, method, int number_of_labels, int category_count = -1):
        cdef int n = len(X)
        cdef int* count 
        cdef int last_same = 0
        cdef int max_ind = 0
        cdef int best_feature_ind = -1
        cdef float gain
        cdef float best_gain = -10000.0
        cdef _DecisionTree new_child

        self.decision = -1
        self.child_count = 0
        self.divider = -1.0
        self.training_helper = NULL
        self.children = NULL

        if height_left <= 0:
            count = <int*> PyMem_Malloc(number_of_labels * sizeof(int));
            for i in range(number_of_labels):
                count[i] = 0

            for i in range(n):
                count[y[i]] += 1

            for i in range(1, number_of_labels):
                if count[i] > count[max_ind]:
                    max_ind = i

            self.decision = max_ind
            
            PyMem_Free(count)
            return

        if len(y) != 0 and np.all(y == y[0]):
            self.decision = y[0]
            return 

        if category_count == -1:
            for feature in features:
                if category_count < feature[2]:
                    category_count = feature[2]

        self.training_helper = <int**> PyMem_Malloc(category_count * sizeof(int*))
        for i in range(category_count):
            self.training_helper[i] = <int*> PyMem_Malloc((number_of_labels + 1) * sizeof(int))


        for i in range(len(features)):
            curr_feature = features[i]
            gain, divider_for_feature = self._calculate_goodness(X[:, curr_feature[0]], y, method, curr_feature[1], curr_feature[2], number_of_labels)
            if gain > best_gain:
                best_gain = gain
                best_feature_ind = i
                if not curr_feature[1]:
                    self.divider = divider_for_feature


        best_feature = features[best_feature_ind]
        self.best_feature = best_feature[0]
        self.is_categorical = best_feature[1]
        self.child_count = best_feature[2]
        self.children = <PyObject**> PyMem_Malloc(self.child_count * sizeof(PyObject*))
        #features.pop(best_feature_ind)

        if self.is_categorical:
            column = X[:, self.best_feature]
            
            indices = []
            for i in range(self.child_count):
                indices.append([])

            for i in range(n):
                indices[column[i]].append(i)

            for i in range(self.child_count):
                if len(indices[i]) != 0:
                    new_child = _DecisionTree(X.reindex(indices[i]), y.reindex(indices[i]), features.copy(), height_left - 1, method, number_of_labels, category_count)
                    Py_INCREF(new_child)
                    self.children[i] = <PyObject*> new_child
                else:
                    self.children[i] = NULL

        else:
            indices = X[:, self.best_feature] < self.divider
            new_child = _DecisionTree(X[indices], y[indices], features.copy(), height_left - 1, method, number_of_labels, category_count)
            Py_INCREF(new_child)
            self.children[0] = <PyObject*> new_child 

            indices = ~indices
            new_child = _DecisionTree(X[indices], y[indices], features.copy(), height_left - 1, method, number_of_labels, category_count)
            Py_INCREF(new_child)
            self.children[1] = <PyObject*> new_child

        for i in range(category_count):
            PyMem_Free(self.training_helper[i])
        PyMem_Free(self.training_helper)
    

    cdef int evaluate(self, x):
        if self.decision != -1:
            return self.decision
        
        if self.is_categorical:
            return (<_DecisionTree> self.children[x[self.best_feature]]).evaluate(x)
       

        return (<_DecisionTree> self.children[1]).evaluate(x) if x[self.best_feature] > self.divider else (<_DecisionTree> self.children[0]).evaluate(x)


    def __dealloc__(self):
        for i in range(self.child_count):
            if self.children[i] != NULL:
                Py_DECREF(<object> self.children[i])
        PyMem_Free(self.children)
            

cdef class RandomForest:
    cdef int tree_count
    cdef PyObject** trees
    cdef int* count
    cdef number_of_labels


    cdef _do_it(self, cnp.ndarray X, cnp.ndarray y, features, int max_height, method, int number_of_labels, int idx):
        new_tree = _DecisionTree(X, y, features, max_height, method, number_of_labels)
        Py_INCREF(new_tree)
        self.trees[idx] = <PyObject *> new_tree


    def __cinit__(self, X, y, feature_names, int tree_count, int data_per_tree, int max_height, method, int number_of_labels):
        self.tree_count = tree_count
        self.trees = <PyObject**> PyMem_Malloc(tree_count * sizeof(PyObject*))
        self.count = <int*> PyMem_Malloc(number_of_labels * sizeof(int))
        self.number_of_labels = number_of_labels
        features = []

        if method == "gini":
            method = lambda p: p * p 
        else:
            method = lambda p: p * math.log(p)

        for i in range(len(feature_names)):
            feature_name = feature_names[i]
            if np.issubdtype(type(X[feature_name].iloc[0]), np.integer):
                categories = set()
                for category in X[feature_name]:
                    categories.add(category)
                features.append((i, True, len(categories)))
            elif np.issubdtype(type(X[feature_name].iloc[0]), np.floating):
                features.append((i, False, 2))
            else:
                pass
        
        threads = []
        for i in range(tree_count): 
            X_sample = X.sample(data_per_tree, replace=True)
            y_sample = y.reindex_like(X_sample)
            X_sample = X_sample.to_numpy()
            y_sample = y_sample.to_numpy()

            t = threading.Thread(target=self._do_it, args=(X_sample, y_sample, features.copy(), max_height, method, number_of_labels, i))
            threads.append(t)

        for t in threads:
            t.start()
        
        for t in threads:
            t.join()

    
    def evaluate(self, x):
        cdef int max_ind = 0
        x = x.to_numpy()

        for i in range(self.number_of_labels):
            self.count[i] = 0

        for i in range(self.tree_count):
            self.count[(<_DecisionTree> self.trees[i]).evaluate(x)] += 1

        for i in range(self.number_of_labels): 
            if self.count[i] > self.count[max_ind]:
                max_ind = i

        return max_ind


    def __dealloc__(self):
        PyMem_Free(self.count)
        for i in range(self.tree_count):
            Py_DECREF(<object> self.trees[i])
        PyMem_Free(self.trees)



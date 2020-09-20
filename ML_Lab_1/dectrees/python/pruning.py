import monkdata as m
import dtree as dt
from matplotlib import pyplot as plt
import numpy as np

import random

#Create a partition data and randomly partition the original training set into training and validation set
#The random function reorders the data samples and returns the first and second parts separately4

def partition(data, f):

    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * f)
    return ldata[:breakPoint], ldata[breakPoint:]


def prune_trees(data, test):

    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    p_error = []

    for f in fractions:
        #MAke a partition randomly out of the training set
        train, validate = partition(data, f)
        #Build a tree from the partitioned training
        tree = dt.buildTree(train, m.attributes)
        #In the file dtree.py there is a utility function allPruned which returns
        #a sequence of all possible ways a given tree can be pruned.
        Trees = dt.allPruned(tree)

        #Write code which performs the complete pruning by repeatedly calling
        #allPruned and picking the tree which gives the best classification perfor-
        #mance on the validation dataset.

        best_performance = dt.check(tree, validate)

        temp_tree = 0
        best_tree = tree

        #You should stop pruning when all the
        #pruned trees perform worse than the current candidate.
        for t in Trees:
            temp_performance = dt.check(t, validate)
            if best_performance < temp_performance:
                best_performance = temp_performance
                best_tree = t

        p_error.append(1 - dt.check(best_tree, test))

    return p_error

def evaluate_pruning():
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    monk1_pruned = []
    monk3_pruned = []


    for i in range(100):
        monk1_pruned.append(prune_trees(m.monk1, m.monk1test))
        monk3_pruned.append(prune_trees(m.monk3, m.monk3test))

    monk1_pruned = np.transpose(monk1_pruned)
    monk3_pruned = np.transpose(monk3_pruned)

    #t1 = dt.buildTree(m.monk1,m.attributes)
    #t2 = dt.buildTree(m.monk3,m.attributes)



    mean1 = np.mean(monk1_pruned, axis=1)
    print("mean of Monk 1 :", mean1)
    mean3 = np.mean(monk3_pruned, axis=1)
    print("mean of Monk 3 :", mean3)
    std1 = np.std(monk1_pruned, axis=1)
    print("std of Monk 1 :", std1)
    std3 = np.std(monk3_pruned, axis=1)
    print("std of Monk 3 :", std3)

    plt.plot(fractions, mean1, marker='o', label="Mean-1")
    plt.title("Mean Error vs Fractions on MONK-1")
    plt.xlabel("Fractions")
    plt.ylabel("Means of Error")
    plt.legend(loc='upper right', frameon=False)
    #plt.show()

    plt.plot(fractions, mean3,  marker='o', label="Mean-3")
    plt.title("Mean Error vs Fractions on MONK-3")
    plt.xlabel("Fractions")
    plt.ylabel("Means of Error")
    plt.legend(loc='upper right', frameon=False)
    #plt.show()

    plt.plot(fractions, std1,  marker='o', label="STD-1")
    plt.title("Standard Deviation vs Fractions on MONK-1")
    plt.xlabel("Fractions")
    plt.ylabel("Standard Deviation from the Error")
    plt.legend(loc='upper right', frameon=False)
    # plt.show()

    plt.plot(fractions, std3,  marker='o', label="STD-3")
    plt.title("Mean Error/Standard Deviation vs Fractions on MONK-3")
    plt.xlabel("Fractions")
    plt.ylabel("Error of Pruned tree")
    plt.legend(loc='upper right', frameon=False)
    plt.show()

if __name__ == '__main__':
     evaluate_pruning()





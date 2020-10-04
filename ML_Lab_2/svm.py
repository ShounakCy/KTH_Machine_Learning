#!/usr/bin/env python3

import numpy as np
import random
from math import pow, exp

from matplotlib import patches
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Things to implement

# -> A suitable kernel function
def kernel(x1, x2, k_type):
    if k_type == "Linear":
        # Linear kernel simply returns the scalar product between the two points.This results in a linear separation.
        x1 = np.transpose(x1)
        return np.dot(x1, x2)
    elif k_type == 'poly2':
        x1 = np.transpose(x1)
        return pow(np.dot(x1, x2) + 1, 2)
    elif k_type == 'poly3':
        x1 = np.transpose(x1)
        return pow(np.dot(x1, x2) + 1, 3)
    elif k_type == 'RBF':
        sigma = 0.1
        return exp(-np.linalg.norm(x1 - x2, 2) ** 2 / (2 * sigma ** 2))


# -> Implement the function objective, This function will only receive the vector as a parameter.
def objectve(a):
    # implementing equation 4 as per the pdf, the dual problem
    obj1 = -sum(a)

    for i in range(N):
        for j in range(N):
            obj = 0.5 * a[i] * a[j] * pre_matrix[i][j]
            obj1 = obj + obj1
    '''
    for i in range(len(a)):
        obj1 = 0.5 * np.sum(np.dot(a[i],pre_matrix(inputs[i],targets[i],N, kernel_type))) + obj1
    '''
    return obj1


# You can pre-compute a matrix with these values:
def pre_matrix(inputs, targets, N, kernel_type):
    P = []
    # Indices i and j run over all the data points. Thus, if you have N data points, P should be an N*N matrix.
    # This matrix should be computed only once, outside of the function objective.
    for i in range(N):
        Q = []
        for j in range(N):
            Q.append(targets[i] * targets[j] * kernel(inputs[i], inputs[j], kernel_type))

        # Therefore,store it as a numpy array in a global variable
        P.append(np.array(Q))
    return np.array(P)


# ->Implement the function zerofun. This function should implement the equality constraint of (10).
# Also here,you can make use of numpy.dot to be efficient.
# zerofun is a function you have defined which calculates the value which should be constrained to zero
def zerofun(a):
    return np.dot(a, targets)


# -> Extract the non-zero alpha values
# Therefore,use a low threshold to determine which are to be regarded as non-zero.
# You need to save the non-zero alphas’s along with the corresponding data inputs and target values in a separate data structure, for instance a list.
def nz_alpha_values(alpha, inputs, targets, threshold):
    nonzero_alpha = []
    nonzero_input = []
    nonzero_target = []

    for i in range(len(alpha)):
        if alpha[i] > threshold:
            nonzero_alpha.append(alpha[i])
            nonzero_input.append(inputs[i])
            nonzero_target.append(targets[i])

    return nonzero_alpha, nonzero_input, nonzero_target


# -> Calculate the b value using equation (7)
def cal_b(alphas, inputs, targets, C, kernel_type):
    # Note that you must use a point on the margin.
    s = 0
    # This corresponds to a point with an alpha-value larger than zero, but less than C (if slack is used).
    for i in range(len(alphas)):
        if alphas[i] < C:
            s = i
            break
    b = 0
    for i in range(len(inputs)):
        b = b + alpha[i] * targets[i] * kernel(inputs[s], inputs[i], kernel_type)
    b = b - targets[s]
    return b


# ->Implement the indicator function
# Implement the indicator function (equation 6) which uses the non-zero alpha[i]’s together with the inputs[i]’s and targets[i]'s to classify new points.
def indicator(alpha, inputs, targets, b, s, kernel_type):
    ind = 0
    for i in range(len(inputs)):
        ind = ind + alpha[i] * targets[i] * kernel(s, inputs[i], kernel_type)
    ind = ind - b
    return ind


if __name__ == "__main__":
    # To get the same random data every time you run the program
    np.random.seed(100)
    # We use the function random.randn to generate arrays with random numbers from a normal distribution with zero
    # mean and unit variance. By multiplying with a number and adding a 2D-vector, we can scale and shift this
    # cluster to any position. The clusters all have a standard deviation of 0.2.

    # -> Implementation
    classA = np.concatenate(
        (np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
         np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

    # The samples are stored in the array inputs
    inputs = np.concatenate((classA, classB))

    # and the corresponding class labels (1 and -1) are stored in the array targets at corresponding indices.
    # There is also a corresponding N*1 array,targets,which contains the classes, i.e. the ti values, encoded as (-1 or 1)
    targets = np.concatenate(
        (np.ones(classA.shape[0]),
         -np.ones(classB.shape[0])))

    # The last four lines randomly reorders the sample
    N = inputs.shape[0]  # Number  of  rows  ( samples )
    permute = list(range(N))
    random.shuffle(permute)

    inputs = inputs[permute, :]
    targets = targets[permute]
    # global kernel_type
    kernel_type = "RBF"
    pre_matrix = pre_matrix(inputs, targets, N, kernel_type)
    # print(pre_matrix)

    threshold = pow(10, -5)

    C = 10
    # B is a list of pairs of the same length as the alpha -vector, stating the lower and upper bounds for the corresponding element in alpha
    B = [(0, C) for b in range(N)]
    # XC is used to impose other constraints, in addition to the bounds. We will use this to impose the equality constrain
    XC = {'type': 'eq', 'fun': zerofun}
    # start is a vector with the initial guess of the alpha vector
    start = np.zeros(N)
    # minimize
    ret = minimize(objectve, start, bounds=B, constraints=XC)
    print("ret", ret)
    alpha = ret['x']
    print(alpha)

    # Calculate non-zero alpha
    nonzero_alpha, nonzero_input, nonzero_target = nz_alpha_values(alpha, inputs, targets, threshold)
    print("Non_Zero_Alphas =", nonzero_alpha)
    print("Non_Zero_Inputs =", nonzero_input)
    print("Non_Zero_Targets =", nonzero_target)

    # Plotting

    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
             'b.')

    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'r.')

    plt.axis('equal')  # force same scale on both the axis
    # plt.savefig('svm_plot.pdf') # save a copy on the file
    # plt.show() # show plot on the screen

    xgrid = np.linspace(-5, 5)
    # print(xgrid)
    ygrid = np.linspace(-4, 4)

    b = cal_b(nonzero_alpha, nonzero_input, nonzero_target, C, kernel_type)
    # print("b", b)

    grid = np.array([[indicator(nonzero_alpha, nonzero_input, nonzero_target, b, np.array([x, y]), kernel_type)
                      for x in xgrid]
                     for y in ygrid])
    # print("grid : ", grid)

    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'),
                linewidths=(1, 1, 1))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('svmplt.pdf')
    plt.show()

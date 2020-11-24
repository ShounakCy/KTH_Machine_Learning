import numpy as np

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    for jdx, classname in enumerate(classes):    
        idx = labels == classname # Returns a true or false with the length of y
        # Or more compactly extract the indices for which y==class is true,
        # analogous to MATLAB's find
        idx = np.where(labels==classname)[0]
        xlc = X[idx,:] # Get the x for the class labels. Vectors are rows.
    
        weightsum = sum(W[idx,:])
        
        # Mean
        mu[jdx] += np.sum(X[idx,:]*W[idx,:], axis=0) / weightsum
                
        # Sigma - contains the covariance
        for dim in range(Ndims):
            totsum = 0
            for ind in idx:
                totsum += W[ind]*(X[ind][dim] - mu[jdx][dim]) ** 2
            sigma[jdx][dim][dim] = totsum / weightsum

    # ==========================

    return mu, sigma

# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
    for idx, classname in enumerate(labels):
        prior[classname] += W[idx]
        
    prior /= np.linalg.norm(prior)
    
    # ==========================

    return prior

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================
    for x_ind in range(Npts):
        for classname in range(Nclasses):
            term_one = (1/2)*np.log(np.linalg.det(sigma[classname]))
            
            term_two_sub_one = X[x_ind] - mu[classname]
            term_two_sub_two = np.diag(1/np.diag(sigma[classname]))
            term_two_sub_three = np.transpose(X[x_ind] - mu[classname])
            term_two = np.linalg.multi_dot([term_two_sub_one, term_two_sub_two, term_two_sub_three])
            
            term_three = np.log(prior[classname])
            
            logProb[classname][x_ind] = - term_one - term_two + term_three

    # ==========================
    
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
                
        weightsum = np.sum(wCur)
        
        # Adaboost step (2)
        error = weightsum      
        correctvote = np.where(vote == labels)[0]
        for i in correctvote:
            error -= wCur[i] 
        
        # Adaboost step (3)
        curAlpha = (np.log(1 - error) - np.log(error))/2 # Compute new alpha
        alphas.append(curAlpha) # you will need to append the new alpha

        # Adaboost step (4)
        falsevote = np.where(vote != labels)[0]
        wOld = wCur

        for i in correctvote:
            wCur[i] = wOld[i] * np.exp(-curAlpha)
        for i in falsevote:
            wCur[i] = wOld[i] * np.exp(curAlpha)
        wCur /= np.sum(wCur)
        
        # alphas.append(alpha) # you will need to append the new alpha
        # ==========================

        
    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        for index, classifier in enumerate(classifiers):
            classified = classifier.classify(X)
            for i in range(Npts):
                votes[i][classified[i]] += alphas[index]

        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)
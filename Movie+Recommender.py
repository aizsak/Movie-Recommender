import pandas as pd
import numpy as np
from math import sqrt

dfRatings = pd.read_csv("ml-latest-small/ratings.csv", usecols = range(3))
dfMovies = pd.read_csv("ml-latest-small/movies.csv", usecols = range(2))

nusers = dfRatings.userId.nunique()
nmovies = dfRatings.movieId.nunique()

#We split the dataset into training and test sets by removing a fixed percent of ratings per user.
#In a future version I will also include a validation set.
def splitTrainTest(df):
    
    train = pd.DataFrame(columns=df.columns)
    test = pd.DataFrame(columns=df.columns)
    for i in df.userId.unique():
        row = df.loc[df['userId'] == i]
        trainRow = row.sample(frac = 0.8)
        train = train.append(trainRow,ignore_index=True)
        testRow = row.drop(trainRow.index)
        test = test.append(testRow,ignore_index=True)
        
    #Our train and test dataframes now likely don't include every movie, add missing movies back in 
    for movie in df.movieId.unique():
        if movie not in train.movieId.values:
            train = train.append({'userId':1,'movieId':movie, 'rating': 0}, ignore_index=True)
        if movie not in test.movieId.values:
            test = test.append({'userId':1,'movieId':movie, 'rating': 0}, ignore_index=True)
    train = train.pivot(index='movieId', columns='userId', values='rating')
    test = test.pivot(index='movieId', columns='userId', values='rating')
    
    train.fillna(value=0, inplace=True)
    test.fillna(value=0, inplace=True)
    return train.to_numpy(), test.to_numpy()

train, test = splitTrainTest(dfRatings)

def initModel(nfeatures, train):
    #Creates the matrices that define our model, setting them initially to small random values.

    #Inputs: nfeatures, an int for the number of features our model will use
    #        train, a numpy array of our training data

    #Theta is our features-user matrix and X will be our movies-features matrix. Our  estimates will come from 
    #the matrix X*Theta. First we initialize them.

    Theta = np.random.uniform(size = (nfeatures, train.shape[1]), low=0, high=0.3)
    X = np.random.uniform(size = (train.shape[0], nfeatures), low=0, high=0.3)
    rated = train.copy()
    rated[rated>0] = 1
    lam = 0.1
    return Theta, X, rated

def cost(Theta, X, train, rated, lam):
    #This is the cost function with regularization
    
    regCost = (np.sum(np.square(Theta)) + np.sum(np.square(X))) * lam/2.0
    return np.sum(np.square(np.multiply(np.matmul(X, Theta) - train, rated))) + regCost

def grad(Theta, X, train, lam):
    #Finds the gradients and returns Theta and X after subtracting the gradients weighted by the learning rate.
    
    nmovies, nusers = train.shape
    Thetagrad = np.zeros(Theta.shape)
    Xgrad = np.zeros(X.shape)
    for i in xrange(nmovies):
        idx = train[i,:].nonzero()
        ratedUF = Theta[:, idx[0]]
        ratings = train[[i],idx[0]]
        Xgrad[i,:] = np.matmul((np.matmul(X[[i],:], ratedUF) - ratings), ratedUF.transpose()) + lam*X[i,:]
        
    for j in xrange(nusers):
        idx = train[:,j].nonzero()
        ratedMF = X[idx[0],:]
        ratings = train[idx[0], [j]]
        ratings.shape = (ratings.shape[0], 1)
        error = np.matmul(ratedMF,Theta[:,[j]]) - ratings
        Thetagrad[:,[j]] =np.matmul( ratedMF.transpose(), error) + lam * Theta[:,[j]]
        
    return Xgrad, Thetagrad

def fit(epochs,Theta, X, train, rated, lam, alpha, log = False):
    #fit takes our initialized Theta, X and hyperparameters and number of epochs and returns X and Theta fitted
    #by gradient descent to the training data. Also returns an array to show how the cost function changes as we change
    #X and Theta.
    
    costLog = []
    for e in xrange(epochs):
        
        if e%50 == 0:
            curCost = cost(Theta, X, train, rated, lam)
            costLog.append(curCost)
        Xgrad, Thetagrad = grad(Theta, X, train, lam)
        X = X - alpha*Xgrad
        Theta = Theta - alpha *Thetagrad
    if not log:
        return X, Theta
    return X, Theta, costLog

def tuneParams(ntrials, train, test):
    #Here I use random search to tune the hyperparameters
    
    minError = float("inf")
    params = None
    rated = train.copy()
    rated[rated>0] = 1
    bestT = None
    bestX = None
    
    for _ in xrange(ntrials):   
        #Theta is our features-user matrix and X will be our movies-features matrix. Our  estimates will come from 
        #the matrix X*Theta. Here we initialize them.
        
        nfeatures = int(np.random.uniform(10,150))
        Theta = np.random.uniform(size = (nfeatures, train.shape[1]), low=0, high=0.3)
        X = np.random.uniform(size = (train.shape[0], nfeatures), low=0, high=0.3)
        lam = np.random.uniform(0,1)/nfeatures
        alpha = np.random.uniform(0.00001, 0.0005)
        fittedX, fittedT = fit(50,Theta, X, train, rated, lam, alpha)
        error = rmse(fittedT, fittedX, test)
        
        if error < minError:
            minError = error
            params =[nfeatures, lam, alpha]
            bestT = fittedT
            bestX = fittedX
            
    return bestT, bestX, params, minError

T, X, params, err = tuneParams(20, train, test)


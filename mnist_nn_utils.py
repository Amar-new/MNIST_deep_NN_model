import pandas as pd
import numpy as np

def load_train_test_dataset():
    train=pd.read_csv("Dataset/mnist_train.csv")      #shape(60000x785) -- one column is the label column
    test=pd.read_csv("Dataset/mnist_test.csv")        #shape(10000x785)
    
    print("-------------Train Dataframe:---------\n\n", train.head())
    print("\n\n--------------Test Dataframe:---------\n\n", test.head(),"\n\n")

    #Create X_train and Y_train
    #X_train--(60000,784)    Y_train--(60000,)
    #Y_train ---- need to be developed to a row
    X_train, Y_train= train.iloc[:,1:785], train.iloc[:,0]
    print("\nDataframe X_train shape",X_train.shape)
    print("Dataframe Y_train shape",Y_train.shape)

    #Create X_test and Y_test
    #X_test--(10000,784)    Y_test--(10000,)
    #Y_test ---- need to be developed to a row
    X_test, Y_test= test.iloc[:,1:785], test.iloc[:,0]
    print("\nDataframe X_test shape",X_test.shape)
    print("Dataframe Y_test shape",Y_test.shape)

    #Convert Train Dataframe to Numpy 
    X_train=X_train.to_numpy()
    Y_train=Y_train.to_numpy()
    #Reshape X_train & Y_train
    X_train=X_train.transpose()
    Y_train=Y_train.transpose().reshape((1,60000))
    print("\nNumpy array X_train shape: ",X_train.shape)
    print("NUmpy array Y_train shape: ",Y_train.shape)

    #Convert Test Dataframe to Numpy
    X_test=X_test.to_numpy()
    Y_test=Y_test.to_numpy()
    #Reshape X_test & Y_test
    X_test=X_test.transpose()
    Y_test=Y_test.transpose().reshape((1,10000))
    print("\nNumpy array X_test shape: ",X_test.shape)
    print("NUmpy array Y_test shape: ",Y_test.shape)
    
    return X_train, Y_train, X_test, Y_test

def softmax(Z):
    '''############ Softmax activation function ###########'''
    '''The function returns two values A and cache. The first to be used for forward propagation and the later to be used during Backprop'''
    
    t=np.exp(Z)
    A=t/sum(t)
    
    cache = Z
    
    return A, cache

def relu(Z):
    '''############ Relu Activation Function #############'''
    
    A = np.maximum(0,Z)
    
    assert (A.shape==Z.shape)
    
    cache = Z
    
    return A, cache

def relu_backward(dA, cache):
    ''' ########### Backprop for Single RELU unit ##############'''
    
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0 ########### when Z<=0, we must put the value of dZ=0 as well #################
    
    assert(dZ.shape == Z.shape)
    
    return dZ

def softmax_backward(dA, cache):
    '''############### Backprop for Single SoftMax Unit #############'''
    
    Z = cache
    
    t=np.exp(Z)
    s=t/sum(t)
    
    dZ = dA*s*(1-s)
    
    return dZ
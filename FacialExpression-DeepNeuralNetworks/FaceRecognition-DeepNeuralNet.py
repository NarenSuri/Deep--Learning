# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:22:59 2017
@author: Naren Suri
"""

# This is a face expression recognition projecct
# goal is to use the Logistic, DNN, and CNN methodologies with softmax to classify between one of the 7 face exppressions
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
# load the packages / libraries / modules

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np

# this code is to classify the face expressions just by using the Logistic regression with the softmax

class LogisticSoftMax(object):
    
    def __init__(self):
        # any defualt params to set
        pass

HiddenLayers  = 150
learning_rate=10e-7
reg=10e-7
epochs=200
best_validation_error = 1
costs=[]

def LoadData():
    # this data is collected from the kaggle competetion - link shared in our intial proposal report
    # https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge    
    # implementing the logistic regression, and the data or int pixels are considered as the numpy array with spatial information of the image sacrified
    # The iamges are 48 X 48 pixel
    Header = 1 # Data has the Header
    X_Features = []
    Y_predict =[]
    
    # read the source data file
    filepath = "C:/Users/nsuri/Downloads/cvData/fer2013.csv"
    for imageRecord in open(filepath):
        if Header == 1:
            Header = 0
            #intentionally  skipping the first row, to avoid processing the header
            
        else:
            Currentfeature=[]
            imageCurrentRow = imageRecord.split(',')
            # get the x feature and the Y prediction values
            features = imageCurrentRow[1].split()
            for f in features:
                Currentfeature.append(int(f))
            X_Features.append(Currentfeature)
            
            #X_Features.append([int(p) for p in imageCurrentRow[1].split()])
            
            pred = imageCurrentRow[0]
            Y_predict.append(int(pred))
            
    # after all the rows of images is collected, convert this is numpy array, that makes our job easy to work on the data
    # normalize the data, that helps in better number ranges from 0 to 1, and would help in predection
            
    X_Features =  np.array(X_Features)  / 255.0
    Y_predict = np.array(Y_predict)
    
    return X_Features,Y_predict
            
def LoadValidationData():
    # this data is collected from the kaggle competetion - link shared in our intial proposal report
    # https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge    
    # implementing the logistic regression, and the data or int pixels are considered as the numpy array with spatial information of the image sacrified
    # The iamges are 48 X 48 pixel
    Header = 1 # Data has the Header
    X_Features_Validation = []
    Y_predict_Validation =[]
    
    # read the source data file
    filepath = "C:/Users/nsuri/Downloads/cvData/fer2013_publicTest.csv"
    for imageRecord in open(filepath):
        if Header == 1:
            Header = 0
            #intentionally  skipping the first row, to avoid processing the header
            
        else:
            Currentfeature=[]
            imageCurrentRow = imageRecord.split(',')
            # get the x feature and the Y prediction values
            features = imageCurrentRow[1].split()
            for f in features:
                Currentfeature.append(int(f))
            X_Features_Validation.append(Currentfeature)
            
            #X_Features_Validation.append([int(p) for p in imageCurrentRow[1].split()])
            
            pred = imageCurrentRow[0]
            Y_predict_Validation.append(int(pred))
            
    # after all the rows of images is collected, convert this is numpy array, that makes our job easy to work on the data
    # normalize the data, that helps in better number ranges from 0 to 1, and would help in predection
            
    X_Features_Validation =  np.array(X_Features_Validation)  / 255.0
    Y_predict_Validation = np.array(Y_predict_Validation)
    
    return X_Features_Validation,Y_predict_Validation

       
def ClassImbalanace(X_Features,Y_predict):
    # this is used to solve the class imbalance problem, which improves the prediction
    # for example im balancing only the class 1, we can look for all other classes which are less in proportin and we can balance them
    # seperate alll the classes with 1 and others
    xNotOfClass1 = X_Features[Y_predict!=1,:]
    yNotOfClass1 = Y_predict[Y_predict!=1]
    
    xOfClass1 = X_Features[Y_predict==1]
    #yOfClass1 = Y_predict[Y_predict==1,:]
    
    # now there are different ways to solve the imbalance problem, either selecting from the same group that is selecting samples from the same set
    # or find the distribution of the data, and try to generate the data from that distribution using the bayesian approach
    # and few other methods
    
    # but im using the repeatitive seelction process.
    XNew = np.repeat(xOfClass1,10,axis=0)
    
    X_Features = np.vstack([xNotOfClass1,XNew])
    Y_predict = np.concatenate((yNotOfClass1,[1]*len(XNew)))
    
    return X_Features,Y_predict

def DeepNeuralNet(X_Features,Y_predict,X_Features_Validation,Y_predict_Validation):
    global HiddenLayers
    print("Started Deep Neaural Net learning..........")
    ## lets shuffle the loaded data 
    X_Features, Y_predict = shuffle(X_Features, Y_predict)
    # now lets create the matrices required for the deep neaural net. 
    # to create the number of hidden layers and the data matrix in the linear algebra form, lets create everything in the vectors form
    NumberOfImages, D_NumberOfFeatureColumns = X_Features.shape
    K_classesOfPrediction = len(set(Y_predict))
    
    # creating an Indicator one-hottencoding scheme for this data
    Y_Indicator = np.zeros((len(Y_predict), len(set(Y_predict))))
    for i in xrange(len(Y_predict)):
        Y_Indicator[i, Y_predict[i]] = 1    
    T_targetClassOfPrediction = Y_Indicator
    
    ## Now lets create the weights randomly from a gaussian process to initialize the weights
    ## Im using the weights for hidden layers and the weights at the outputlayer seperately and also the biases seperately
    ## initalize the weights - gaussian weights and also rmember to normalize
    HiddenLayersWeights = np.random.randn(D_NumberOfFeatureColumns,HiddenLayers) / np.sqrt(D_NumberOfFeatureColumns+HiddenLayers)
    OutPutLayerWeights =  np.random.randn(HiddenLayers,K_classesOfPrediction) / np.sqrt(HiddenLayers+K_classesOfPrediction)
    ## setting the bias. But initially giving the bias as zero. However during the back propgation these bias terms will be updated
    BiasForHiddenLayers = np.zeros(HiddenLayers)
    BiasForOutPutLayer = np.zeros(K_classesOfPrediction)  
    ## lets calcualte the cost function; that is how much the cost is getting updated for every forward and backward
    ## at the interest of time, I will not save every iteration cost value, but i will store the value at every kth epochs iteration
    global costs
    global best_validation_error
    
    
    ######
    ## Now we have data and the weights and bias for the forward propagationm and bias,weights update during back-propagation
    ### Forward propagation
    ## but since the forward and backward propagation is to be done epoch times, lets put it in the loop
    for i in xrange(epochs):        
        ### Forward propagation        
        Forward_prop_result = forwardPropagation(X_Features,HiddenLayersWeights,BiasForHiddenLayers)
        # now lets caclulate the lastlayer results; that is convert the forward prop results in to the soft-max K class classification
        LastlayerResults = Forward_prop_result.dot(OutPutLayerWeights) + BiasForOutPutLayer
        # lets apply soft max to the last layer results
        FinalClassificationResults = softmax(LastlayerResults)
        ### finished with the forward propagation and got the results
        
        ## lets calcualte the error between predicted and the observed
        errorInForwardProp =  FinalClassificationResults - T_targetClassOfPrediction
        ### now indorder to update the weights we will do a backward prop with corrected weights
        ## the beauty of the backwrd propagation is usign the chain partial derivative to solve the change or dependency of various parameters involved in the forward (Z, A, W) = refer books for these terms and proof
        # updating weights from backward to forward
        # look for prrof from Dr. Minje Kim material - Indiana university  
        ### Gradient descent
        OutPutLayerWeights = OutPutLayerWeights - learning_rate*(Forward_prop_result.T.dot(errorInForwardProp) + reg*OutPutLayerWeights)
        BiasForOutPutLayer = BiasForOutPutLayer - learning_rate*(errorInForwardProp.sum(axis=0) + reg*BiasForOutPutLayer)
        # im using the RELu, you can use tanh function too
        dZ = errorInForwardProp.dot(OutPutLayerWeights.T) * (Forward_prop_result > 0) # relu 
        ## output layer weights and bias updated
        ## now lets update the weights of all hidden layers
        HiddenLayersWeights = HiddenLayersWeights - learning_rate*(X_Features.T.dot(dZ) + reg*HiddenLayersWeights)
        BiasForHiddenLayers = BiasForHiddenLayers - learning_rate*(dZ.sum(axis=0) + reg*BiasForHiddenLayers)        
        ### now we got new weights and bias both for all hidden layers and the output layer
        ## lets repaeat the steps iteratively again and again unitl we get to a less minimized cost function or epoch times
        
        ### lets try to see how good we are doing at kth interval's by using the current weights. 
        ## use the weights at current iteration and do a forward prop and see how well the classification is done
        # hence we calcualte the error at this stage. We can use the cross entropy the better way to see the error
        if i % 50 == 0:
            # use the test data to validate
            ### Forward propagation        
            Forward_prop_result = forwardPropagation(X_Features_Validation,HiddenLayersWeights,BiasForHiddenLayers)
            # now lets caclulate the lastlayer results; that is convert the forward prop results in to the soft-max K class classification
            LastlayerResults = Forward_prop_result.dot(OutPutLayerWeights) + BiasForOutPutLayer
            # lets apply soft max to the last layer results
            FinalClassificationResults = softmax(LastlayerResults)
            ### finished with the forward propagation and got the results
            
            ## Now lets calcualte the cross enropy error rate
            # we are very consciely writing below 2 lines. I have selected only the values calcualted at the index of expected result index.
            # that is if the class is 2, then the index at 2 should have 1 in the indicator. if there is something else we will use the cross entropy to calculate the error
            # also, we used only the positive 1 cofficent of the cross entropy as the others in the indicator are supposed to be zero.
            N = len(Y_predict_Validation)
            costResult = -np.log(FinalClassificationResults[np.arange(N), Y_predict_Validation]).mean()
            # above costResult computation very very concise way of writing; think about it; this makes numpy powerful or linear algebra powerful for caluclations
            costs.append(costResult) # storing costs at only the ith interation
            
            #### now lets calcualte the accuracy using the general approach
            predictedResult = np.argmax(FinalClassificationResults, axis=1) # by a column which column has the max value for a given row
            error = Calcualte_errorRate(Y_predict_Validation, predictedResult)
            print "Iteration i:", i, " ||cost_function value :", costResult, " ||error Value:", error
            
            ## now lets see in the whole epochs iterations, what is the best min error we got
            if error < best_validation_error:
                best_validation_error = error
    print("best_validation_error:"+str(best_validation_error))
    plt.plot(costs)
    plt.show()        

            
def Calcualte_errorRate(Yvalid, predictedResult):
    # how many of the results were wrongly predicted or 1-correctly predicted
    ErrorVal = 1 - np.mean(Yvalid == predictedResult)
    return ErrorVal
    
def relu(x):
    # condition check return 0 or 1
    # we will pass values only greter than 0 and all others converted to 0
    return x * (x > 0)
    
def softmax(layerResults):
    # look for the proof in stanford material of Machine learning - Dr. Andrew ng
    return np.exp(layerResults)/ np.exp(layerResults).sum(axis=1,keepdims=True)

    
def forwardPropagation(xfeatures,HiddenLayersWeights,BiasForHiddenLayers):
    # Im using the ReLU here, but we can also use the tanh function
    Z_resultsUntilLastLayer = relu(xfeatures.dot(HiddenLayersWeights) + BiasForHiddenLayers)
    return Z_resultsUntilLastLayer 
       
def MainProgramStartsHere():
    # the logistic regression with soft max starts here
    # load the data
    # this data is collected from the kaggle competetion - link shared in our intial proposal report
    X_Features,Y_predict = LoadData()
    # after the data is loaded, we should solve the class imbalance
    print "Data is loaded"
    ClassImbalanace(X_Features,Y_predict)
    ## now lets load the valkidation data for testing the model
    X_Features_Validation,Y_predict_Validation = LoadValidationData()
    
    #### Now lets train the deep neural net
    DeepNeuralNet(X_Features,Y_predict,X_Features_Validation,Y_predict_Validation)
 
    
if __name__ == '__main__':
    MainProgramStartsHere()
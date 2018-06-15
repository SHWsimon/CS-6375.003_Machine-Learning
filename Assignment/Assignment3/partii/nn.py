#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:23:24 2018

@author: simonwang
"""

#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import pickle
import sys

ww=[1,2,3,4]
ww[1]=0
ww[2]=0
ww[3]=0


class NeuralNet:
    def __init__(self, train, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        #read and mark na
        raw_input = pd.read_csv(train, skipinitialspace=True, index_col=False, na_values=['?'], na_filter='?')
        # TODO: Remember to implement the preprocess method
        
        #clean data
        train_dataset, test_dataset = self.preprocess(raw_input)   
        
        #train data
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        
        #test data
        ncols1 = len(test_dataset.columns)
        nrows1 = len(test_dataset.index)
        self.X1 = test_dataset.iloc[:, 0:(ncols1 -1)].values.reshape(nrows1, ncols1-1)
        self.y1 = test_dataset.iloc[:, (ncols1-1)].values.reshape(nrows1, 1)

        #One hot encoder for y
        # 1. INSTANTIATE
        enc = preprocessing.OneHotEncoder()
        # 2. FIT
        enc.fit(self.y)
        enc.fit(self.y1)
        # 3. Transform
        self.y = enc.transform(self.y).toarray()
        self.y1 = enc.transform(self.y1).toarray()

        #
        # Find number of input and output layers from the dataset
        #
        #how many attributes x1, x2, ..., xn
        input_layer_size = len(self.X[0])
        
        #how many class
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X #all the examples
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "tanh":
            self.__tanh(self, x)
        if activation == "Relu":
            self.__Relu(self, x)
    
    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "tanh":
            self.__tanh_derivative(self, x)
        if activation == "Relu":
            self.__Relu_derivative(self, x)
            
    #
    # Sigmoid
    #

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    #
    # tanh 
    #
    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    # derivative of tanh function, indicates confidence about existing weight
    def __tanh_derivative(self, x):
        return 1-np.power(x, 2)

    #
    # ReLu 
    #
    def __Relu(self, x):    
        return np.maximum(x, 0)

    # derivative of Relu function
    def __Relu_derivative(self, x):
        return 1 * (x > 0)
    
    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self,df):
        #drop na
        df = df.dropna() 
        print(df)
        column_names = df.columns.values
        for i in column_names:
            df = df[~df[i].isin(['?'])]
          
        #transfer str to int
        column_names = df.columns.values
        for column in column_names:
            col_data = df[column];
            unique = set(col_data) #set data into number
            metadata = dict()
            #key, value
            for i, value in enumerate(unique):
                #store each value's transfer value
                metadata[value] = i                
            #modified string to int
            df = df.applymap(lambda s: metadata.get(s) if s in metadata else s)
            #print(df)
            
        ncols = len(df.columns)
        nrows = len(df.index)
        X = df.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        #print(train_dataset.iloc[:, 0:(ncols -1)].values)
        Y = df.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        normalized_X = normalize(X, axis=0)
        #standarized_X = preprocessing.scale(X)
        train_length=int(0.8*nrows)
        test_length=int(0.2*nrows)
        X_train=normalized_X[:train_length]
        X_test=normalized_X[test_length:]
        y_train=Y[:train_length]
        y_test=Y[test_length:]
        df1=pd.DataFrame(X_train);
        df2=pd.DataFrame(y_train);
        dfTrain=pd.concat([df1, df2], axis=1);
        df3=pd.DataFrame(X_test);
        df4=pd.DataFrame(y_test);
        dfTest=pd.concat([df3, df4], axis=1);
        #print(df)
        
        return dfTrain, dfTest
    
    
    def preprocess2(self,x):
        return x

    # Below is the training function

    def train(self, max_iterations, learning_rate, activation):
        for iteration in range(max_iterations):
            #Forward
            out = self.forward_pass(activation, "") #count the output
            error = 0.5 * np.power((out - self.y), 2) #E=1/2 SUM(o-t)^2
            
            #Backpropagation
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)
        
        #save training weight
        #print(self.y)
        ww[0]=self.w01
        ww[1]=self.w12
        ww[2]=self.w23
        #print("out")
        #print(out)
        afile = open(r'ww.pkl', 'wb')
        pickle.dump(ww, afile)
        afile.close()
        
        #encode the out
        out = self.reconstrct(out)
        
        #print("size of")
        #print(self.y.shape)
        #print(out.shape)
        #print("encode out")
        #print(out)
        print("Train accuracy")
        print(accuracy_score(self.y, out))
        print()
        
     #encode the output   
    def reconstrct(self, out):
        ncols = out.shape[1]
        nrows = out.shape[0]
        for i in range(nrows):
            max = 0
            idx = 0
            for j in range(ncols):
                if out[i][j] > max:
                    max = out[i][j]
                    idx = j
                out[i][j] = 0
                     
            out[i][idx] = 1
            
        return out;

    def forward_pass(self, activation, predict):
        
        #use training weight to test
        if(len(predict) != 0):
            #read file
            print("Reading weight from file:")
            file2 = open(r'ww.pkl', "rb")
            ww = pickle.load(file2)
            print(ww)
            file2.close();
            if activation == "sigmoid" : 
                # pass our inputs through our neural network
                in1 = np.dot(self.X, ww[0])
                self.X12 = self.__sigmoid(in1)
                in2 = np.dot(self.X12, ww[1])
                self.X23 = self.__sigmoid(in2)
                in3 = np.dot(self.X23, ww[2])
                out = self.__sigmoid(in3)
            if activation == "tanh" : 
                # pass our inputs through our neural network
                in1 = np.dot(self.X, ww[0])
                self.X12 = self.__sigmoid(in1)
                in2 = np.dot(self.X12, ww[1])
                self.X23 = self.__sigmoid(in2)
                in3 = np.dot(self.X23, ww[2])
                out = self.__tanh(in3)
            if activation == "Relu" : 
                # pass our inputs through our neural network
                in1 = np.dot(self.X, ww[0])
                self.X12 = self.__sigmoid(in1)
                in2 = np.dot(self.X12, ww[1])
                self.X23 = self.__sigmoid(in2)
                in3 = np.dot(self.X23, ww[2])
                out = self.__Relu(in3)
                
        else:       
            if activation == "sigmoid" : 
                # pass our inputs through our neural network
                in1 = np.dot(self.X, self.w01)
                self.X12 = self.__tanh(in1)
                in2 = np.dot(self.X12, self.w12)
                self.X23 = self.__tanh(in2)
                in3 = np.dot(self.X23, self.w23)
                out = self.__sigmoid(in3)
            if activation == "tanh" : 
                # pass our inputs through our neural network
                in1 = np.dot(self.X, self.w01 )
                self.X12 = self.__tanh(in1)
                in2 = np.dot(self.X12, self.w12)
                self.X23 = self.__tanh(in2)
                in3 = np.dot(self.X23, self.w23)
                out = self.__tanh(in3)
            if activation == "Relu" : 
                # pass our inputs through our neural network
                in1 = np.dot(self.X, self.w01 )
                self.X12 = self.__Relu(in1) #int is array
                in2 = np.dot(self.X12, self.w12)
                self.X23 = self.__Relu(in2)
                in3 = np.dot(self.X23, self.w23)
                out = self.__Relu(in3)
            
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions
    #__activation
    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        if activation == "Relu":
            delta_output = (self.y - out) * (self.__Relu_derivative(out))

        self.deltaOut = delta_output

    # TODO: Implement other activation functions
    #__activation_derivative
    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        if activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        if activation == "Relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__Relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        if activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        if activation == "Relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__Relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    # TODO: Implement other activation functions

    def compute_input_layer_delta(self, activation):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        if activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        if activation == "Relu":
            delta_input_layer = np.multiply(self.__Relu_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self):
        self.X = self.X1
        
        #Activation
        if act == '1':           
            out = self.forward_pass("sigmoid", "1")
        elif act == '2':
            out = self.forward_pass("tanh", "1")
        elif act == '3':
            out = self.forward_pass("Relu", "1")
        
        #reconstruct
        out = self.reconstrct(out)
        print("Test accuracy")
        print(accuracy_score(self.y1, out))
        return 0


if __name__ == "__main__":
    dataset = sys.argv[1];
    act = sys.argv[2];
    neural_network = NeuralNet(dataset)
    
    #Activation
    if act == '1':
        print("sigmoid:")
        neural_network.train(max_iterations = 1000, learning_rate = 0.055, activation = "sigmoid")
    elif act == '2':
         print("tanh:")
         neural_network.train(max_iterations = 1000, learning_rate = 0.05, activation = "tanh")
    elif act == '3':
         print("Relu:")
         neural_network.train(max_iterations = 1000, learning_rate = 0.05, activation = "Relu")
    
    #predict
    testError = neural_network.predict()


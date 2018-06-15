#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:23:24 2018

@author: simonwang
"""

import numpy as np
import math
import csv
import copy 
import random

class Node:
    def __init__(self, val):
        self.l = None;
        self.r = None;
        self.inter = True; #interior node?
        self.cla = False; #leaf label
        self.v = val; #store index
        self.flag =None; #use or not use
        self.depth = 0; #tree level
        self.parent = None; #parent node
        self.lrTree = None; #show is it a right or left node
        self.kill = True; #kill node - False be killed
     
    #nodeCount
    def NodeCounter(self,counter):
        if self:
            counter[0] += 1;
        if self.l:
            self.l.NodeCounter(counter);
        if self.r:
            self.r.NodeCounter(counter);

    def nodeCount(self):
        counter =np.zeros(1);
        self.NodeCounter(counter);
        return counter[0].astype('int');
    
    #leaveCount
    def leaveCounter(self,counter):
        if self.inter == False:
            counter[0] += 1;
        if self.l:
            self.l.leaveCounter(counter);
        if self.r:
            self.r.leaveCounter(counter);

    def leaveCount(self):
        counter =np.zeros(1);
        self.leaveCounter(counter);
        return counter[0].astype('int');
    
    #Pruning Kill Node
    def killNode(self,idx,step):
        if self.inter:
            if idx == step[0]: #arrive
                self.kill = False;
                step[0] += 1;
            else:
                step[0] += 1;
                self.l.killNode(idx,step);
                self.r.killNode(idx,step);
                
    #update node            
    def update(self):
        if self.kill == False:
            self.inter = False;
            self.cla = labelClass(self.data);
            self.l = None;
            self.r = None;
        if self.l:
            self.l.update();
        if self.r:
            self.r.update();
            

def entropy(dataVec):
    numEx = dataVec.shape[0];
    numTrue = 0;
    numFalse = 0;
    #read column, and sum the # of 0 and 1
    for val in dataVec:	
        if val:
            numTrue += 1;
        else:
            numFalse += 1;
             
    pTrue = numTrue/numEx;
    pFalse = numFalse/numEx;
    logTrue = 0;
    logFalse = 0;
    if pTrue == 0:
        logTrue = 0;
    else:
        logTrue = pTrue * math.log(pTrue,2);

    if pFalse == 0:
        logFalse = 0;
    else:
        logFalse = pFalse * math.log(pFalse,2);
        
    entropy = math.fabs(logTrue + logFalse);
    return entropy;

def childEntropy(attrVec, classVec):
    totalEntropy=0;
    numEx = attrVec.shape[0];
    numTrue = 0;
    numFalse = 0;
    attrVecTrue = [];
    attrVecFalse = [];
    idx=0;
    for val in attrVec:
        if val:
            attrVecTrue = np.append(attrVecTrue,classVec[idx]);     
            numTrue += 1;
        else:
            attrVecFalse = np.append(attrVecFalse,classVec[idx]);
            numFalse+= 1;
        idx +=1;
    if numTrue == 0: #Entropy=0
        totalEntropy = numFalse / numEx*entropy(attrVecFalse);
    elif numFalse == 0: #Entropy=0
        totalEntropy = numTrue / numEx*entropy(attrVecTrue);
    else:
        totalEntropy = numTrue / numEx*entropy(attrVecTrue) + numFalse / numEx*entropy(attrVecFalse);
    return totalEntropy

def findBestEntropy(data, flag):
    maxInfoGain=0;
    attrMat = data[:,0:data.shape[1]-1];
    classVec = data[:,data.shape[1]-1];
    HparentAttrIdx=-1;
    for idx in range(attrMat.shape[1]):
        if flag[idx]:
            Hparent = entropy(data[:,idx]);
            Hchild = childEntropy(attrMat[:,idx],classVec);
            infoGain = Hparent - Hchild;
            #print(Hparent, Hchild, infoGain)
            if infoGain>maxInfoGain:
                maxInfoGain = infoGain;
                HparentAttrIdx = idx;
    flag[HparentAttrIdx]=False;
    return HparentAttrIdx; #return the attributes node index

#split data
def splitData(data,idxAttr):
    attrVec=data[:,idxAttr];
    idx=0;
    idxTrue=np.zeros(data.shape[0], dtype=bool);
    idxFalse=np.zeros(data.shape[0], dtype=bool);
    
    for val in attrVec:
        if val:
            idxTrue[idx] = True;
        else:
            idxFalse[idx] = True;
        idx +=1;
   
    attrTrue = data[idxTrue,:];
    attrFalse = data[idxFalse,:];
    return attrTrue, attrFalse;

#leaf labeled
def labelClass(data):
    trueNum = 0;
    falseNum = 0;
    classVec=data[:,data.shape[1]-1];
    for val in classVec:
        if val:
            trueNum +=1;
        else:
            falseNum +=1;

    if trueNum >= falseNum:
        return True;
    else:
        return False;

def builtDecisionTree(data,flag,node):
    decAttr = findBestEntropy(data,flag); 
    if decAttr ==-1: 
        node.inter = False;
        node.cla = labelClass(data);
        return;
        
    node.l = Node(-1);
    node.l.depth = node.depth+1;
    node.l.parent = decAttr;
    node.l.lrTree = "l"
    node.v = decAttr;
    node.data = data;
    node.r = Node(-2);
    node.r.depth = node.depth+1;
    node.r.parent = decAttr;
    node.r.lrTree = "r"
    flag[decAttr] = False;
    attrTrue, attrFalse = splitData(data,decAttr);#split data to X=true & X=false
    node.flag = flag;
    flagFalse = copy.copy(flag); #flag for X=false
    flagTrue = copy.copy(flag); #flag for X=true

    # empty & entropy!=0
    if attrFalse.shape[0]!=0 and entropy(attrFalse[:,attrFalse.shape[1]-1]) != 0:
        builtDecisionTree(attrFalse, flagFalse, node.l);
    #leaf
    else: 
        node.l.inter = False;
        node.l.cla = labelClass(attrFalse);
    # empty & entropy!=0
    if attrTrue.shape[0]!=0 and entropy(attrTrue[:,attrTrue.shape[1]-1]) != 0:
        builtDecisionTree(attrTrue, flagTrue, node.r);
    #leaf
    else: 
        node.r.inter = False;
        node.r.cla = labelClass(attrTrue);

def printTree(node,header):
    if ((node.inter) and (node.parent or node.parent==0)):
        indent = "";
        for i in range(node.depth-1):
            indent += "|";
        if node.lrTree == "l":
            print (indent, header[node.parent], " = 0 : ");
        else:
            print (indent, header[node.parent], " = 1 : ");
    elif node.parent:
        indent = "";
        for i in range(node.depth-1):
            indent += "|";
        if node.lrTree == "l":
            print (indent, header[node.parent], " = 0 :",int(node.cla));
        else:
            print (indent, header[node.parent]," = 1 :", int(node.cla));
    if node.l:
        printTree(node.l, header);
    if node.r:
        printTree(node.r, header);        
        
def pruneTree(root, pruningFactor,numAttr):
    pruneNum = int(pruningFactor * numAttr);
    nodeNum = root.nodeCount();
    leaveNum = root.leaveCount();
    interNodeNum = nodeNum-leaveNum;
    pruneNodeIdx=random.sample(range(interNodeNum), pruneNum);
    newPruneNodeIdx=np.zeros(1);
    pruneLimit=[20]; #not to prune the top few layer
    
    #check prune node
    for idx in range(pruneNum):
        while pruneNodeIdx[idx] < pruneLimit[0]:
            newPruneNodeIdx=random.sample(range(interNodeNum), 1);
            pruneNodeIdx.pop(idx);
            pruneNodeIdx.insert(idx, newPruneNodeIdx[0]);
            
    #Kill node
    for idx in range(pruneNum):
        global step;
        step=np.zeros(1);
        root.killNode(pruneNodeIdx[idx],step);
    root.update();    

def verifyData(root, data):
    predictResult = np.zeros(data.shape[0]);
    actualResult = data[:,data.shape[1]-1];
    #Read each instance
    for idx,row in enumerate(data):
        predictEx = classification(root, row);
        predictResult[idx] = predictEx;

    predictResult = predictResult.astype('bool_');
    correct=0;
    for idx in range(data.shape[0]):
        if predictResult[idx]==actualResult[idx]:
            correct=correct + 1;  

    return correct/data.shape[0];

def classification(root, row):
    if root.inter:
        if row[root.v]:
            return classification(root.r, row);
        else:
            return classification(root.l, row);
    #leaf
    else:
        return root.cla;  
    
    

#Read CSV
fileTraining = "//Users/simonwang/Downloads/UTD/2018S/CS 6375.003_ML/Assignment/Assignment2/Assignment 2/partii/data_sets1/training_set.csv";
#fileTraining = "//Users/simonwang/Downloads/Python/Python_UTD/ML/data_sets2/training_set2.csv";
fileValidate = "//Users/simonwang/Downloads/UTD/2018S/CS 6375.003_ML/Assignment/Assignment2/Assignment 2/partii/data_sets1/validation_set.csv";
fileTest = "//Users/simonwang/Downloads/UTD/2018S/CS 6375.003_ML/Assignment/Assignment2/Assignment 2/partii/data_sets1/test_set.csv";

#Training data
rawData_Training = open(fileTraining)
reader = csv.reader(rawData_Training, delimiter=',', quoting=csv.QUOTE_NONE);
Temp = np.array(list(reader))
header = Temp[0,];
dataTraining = Temp[1:,].astype('bool_'); 
#print(dataTraining)

#Validation data
rawData_Validate = open(fileValidate)
reader = csv.reader(rawData_Validate, delimiter=',', quoting=csv.QUOTE_NONE);
Temp = np.array(list(reader))
dataValidation = Temp[1:,].astype('bool_');

#Testing data
rawData_Test = open(fileTest)
reader = csv.reader(rawData_Test, delimiter=',', quoting=csv.QUOTE_NONE);
Temp = np.array(list(reader))
dataTest = Temp[1:,].astype('bool_');      

       

#setup basic parameters
numAttr = header.shape[0]-1;
numTrain = dataTraining.shape[0]; 
numValid = dataValidation.shape[0];
numTest = dataTest.shape[0];
flag = np.ones(numAttr, dtype=bool); #mark each attributes usage, default false 
pruningFactor =0.2; 

 
#Execution               
flag = np.ones(numAttr, dtype=bool);
root = Node(-1);
builtDecisionTree(dataTraining,flag,root); 
print ("The structure of the tree:");
printTree(root, header)
print ();
print ();



nodeNum = root.nodeCount();
leaveNum = root.leaveCount();

TrainingDataAc = verifyData(root, dataTraining);
ValidationDataAc = verifyData(root, dataValidation);
TestingDataAc = verifyData(root, dataTest);

print ("Pre-Pruned Accuracy");
print ("-------------------------------------");
print ("Number of training instances = ",numTrain);
print ("Number of training attributes = ", numAttr);
print ("Total number of nodes in the tree = ", nodeNum);
print ("Number of leaf nodes in the tree = ",leaveNum);
print ("Accuracy of the model on the training dataset = ", TrainingDataAc);
print ();

print ("Number of validation instances = ",numValid);
print ("Number of validation attributes = ", numAttr);
print ("Accuracy of the model on the validation dataset before pruning = ", ValidationDataAc);
print ();

print ("Number of test instances = ",numTest);
print ("Number of test attributes = ", numAttr);
print ("Accuracy of the model on the test dataset before pruning = ", TestingDataAc);
print ();
print ();


pruneTree(root,pruningFactor,numAttr);

nodeNum = root.nodeCount();
leaveNum = root.leaveCount();

TrainingDataAc = verifyData(root, dataTraining);
ValidationDataAc = verifyData(root, dataValidation);
TestingDataAc = verifyData(root, dataTest);

print ("Post-Pruned Accuracy");
print ("-------------------------------------");
print ("Number of training instances = ",numTrain);
print ("Number of training attributes=", numAttr);
print ("Total number of nodes in the tree =", nodeNum);
print ("Number of leaf nodes in the tree =",leaveNum);
print ("Accuracy of the model on the training dataset = ", TrainingDataAc);
print ();

print ("Number of validation instances = ",numValid);
print ("Number of validation attributes=", numAttr);
print ("Accuracy of the model on the validation dataset after pruning = ", ValidationDataAc);
print ();

print ("Number of test instances = ",numTest);
print ("Number of test attributes=", numAttr);
print ("Accuracy of the model on the test dataset = ", TestingDataAc);
print ();

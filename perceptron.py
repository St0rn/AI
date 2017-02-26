#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:27:07 2017

@author: St0rn
"""

from numpy import dot, zeros, random, atleast_2d

class perceptron:
    # Define percetron parameters
    def __init__(self,i,o):
        
        # Let the possibility to give a name to perceptron
        self.name=str()
        # Settings Input/Output
        self.X=zeros(i+1) # Set number of Inputs + bias
        self.O=zeros(o) # Set number of Outputs
        
        # Settings Weights
        self.W=zeros((o,i+1)) # Set number of Weights 
        # Set base Weights
        self.W[...]=2*random.random(self.W.shape)-1

    # Define heavyside activation function
    def heaviside(self,x):
        return x>0

    # Define forward propagation (Output with activation function result)
    def propagate_forward(self,pattern):
         self.X[1:]=pattern
         self.O[...]=self.heaviside(dot(self.W,self.X))
         return self.O
    
    # Define weights modification
    def propagate_backward(self,rate,expected):
        self.W+=rate*dot(atleast_2d(expected-self.O),atleast_2d(self.X))
        
    # learning routine
    def supervisedLearn(self,loop,rate,pattern,expected):
        for turn in range(loop):
                i=random.randint(pattern.size/2)
                self.propagate_forward(pattern[i])
                self.propagate_backward(rate,expected[i])
                
    # Show weights
    def showWeights(self):
        print self.W[0]

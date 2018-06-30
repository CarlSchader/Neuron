import numpy as np
import os
import math

def round(x):
    for k in xrange(len(x)):
        out = x
        if out[k] > .5:
            out[k] = 1
        else:
            out[k] = 0
    return out

def sigmoid(x,derivative=False,inverse=False):
    if inverse == True:
        return -(math.log1p((1/x) - 1))
    if derivative == True:
        return sigmoid(x)*(1 - sigmoid(x))
    else:
        return 1/(1+np.exp(-x))

class neuralNet:
    def __init__(self,layerSizes):
        self.layers = len(layerSizes)
        self.layerSizes = layerSizes
        self.W = [None] * (self.layers-1)
        self.avg_dW = [None] * (self.layers-1)
        self.trainSize = 0
        for i in xrange(self.layers-1):
            self.W[i] = np.random.randn(self.layerSizes[i],self.layerSizes[i+1])

    def forward(self,X):
        self.z = [None] * (self.layers)
        self.a = [None] * (self.layers)
        self.a[0] = X
        for i in xrange(1,self.layers):
            self.z[i] = np.dot(self.a[i-1],self.W[i-1])
            self.a[i] = sigmoid(self.z[i])
        return self.a[self.layers-1]

    def cost(self,Y):
        return .5 * ( (Y - self.a[self.layers-1]) * (Y - self.a[self.layers-1]) )

#    def costPrime(self,Y):
#        dW = [None] * (self.layers-1)
#        delta = np.multiply(-(Y-self.a[self.layers-1]),sigmoid(self.z[self.layers-1],derivative=True))
#        for i in xrange(self.layers-2,-1,-1):
#            dW[i] = np.dot(self.a[i].transpose(), delta)
#            if(i==0):
#                break
#            delta = np.dot(delta, self.W[i].transpose())*sigmoid(self.z[i],derivative=True)
#        return dW
#
#    def learn(self,Y):
#        dW = self.costPrime(Y)
#        for i in xrange(self.layers-1):
#            self.W[i] -= dW[i]

    def backward(self,Y):
        dW = [None] * (self.layers-1)
        delta = np.multiply(-(Y-self.a[self.layers-1]),sigmoid(self.z[self.layers-1],derivative=True))
        for i in xrange(self.layers-2,-1,-1):
            dW[i] = np.dot(self.a[i].transpose(), delta)
            if(i==0):
                break
            delta = np.dot(delta, self.W[i].transpose())*sigmoid(self.z[i],derivative=True)
        if self.trainSize == 0:
            self.avg_dW = dW
        else:
            for i in xrange(self.layers-1):
                self.avg_dW[i] += dW[i]
        self.trainSize += 1

    def learn(self):
        for i in xrange(self.layers-1):
            self.W[i] -= (self.avg_dW[i]/self.trainSize)
        self.avg_dW = [None] * (self.layers - 1)
        self.trainSize = 0

    def save(self,directory):
        if not(os.path.exists(directory)):
            os.mkdir(directory)
        for W in xrange(self.layers-1):
            filename = directory+"/"+str(W)
            if os.path.exists(filename):
                os.remove(filename)
            np.save(filename,self.W[W])

    def load(self,directory):
        for W in xrange(self.layers-1):
            filename = directory+"/"+str(W)+".npy"
            self.W[W] = np.load(filename)


    
    

    

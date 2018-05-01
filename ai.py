# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action - nb_action
        # this will make a full connection between neurons of the input layer and neurons of the hidden layer
        # the number '30' of hidden layers was arrived on by trial an error. Gettign good results at 30.
        # here 'Full Connection'(fc) means all the neurons of the input layer will be connected to all the neurons of the hidden layer.
        self.fc1 = nn.Linear(input_size, 30)
        # this will make a full connection between neurons of the hidden layer and neurons of the output layer
        self.fc2 = nn.Linear(30, nb_action)
        
    def forward(self, state):
        #first thing to do here is to activate the hidden neurons
        #x rrepresents hidden neurons
        # we will get the hidden neurons using 'fc1' and then apply activation function on them (this iwll be the rectifier function)
        #F (functional module) contains all the functions in pytorch to implement neural network
        # 'relu' is the rectifier function we are using to activate hidden neuroms
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay
        
class ReplayMemory (Object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if(len(self.memory) > self.capacity):
            del self.memory[0];
            
    def sample(self, batch_size):
        #zip function is similar to reshape function
        #we will use it to convert (s1,a1,r1) and (s2,a2,r2) into (s1,s2)(a1,a2)(r1,r2)
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x,0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        #reward_window is a sliding window of last 100 mean rewards.
        #it is initiatlized to an empty list since we will append the mean reward at every iteration
        self.reward_window = []
        #Now we create tey neural network
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        #torch.optim module of torch contrains all the tools to perform stochastic gradient decent.
        #Adam is the optimizer we are chosing for this self driving car
        #while inititalizing the Adam optimizer we are connecting it to the neural net we have created above by using model.parameters()
        #and applying learning rate lr = 0.001; this is not too large so that the AI gets enough time to learn
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        #last_state will a vector of 5 dimensions 
        #5 dimensions are = [input signals from the 3 sensors (left, straight, right), oritntation, -orientation]
        #Next, this vector will need to be more then vector for PyTorch
        #Specifically it will need to be a torch tensor
        #Also it will need to have an additional dimension corresponding to the Batch
        #This is because altough the last state will be the input of the neural network, 
        #while working with a neural networks in general (tensorflow, keras, Pytorch) teh input vector has to be in a Batch.
        #this batch dimension will need to be the first dimension of the input
        #unsqueeze(0) will add the additional dimension to the start
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    
    
    
    
    
    
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


# Implementing Deep Q Learning

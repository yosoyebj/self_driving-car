import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#creatin architecture of NN

class Network(nn.Modules):
    
    
     def __init__(self, input_size, nb_action):
         super(Network, self).__init__()
         self.input_size = input_size
         self.nb_action = nb_action
         self.fc1 = nn.Linear(input_size, 30)# input node to hidden
         self.fc2 = nn.Linear(30, nb_action)# hidden to output node
     
     def forward(self, state):
         x = f.relu(self.fc1(state)) #activate hidden nuerones
         q_values= self.fc2(x)
         return q_values
     
         
#implement experience replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity =capacity
        self.memory=[]
     
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
        
#implementing deep q learning

class dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model =Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer =optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = f.softmax(self.model(Variable(state, volatile =True))*7)
        action = probs.multinomial()      
        return action.data[0,0]
        
           
    
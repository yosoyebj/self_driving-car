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

class Network(nn.Module):
    
    
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

class Dqn():
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
        probs = f.softmax(self.model(Variable(state, volatile =True))*0)
        action = probs.multinomial()      
        return action.data[0,0]
        
           
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs =self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs =self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = f.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tesnsor(new_signal).float().unsqueeze(0)
        #there might be some errir in the next line
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action =self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1,)
    def save(self):
        torch.save({ 'state_dict' : self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict,
                    }, 'last_brain.pth')
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("> loading check point......")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.optimizer_state_dict(checkpoint['optimizer'])
            print('done')
        else:
            print("not saved")
    
    
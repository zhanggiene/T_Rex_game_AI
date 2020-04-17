

import random
from Model import MModel
from collections import deque  #add and pop from both side 
import numpy as np
from SumTree import SumTree
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
        self.maxSize=max_size
        self.currentSize=0

    def add(self,experience):
        #experience here is the tuple containing state,action,newstate, iscrashed
        self.buffer.append(experience)
        #p=self._getpriority(error)
        #self.tree.add(p,experience)
        self.currentSize=len(self.buffer)


    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]


## we need to create one agent as agent will react to the environment. 
class tRexAgent:
    def __init__(self,num_actions):
        self.epsilon=1
        self.epsilonMin=0.001
        self.epsilonDecay=0.99
        self.discount=0.95
        self.num_actions=num_actions
        self.model=MModel(num_actions,84,84,4)
        self.memorySize=10000
        self.memory=Memory(self.memorySize)
        self.batchSize = 128
        self.model_target=MModel(num_actions,84,84,4) # not policy model,
        self.transferLearning()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim([-90,40])
        self.graph=self.ax.bar(["dont move","jump"],[0,0],width=0.4)
        plt.ion()
        plt.show() 
    def act(self,StackOfImage):
        #input is the stack of images (1,84,84,4)    So input data has a shape of (batch_size, height, width, depth), 
        # it will return numpy , then 
        actionsToTake = np.zeros([self.num_actions]) # action at t a_t[0,0]
        if  random.random() <= self.epsilon: #randomly explore an action
            #print("----------Random Action----------")
            action_index = random.randrange(self.num_actions) # it will be 0,1
            actionsToTake[action_index]=1
        else:
            q=self.model.predict(StackOfImage)
            #print(q)
            self.display(q[0][0],q[0][1])
            action_index=np.argmax(q)
            actionsToTake[action_index]=1
            if action_index:
                print("jump")
            else:
                print("dont move")
        return actionsToTake
    def remember(self,bitExperience):
        #bitExperience is in the form of (state,action,reward,next_state,crashed)
        #the states here is 1 84 84 32
        self.memory.add(bitExperience)
    def replay(self):
        loss=0
        if self.memory.currentSize<self.batchSize:
            return
        batch=self.memory.sample(self.batchSize)
        inputs = np.zeros((self.batchSize, 84, 84, 4))   # 32 84 84 4
        targets = np.zeros((self.batchSize, self.num_actions))

        for i in range(0, len(batch)):
            temp=batch[i] #get the data part
            state_t = temp[0]    # 4D stack of images 1*84*84*4
            action_t = temp[1]   #This is action index

            reward_t = temp[2]   #reward at state_t due to action_t
            state_t1 = temp[3]   #next state
            terminal = temp[4]   #wheather the agent died or survided due the action
            

            #inputs[i:i + 1] = state_t
            inputs[i]=state_t[0]  # change from 1*84*84*4 to 84*84*4

            targets[i] = self.model.predict(state_t)  # predicted q values original prediction
            actionNumber=np.argmax(targets[i])
            Q_sa = self.model_target.predict(state_t1)      #predict q values for next step
            
            if terminal:   # overhere, adjust the existing prediction based on new evidence. 
                targets[i, actionNumber] = reward_t # if terminated, only equals reward
            else:
                targets[i, actionNumber] = reward_t + self.discount * np.max(Q_sa)
        self.model.train(inputs, targets)
  


    def exploreLess(self):
        self.epsilon = max(self.epsilonMin, self.epsilon * self.epsilonDecay)
        #print("self.epsilon is ",self.epsilon)
    def saveModel(self,name):
        self.model.save(name)
    def load(self,name):
        self.model.load(name)
    def load_previousTrain(self,name):
        self.epsilon=0.2
        self.model.load(name)
        self.model_target.load(name)

    def transferLearning(self):
        self.model_target.getModel().set_weights(self.model.getModel().get_weights()) 
    def display(self,action1,action2):
        self.graph[0].set_height(action1)
        self.graph[1].set_height(action2) # it can only control one bar, not the container 
        if(action1>action2):
            self.graph[0].set_color('r')
            self.graph[1].set_color('grey')
        else:
            self.graph[1].set_color('r')
            self.graph[0].set_color('grey')

        plt.pause(0.001)

    








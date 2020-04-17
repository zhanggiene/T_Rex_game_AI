import numpy as np
import cv2
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
from PIL import Image
from io import BytesIO
import base64
from collections import deque  #add and pop from both side 
import tensorflow as tf

from matplotlib import pyplot as plt
import pickle



from keras.models import Sequential  # sequential is like a feed forward one 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from agent import tRexAgent
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ACTIONS=2
actionsToTake=np.zeros([ACTIONS])
UP=np.copy(actionsToTake)
UP[1]=1
NOTHIHNG=np.copy(actionsToTake)

class Game:
    def __init__(self):
        self.driver=webdriver.Chrome("/Users/zhangzhuyan/Downloads/chromedriver")
        self.driver.get("chrome://dino/")
        self.driver.set_window_position(x=100,y=0)
        time.sleep(1)
        self.image=None
    def jump(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
        #https://selenium-python.readthedocs.io/api.html
    def quit(self):
        self.driver.quit()
    def new_episode(self):
        self.driver.execute_script("Runner.instance_.restart()")
        return self.get_state(UP)
    def pause(self):
        return self.driver.execute_script("return Runner.instance_.stop()")
    def getScore(self):
        score_array=self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        # it will return array['0','0','0','4'] distanceMeter is a huge dictionary. 
        score = ''.join(score_array)
        return int(score)
    def isCrashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")
    def getImage(self):
        image_b64 = self.driver.execute_script("return document.getElementsByClassName('runner-canvas')[0].toDataURL()")
        ##The data returned from the toDataURL() function is a string that
        #  represents an encoded URL containing the grabbed graphical data. png here is downloaded as string with extra information in front. 
        #print(image_b64)
        #data:image/png;base64,iVBORw0K............... 

        f = BytesIO(base64.b64decode(image_b64.split(",")[1]))
        # reason :https://stackoverflow.com/questions/32428950/converting-base64-to-an-image-incorrect-padding-error
        pilimage = Image.open(f) #image f is raw file
        image = Image.new("RGB", pilimage.size, "WHITE")
        image.paste(pilimage, (0, 0),pilimage) 
        #print("size of the original iamge is ",pilimage.size)
        image = np.array(image) #need to convert to numpy array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # change to grey
        image=image[:300,:700] #crop the image y ,x   
        image = cv2.inRange(image, 70,90)   # take out cloud
        kernel = np.ones((5,5),np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) #apply errosion  followed by dialation
        #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        self.image = cv2.resize(image, (84,84))
        #cv2.imshow('image',image) #images would be stored as a numpy array in opencv2.
    
        #cv2.waitKey(25)
    def get_state(self,actions):
        
        if actions[1]==1:  #else do nothing
            self.jump()
        self.getImage()
        is_over=self.isCrashed()
        if is_over:
            reward=-100
        else:
            if actions[1]==1: #if jump
                reward=-5
            else:
                reward=1
        return self.image,reward,is_over
'''    def render(self):
        cv2.imshow('image',self.image)
        cv2.waitKey(25)'''





stack_size = 4 # We stack 4 frames

# Initialize deque with zero-images one array for each image


#achieve the maximum limit of out deque it will simply pop out the items from the opposite end. 
def stack_frames(stacked_frames,image,is_new_episode):
    if is_new_episode:
        stacked_frames.append(image)
        stacked_frames.append(image)
        stacked_frames.append(image)
        stacked_frames.append(image)
        stacked_state = np.stack(stacked_frames, axis=2)
        # transform deque object into numpy. 
        #(84,84,4)
    else:
        stacked_frames.append(image)
        stacked_state=np.stack(stacked_frames,axis=2)
    return stacked_state



Total_Episode=1000
#it is global variable here 
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 
finalReward=[]
#stack_frames is not a numpy, it is deque object 
def trainNetwork(agent,game_state):
#state means 4 stacks of frame  
    #agent.load_previousTrain("newest950")
    for episode in range(Total_Episode): #endless running
        step=0# how many step into one game
        print("episode:",episode)
        episode_rewards=0
        frame,_,crashed=game_state.new_episode()
        inputStates=stack_frames(stacked_frames,frame,True)
        state=inputStates.reshape(1,*inputStates.shape)

        while not crashed:
            #the main game programe is responsible for maintaining and updating the stack of images
            
            action= agent.act(state)
            #https://stackoverflow.com/questions/36980992/asterisk-in-tuple-list-and-set-definitions-double-asterisk-in-dict-definition
            #(1,*range(5)) will become (1,2,3,4,5)
            next_frame, reward, crashed = game_state.get_state(action)

            episode_rewards+=reward 

            next_state = stack_frames(stacked_frames, next_frame, False)
            next_state_reshape=next_state.reshape(1,*next_state.shape)
            agent.remember((state,action,reward,next_state_reshape,crashed))

            # inside the memory, state are (1,84,84,4)
            step+=1
            state=next_state_reshape
            #agent.display()
        agent.replay()
        agent.exploreLess()
        finalReward.append(episode_rewards)
        if(episode%20==0):
            agent.transferLearning()
        print("reward",episode_rewards)
        if(episode%50==0):
            print("saving the process")
            agent.saveModel("newestest"+str(episode))
            with open('reward_history'+str(episode), 'wb') as fp:
                pickle.dump(finalReward, fp)
    


stacked_frames2  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 
def play(name,agent, env):
    # load pretrained model,
    # will fail if the given path doesn't hold a valid model

    agent.load(name)
    agent.epsilon=0

    while True:
        frame,_,crashed=env.new_episode()
        inputStates=stack_frames(stacked_frames2,frame,True)
        state=inputStates.reshape(1,*inputStates.shape)

        while not crashed:
            action= agent.act(state)
            next_frame, reward, crashed = env.get_state(action)
            #cv2.imshow('image',next_frame)
            #cv2.waitKey(1)
            next_state = stack_frames(stacked_frames2, next_frame, False)
            next_state_reshape=next_state.reshape(1,*next_state.shape)
            state=next_state_reshape

        print("Crash")
        print(env.getScore())
'''
test=Game()
for i in range(10):
    print(i)
    test.new_episode()
    step=0
    while step<100:
        step+=1
        test.duck()
        time.sleep(2)
        if test.isCrashed():
            break

'''

game=Game()
agent=tRexAgent(2)
trainNetwork(agent,game)
#play("newestest950",agent,game)

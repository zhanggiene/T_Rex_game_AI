


import tensorflow as tf
from keras.models import Sequential,Model  # sequential is like a feed forward one 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Input,Lambda,Add
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras import backend as K



class MModel():
    def __init__(self,numActions,img_rows,img_cols,img_channels):
            
        '''print("Now we build the model")
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=(img_cols,img_rows,img_channels))) 
         #It's the case of the 2D convolutional layers, which need (size1,size2,channels)
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())   #means 3d neutral flatten out. might have 9800 number
        model.add(Dense(512))
        model.add(Activation('relu'))


        model.add(Dense(numActions))
        adam = Adam(lr=1e-4)
        model.compile(loss='mse',optimizer=adam)
        print("We finish building the model")
        self.model=model'''

        input1=Input(shape=(img_cols,img_rows,img_channels))
        c1=Conv2D(32, (8, 8), strides=(4, 4), padding='same',activation='relu')(input1)
        c2=Conv2D(64, (4, 4), strides=(2, 2), padding='same',activation='relu')(c1)
        c3=Conv2D(64, (3, 3), strides=(1, 1), padding='same',activation='relu')(c2)
        f1 = Flatten()(c3)



        d1=Dense(512,activation="relu")(f1)
        state_value = Dense(1)(d1)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(numActions,))(state_value)

        d2=Dense(512,activation="relu")(f1)
        action_advantage = Dense(numActions)(d2)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(numActions,))(action_advantage)
        q1=Add()([state_value,action_advantage])


        model = Model(inputs = input1, outputs = q1)
        adam = Adam(lr=1e-4)
        model.compile(loss="mse", optimizer=adam)
        print("We finish building the Model")
        self.model=model






    def predict(self,imageStack):
        #return action index of the action 
        # So input data has a shape of (batch_size, height, width, depth),1,84,84,4
        #after training , it should produce the q value for each action
        q=self.model.predict(imageStack)
        return q
    def train(self,x,y):
        loss=self.model.train_on_batch(x,y)
        return loss
    def save(self,name):
        self.model.save_weights(name,overwrite=True)
    def load(self,name):
        self.model.load_weights(name)
        print("weight loaded")
    def getModel(self):
        return self.model

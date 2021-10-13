import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler
from sklearn.preprocessing import OneHotEncoder
import json
root=r'D:\Python_data\dog_vs_cat\train'
files=os.listdir(root)
cat_files=[file for file in files if 'cat' in file]
dog_files=[file for file in files if 'dog' in file]
cat_train=np.random.choice(cat_files,size=150,replace=False)
cat_val=np.random.choice(cat_files,size=50,replace=False)
cat_test=np.random.choice(cat_files,size=50,replace=False)
dog_train=np.random.choice(dog_files,size=150,replace=False)
dog_val=np.random.choice(dog_files,size=50,replace=False)
dog_test=np.random.choice(dog_files,size=50,replace=False)

class Coder:
    def __init__(self):
        self.OneHot=OneHotEncoder(sparse=False)
        self.LE=LabelEncoder()
    def Encoder(self,y):
        y=self.LE.fit_transform(y)
        y=self.OneHot.fit_transform(y.reshape(-1,1))
        return y
    def Decoder(self,y):
        y=np.argmax(y,axis=1)
        y=self.LE.inverse_transform(y)
        return y
y_Coder=Coder()    
def data_preprocessing(cat,dog):
    x=[]
    y=[]
    for i in cat:
        img=cv2.imread(os.path.join(root,i))
        img=cv2.resize(img,(150,150))
        x.append(img)
        y.append('cat')
    for j in dog:
        img=cv2.imread(os.path.join(root,j))
        img=cv2.resize(img,(150,150))
        x.append(img)
        y.append('dog')
    x=np.array(x)
    y=y_Coder.Encoder(y)
    x,y=shuffle(x,y)
    return x,y
x_train,y_train=data_preprocessing(cat_train,dog_train)
x_val,y_val=data_preprocessing(cat_val,dog_val)
x_test,y_test=data_preprocessing(cat_test,dog_test)
def CNN_model(input_shape,categories=10):
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64,kernel_size=7,activation='relu',
                            padding="same",input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128,3,activation='relu',padding="same"),
        tf.keras.layers.Conv2D(128,3,activation='relu',padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(categories,activation="softmax")  
    ])
    return model
def exp_decay(epoch): 
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate
def accurate_curve(history):
    fig=plt.figure('hello')
    #ax1
    ax1=fig.add_subplot(2,1,1)
    ax1.set_title('accuracy')
    ax1.set_ylim(0,1)
    for i in ['accuracy','val_accuracy']:
        ax1.plot(history.epoch,history.history[i],label=i)
    ax1.grid()
    ax1.legend()
    #ax2
    ax2=fig.add_subplot(2,1,2)
    ax2.set_title('loss')
    for i in ['loss','val_loss']:
        ax2.plot(history.epoch,history.history[i],label=i)
    ax2.grid()
    ax2.legend()
    plt.show()
def model_training():
    lrate = LearningRateScheduler(exp_decay)
    model=CNN_model([150,150,3],2)
    optimizer=tf.optimizers.RMSprop()
    ES=EarlyStopping(patience=20)
    LS=LearningRateScheduler(exp_decay)
    callbacks=[ES,LS]
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    history=model.fit(x_train,y_train,\
        batch_size=100,epochs=10,\
            validation_data=(x_val,y_val),verbose=1,shuffle=True,\
                callbacks=callbacks)
    model.save('cat_dog.h5')
    accurate_curve(history)
    print('success')
if __name__ == '__main__':
    model_training()
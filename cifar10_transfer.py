import gzip
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
cifar10_dir_path=r"D:\Python_data\cifar10\cifar-10-batches-py"
data_list=os.listdir(cifar10_dir_path)
data_1_path=os.path.join(cifar10_dir_path,data_list[1])
onehot=OneHotEncoder(sparse=False)
def load_x_and_y(data_path=data_1_path):
    """
    load the data from path
    """
    with open(data_path, 'rb') as fo:
        data = pickle.load(fo,encoding='bytes')#keys [b'batch_label', b'labels', b'data', b'filenames']
    #process x
    x=data[b'data'].reshape(-1,3,32,32)
    x=x.transpose(0,2,3,1)#3,32,32-->32,32,3
    #process y
    y_hot=onehot.fit_transform(np.array(data[b'labels']).reshape(-1,1))
    return x,y_hot
def create_model(input_shape,num_classes):
    resnet=ResNet50(weights='imagenet',include_top=False,input_shape=input_shape)
    model=tf.keras.Sequential([
        resnet,
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ],name='resnet50_transfer')
    model.summary()
    return model
def train_model(model,x_train,y_train,x_val=None,y_val=None):
    model=create_model((32,32,3),10)
    optimizer=tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',)
    history=model.fit()
    model.save()
if __name__ == '__main__':
    x,y=load_x_and_y()
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4,stratify=y)
    
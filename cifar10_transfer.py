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
    output=resnet.layers[-1].output
    base_model=tf.keras.models.Model(resnet.input,output)
    model=tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ],name='resnet50_transfer')
    model.summary()
    
    return model
if __name__ == '__main__':
    x,y=load_x_and_y()
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4,stratify=y)
    create_model((32,32,3),10)
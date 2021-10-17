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
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ],name='resnet50_transfer')
    model.summary()
    return model
def loss(y_true,y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def train_model(model,x_train,y_train,x_val=None,y_val=None):
    optimizer=tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy','AUC'])
    history=model.fit(x_train,y_train,batch_size=32,epochs=10,validation_data=(x_val,y_val))
    accurate_curve(history)
    model.save('cifar10_transfer.h5')
    print('ok')
if __name__ == '__main__':
    x,y=load_x_and_y()
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4,stratify=y)
    model=create_model((32,32,3),10)
    train_model(model,x_train,y_train,x_test,y_test)
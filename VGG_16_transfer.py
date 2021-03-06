import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from cat_vs_dog import x_train,x_val,y_train,y_val,accurate_curve,y_Coder
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler
import cv2
import os
input_shape=(150,150,3)
def VGG_model(input_shape=input_shape):
    """
    freeze the imagenet weight and create model
    """
    vgg=vgg16.VGG16(include_top=False,\
        weights='imagenet',input_shape=input_shape)
    output=vgg.layers[-1].output
    output=tf.keras.layers.Flatten()(output)
    vgg_model=Model(vgg.input,output)
    vgg_model.trainable=False
    for layer in vgg_model.layers:
        layer.trainable=False
    vgg_model.summary()
    return vgg_model
'''
import pandas as pd
layers=[(layer,layer.name,layer.trainable) for layer in vgg_model.layers]
data=pd.DataFrame(layers,columns=['layer','layer.name','layer.trainable'])
'''

def VGG_inherit_model(input_shape=input_shape,categories=2):
    """
    get the vgg last layer shape and create full_connection layer
    """
    new_model=tf.keras.Sequential([
        VGG_model(input_shape=input_shape),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(categories,activation='sigmoid'),
    ],name='VGG_HERIT')
    new_model.summary()
    return new_model
def exp_decay(epoch): 
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate
def VGG_inherit_model_training(input_shape=input_shape):
    model=VGG_inherit_model(input_shape,categories=2)
    optimizer=tf.optimizers.RMSprop()
    ES=EarlyStopping(patience=20)
    LS=LearningRateScheduler(exp_decay)
    callbacks=[ES,LS]
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])   
    history=model.fit(x_train,y_train,\
        batch_size=10,epochs=50,\
            validation_data=(x_val,y_val),verbose=1,shuffle=True,\
                callbacks=callbacks)
    model.save('VGG_inherit.h5')
    accurate_curve(history)
    
"""
take pictures into VGG_model to get flatten layer,which
is the features of the pictures.
"""
def load_data(x_pred_dirpath,resize=input_shape[:2]):
    """
    x_pred_path:directory
    """
    """
    create x_test
    """
    x_pred_list=os.listdir(x_pred_dirpath)
    x_test=[]
    for file in x_pred_list:
        file_path=os.path.join(x_pred_dirpath,file)
        img=cv2.imread(file_path)
        img=cv2.resize(img,resize)
        x_test.append(img)
    x_test=np.array(x_test)
    return x_pred_list,x_test
def tranfer_model_predict(model_path='VGG_inherit.h5',y_gold=None):
    x_list,x_test=load_data('test_data')
    """
    load model to predict.
    """
    model=tf.keras.models.load_model(model_path)
    y_pred_onehot=model.predict(x_test)
    y_pred=y_Coder.inverse_transform(y_pred_onehot)
    dic={}
    for i,j in zip(x_list,y_pred.tolist()):
        dic[i]=j
    return dic
def draw_img(x_pred,n):
    fig=plt.figure('hello')
    for i in range(1,n+1):
        ax=fig.add_subplot(int(pow(n,0.5)),int(pow(n,0.5)+1),i)
        ax.imshow(x_pred[0][0,:,:,i],cmap='bone')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
def see_the_black_box(model_path='VGG_inherit.h5',y_gold=None):
    x_list,x_test=load_data('test_data')
    model=tf.keras.models.load_model(model_path)
    layer_outputs=[layer.output for layer in model.layers[0].layers[1:9]]
    slice_model=Model(inputs=model.layers[0].layers[1].input,outputs=layer_outputs)  
    x_pred=slice_model.predict(x_test)
    draw_img(x_pred,55)

    
if __name__ == '__main__':
    #VGG_inherit_model_training() #training model
    see_the_black_box()
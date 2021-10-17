from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cat_vs_dog import x_train,y_train,x_val,y_val
import matplotlib.pyplot as plt
import numpy as np
"""
create more data by rotating, shear, zoom
"""
train_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.3,rotation_range=50,width_shift_range=0.2,shear_range=0.4,horizontal_flip=True)
val_datagen=ImageDataGenerator(rescale=1./255)
train=train_datagen.flow(x_train,y_train,batch_size=32) #n*(x_shape*batch,y_shape*batch)
val=val_datagen.flow(x_val,y_val,batch_size=32)
"""
model.fit(x=x_train,y=y_train,validation_data=(x_val,y_val))
-->model.fit_generator(generator=train,validation_data=val)
"""
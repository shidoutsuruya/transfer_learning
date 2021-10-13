from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
import tensorflow as tf
input_shape=(224,224,3)
vgg=vgg16.VGG16(include_top=False,\
    weights='imagenet',input_shape=input_shape)
output=vgg.layers[-1].output
output=tf.keras.layers.Flatten()(output)
vgg_model=Model(vgg.input,output)
vgg_model.trainable=False
for layer in vgg_model.layers:
    layer.trainable=False
vgg_model.summary()


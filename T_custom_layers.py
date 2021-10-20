import tensorflow as tf
import numpy as np
"""
create own's layer
"""
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self,num_outputs):
        super(MyDenseLayer,self).__init__()
        self.num_outputs=num_outputs
    def build(self,input_shape):
        self.kernel=self.add_weight('kernel',\
            shape=[int(input_shape[-1]),self.num_outputs])
    def call(self,inputs):
        return tf.matmul(inputs,self.kernel)
    
#layer=MyDenseLayer(10) #init
#layer(np.zeros((20,5)))#call
class ResnetIndentityBlock(tf.keras.Model):
    def __init__(self,kernel_size,filters):
        super(ResnetIndentityBlock,self).__init__()
        filters1,filters2,filters3=filters
        self.conv2a=tf.keras.layers.Conv2D(filters1,kernel_size=(1,1))
        self.bn2a=tf.keras.layers.BatchNormalization()
        
        self.conv2b=tf.keras.layers.Conv2D(filters2,kernel_size,padding='same')
        self.bn2b=tf.keras.layers.BatchNormalization()
    
        self.conv2c=tf.keras.layers.Conv2D(filters3,kernel_size=(1,1))
        self.bn2c=tf.keras.layers.BatchNormalization()
    
    def call(self,input_tensor,training=False):
        x=self.conv2a(input_tensor)
        x=self.bn2a(x,training=training)
        x=tf.nn.relu(x)
        
        x=self.conv2b(x)
        x=self.bn2b(x,training=training)
        x=tf.nn.relu(x)
        
        x=self.conv2c(x)
        x=self.bn2c(x,training=training)
        
        x+=input_tensor
        return tf.nn.relu(x)
    
#block=ResnetIdentityBlock(1,[1,2,3])
#block(np.zeros((1,2,3,3)))
        
        

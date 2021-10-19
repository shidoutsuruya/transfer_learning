from tensorflow.keras.applications import resnet50,inception_v3
from tensorflow.keras import backend
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import cv2
#backend.set_learning_phase(0)

model=resnet50.ResNet50(weights='imagenet',include_top=True)
def preprocessing_image(input_shape=model.input_shape,image_path='sakura.jpg'):
    img=image.load_img(image_path)
    img=image.img_to_array(img,dtype='uint8')#(1650, 2500, 3)
    img=cv2.resize(img,input_shape[1:3])#(224,224,3)
    img=np.expand_dims(img,axis=0)#(1,224,224,3)
    return img
def model_pred():
    img=preprocessing_image()
    preds=model.predict(img)
    for n,label,prob in decode_predictions(preds)[0]:
        print(label,prob)
def new_model_pred():
    output_layers=[layer.output for layer in model.layers if layer.name.startswith('activation_')]
    layer_names=[layer.name.startswith('activation_') for layer in model.layers]
    new_model=Model(inputs=model.input,outputs=output_layers)
    img=preprocessing_image()
    activations=new_model.predict(img)
    return activations
def draw(layer_idx,activations=new_model_pred()):
    image_per_row=8
    layer_activation=activations[layer_idx]
    print('layer shape:',layer_activation.shape)
    n_features=layer_activation.shape[-1]
    r=layer_activation.shape[1]
    c=layer_activation.shape[2]
    n_cols=n_features//image_per_row
    display_grid=np.zeros((r*n_cols,image_per_row*c))
    for col in range(n_cols):
        for row in range(image_per_row):
            channel_image=layer_activation[0,:,:,col*image_per_row+row]
            #post-process the feature to make it visually palatable
            channel_image-=channel_image.mean()
            channel_image/=channel_image.std()
            channel_image*=64
            channel_image+=128
            channel_image=np.clip(channel_image,0,255).astype('uint8')
            display_grid[col*r:(col+1)*r,row*c:(row+1)*c]=channel_image
    scale=1./r
    fig=plt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0])) 
    img=plt.imshow(display_grid,aspect='auto',cmap='plasma')
    fig.colorbar(img)
    plt.show()

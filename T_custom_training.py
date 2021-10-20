from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
import tensorflow as tf
import numpy as np
x=load_iris()['data']
y=load_iris()['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,stratify=y)
model=tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation=tf.nn.relu,input_shape=(4,)),
    tf.keras.layers.Dense(10,activation=tf.nn.relu),
    tf.keras.layers.Dense(3,activation=tf.nn.softmax)
])
loss_func=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model,x,y_true,training=True):
    y_pred=model(x)
    return loss_func(y_true,y_pred)
def grad(model,inputs,targets):
    with tf.GradientTape(persistent=True) as tape:
        loss_value=loss(model,inputs,targets)
    return loss_value,tape.gradient(loss_value,model.trainable_variables)

#training process
def training():
    train_loss_results=[]
    train_accuracy_results=[]
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
    num_epochs=201
    batch_size=32
    for epoch in range(num_epochs):
        epoch_loss_avg=tf.keras.metrics.Mean()
        epoch_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
        x_batch=tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
        y_batch=tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size)
        for x,y in zip(x_batch,y_batch):
            loss_value,grads=grad(model,x,y)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))#key
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y,model(x))
        train_loss_results.append(epoch_loss_avg.result())
        train_loss_results.append(epoch_accuracy.result())
        if epoch%50==0:
            print('epoch{:03d}:loss:{:.3f},accuracy:{:.3f}'.format(epoch,epoch_loss_avg.result(),epoch_accuracy.result()))
#predict
training()
y_pred=np.argmax(model.predict(x_test),axis=1)
accuracy=tf.keras.metrics.Accuracy()

print(accuracy(y_pred,y_test))

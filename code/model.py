from tensorflow import keras
from tensorflow.keras.applications.mobilenet import MobileNet

from keras.layers import Dropout, GlobalAveragePooling2D, Conv2D, Reshape, Activation, InputLayer
from keras import Sequential
from keras.models import load_model
from keras.optimizers import RMSprop


from math import prod

import numpy as np
# from tensorflow.keras.models import load_model
from prints import printw, printc, printo, printe
import matplotlib.pyplot as plt
from subprocess import call
# from onnx_tf.backend import prepare
# import onnx


model_path = 'models/my_model'
onnx_path = 'models/model.onnx'

def summarize_diagnostics(history, epochs):
    for h in history.history:
        printc(h)
    e = range(1, epochs+1)
    fig, ax, = plt.subplots(1,2)
    ax[0].plot(e, history.history['loss'], color='blue', label='train')
    ax[0].plot(e, history.history['val_loss'], color='orange', label='test')
    ax[0].title.set_text('Loss function')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()

    ax[1].plot(e, history.history['acc'], color='blue', label='train')
    ax[1].plot(e, history.history['val_acc'], color='orange', label='test')
    ax[1].title.set_text('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    # ax[1].axhline(y=.6, color='r')
    # plt.savefig('res.pdf')
    plt.show()

def train(model, X, Y, batch_size=50, epochs=10, validation_split=0.2):

    # X = np.reshape(X, (X.shape[0], prod(X.shape[-3:])))

    history = model.fit(
        x=X,
        y=Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
    )
    return history, epochs

def mobilenet_create():
    mobilenet = MobileNet(
                    weights="imagenet", # Use ImageNet weights
                    include_top=False # Don't include FC-layer/classifiers
        )
    # mobilenet.summary()

    for l in mobilenet.layers:
        l.trainable = False

    return mobilenet

def model_create(d_shape, n_cats=9, lr=2e-4):


    # Create my own classifier to train
    # inp_dim = prod(d_shape[-3:])

    model = Sequential(name="hullifier_0.01")

    model.add(GlobalAveragePooling2D(keepdims=True))
    model.add(Dropout(0.01))
    model.add(Conv2D(filters=n_cats, kernel_size=(1,1), padding='same', activation='linear'))
    model.add(Reshape((n_cats,)))
    model.add(Activation('sigmoid'))

    model.build(d_shape)

    model.compile(
        optimizer = RMSprop(learning_rate=lr), # 2e-4
        loss = 'binary_crossentropy',
        metrics=['acc']
    )
    model.summary()
    
    
    return model

    
def model_load(tf=True):
    if tf: 
        return load_model(model_path)
    # return prepare(onnx.load(onnx_path))

    

def model_save(model):
    model.save(model_path)
    call(f"python3 -m tf2onnx.convert --saved-model {model_path} --output {onnx_path}", shell=True)
    
def main():
    mobilenet = mobilenet_create()
    model = model_create((1,7,7,1024))

    # history = model.fit()
    # -1.107 mm Is good 
    # - 1.102 mm 
    return

if __name__ == "__main__":
    main()
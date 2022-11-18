from tensorflow import keras
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet

from keras.layers import Dropout, GlobalAveragePooling2D, Conv2D, Reshape, Activation, Dense, Resizing, Rescaling, RandomFlip, RandomRotation
from keras import Sequential
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.metrics import BinaryAccuracy


import numpy as np
from prints import printw, printc, printo, printe
import matplotlib.pyplot as plt


model_path = 'models/my_model'
onnx_path = 'models/model.onnx'

def aug_model_create():
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
    ])
    return data_augmentation

def preproc_model_create(resize=False, target_size=224):
    preproc_model = Sequential()
    if resize:
        preproc_model.add(Resizing(target_size,target_size)) 
    preproc_model.add(Rescaling(1./127.5, offset=-1))
    
    return preproc_model 


def summarize_diagnostics(history, epochs, version='', **kwargs):
    e = range(1, epochs+1)
    fig, ax, = plt.subplots(1,2)
    fig.suptitle("Loss and accuracy graphs")
    ax[0].plot(e, history.history['loss'], color='blue', label='train')
    ax[0].plot(e, history.history['val_loss'], color='orange', label='test')
    ax[0].title.set_text('Loss function')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()

    ax[1].plot(e, history.history['binary_accuracy'], color='blue', label='train')
    ax[1].plot(e, history.history['val_binary_accuracy'], color='orange', label='test')
    ax[1].title.set_text('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    ax[1].set_ylim([0.75, 0.94])

    params = f'_epochs-{epochs}'

    for k,v in kwargs.items():
        params += f'_{k}-{v}'

    fig.savefig('../out_imgs/loss_acc'+ params+ '.pdf')
    fig.savefig('../out_imgs/loss_acc'+ params+ '.png')
    # plt.show()

def train(model, X, Y, batch_size=50, epochs=10, v_split=0.2):

    history = model.fit(
        x=X,
        y=Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=v_split,
    )
    return history, epochs

def mobilenet_create(v2=False):
    if v2:
        mobilenet = MobileNetV2(include_top=False)
    else:
        mobilenet = MobileNet(include_top=False) # Don't include FC-layer/classifiers)
    # mobilenet.summary()
    for l in mobilenet.layers:
        l.trainable = False
    return mobilenet

def model_create(d_shape, n_cats=9, v2=False):
    if v2:
        model = Sequential(name="hullifier_0.02")

        model.add(GlobalAveragePooling2D())
        model.add(Dense(n_cats, activation='softmax'))
        
    else:
        # Create my own classifier to train
        model = Sequential(name="hullifier_0.01")

        model.add(GlobalAveragePooling2D(keepdims=True))
        model.add(Dropout(0.01))
        model.add(Conv2D(filters=n_cats, kernel_size=(1,1), padding='same', activation='linear'))
        model.add(Reshape((n_cats,)))
        model.add(Activation('sigmoid'))

    # model.build(d_shape)

    # model.compile(
    #     optimizer = RMSprop(learning_rate=lr), # 2e-4
    #     loss = 'binary_crossentropy',
    #     metrics=[BinaryAccuracy()]
    # )
    # model.summary()
    
    
    return model

def hullifier_load(v2=False, resize=False):
    hullifier = load_model(model_path + ('_v2' if v2 else '_v1'))
    if resize and type(hullifier.layers[0]) != Resizing:
        hullifier = Sequential([Resizing(224,224), hullifier])
    return hullifier

        
def hullifier_create(X, n_cats=9, lr=2e-4, v2=False, resize=False):
    mobilenet = mobilenet_create(v2=v2)
    d_shape = mobilenet.predict(X).shape
    classifier = model_create(d_shape, n_cats, v2)

    hullifier = Sequential()
    hullifier.add(preproc_model_create(resize))
    # hullifier.add(aug_model_create())
    hullifier.add(mobilenet)
    hullifier.add(classifier)
    hullifier.compile(
        optimizer = RMSprop(learning_rate=lr), # 2e-4
        loss = 'binary_crossentropy',
        metrics=[BinaryAccuracy()]
    )
    return hullifier


def hullifier_save(model, postfix, **kwargs):
    model.save(model_path + postfix)
    params = f'\n'

    for k,v in kwargs.items():
        params += f'\n{k} = {v}'

    with open(model_path + postfix + '/params.txt', 'w+') as f:
        f.write(params) 
    
    
def main():
    # mobilenet = mobilenet_create()
    mobilenet = mobilenet_create(
        v2=True
        )
    model = model_create(
        # (1,7,7,1024),
        (1,7,7,1280), v2=True
    )

    # history = model.fit()
    # -1.107 mm Is good 
    # - 1.102 mm 
    return

if __name__ == "__main__":
    main()
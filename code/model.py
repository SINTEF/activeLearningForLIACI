import tensorflow as tf
from keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from math import prod
from keras import Sequential, optimizers
import numpy as np
# from tf.keras import Model
from prints import printw, printc
import matplotlib.pyplot as plt

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

def model_create(d_shape, n_cats=10, lr=2e-4):

    print("\nCode starts here\n=======================\n")

    # Create my own classifier to train
    inp_dim = prod(d_shape[-3:])

    model = Sequential(name="hullifier_0.01")
    # model.add(Input(shape=inp_dim))
    model.add(Input(shape=d_shape[-3:]))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_cats, activation='sigmoid'))
    
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=lr), # 2e-4
        loss = 'binary_crossentropy',
        metrics=['acc']
    )
    model.summary()
    
    return model

def mobilenet_create():
    mobilenet = MobileNet(
                    weights="imagenet", # Use ImageNet weights
                    include_top=False # Don't include FC-layer/classifiers
                )
    mobilenet.summary()
    for l in mobilenet.layers:
        l.trainable = False

    return mobilenet
    
def run_model():
    pass

def main():
    mobilenet = mobilenet_create()
    model_create()
    # history = model.fit()

    # -1.107 mm Is good 
    # - 1.102 mm 
    return

if __name__ == "__main__":
    main()
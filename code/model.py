import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential, optimizers
import numpy as np
# from tf.keras import Model

def labels_get():
    labels = [
        "anode",
        "bilge keel",
        "corrosion",
        "defect",
        "marine growth",
        "overboard valve",
        "paint peel ",
        "propeller",
        "sea-chest grating",
    ]
    n_labels = len(labels)
    return labels, n_labels
    
def model_create():

    mobilenet = MobileNet(
                    weights="imagenet", # Use ImageNet weights
                    include_top=False # Don't include FC-layer/classifiers
                )
    for l in mobilenet.layers:
        l.trainable = False

    # mobilenet.summary()

    print("\nCode starts here\n=======================\n")

    # Create my own classifier to train

    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=7*7*512))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=2e-4),
        loss = 'categorical_crossentropy',
        metrics=['acc']
        )
    return model
def run_model():
    pass

def main():
    model_create()
    # history = model.fit()

    # -1.107 mm Is good 
    # - 1.102 mm 
    return

if __name__ == "__main__":
    main()
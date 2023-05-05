from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet import MobileNet

from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Conv2D, Reshape, Activation, Dense, Resizing, Rescaling, RandomFlip, RandomRotation
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import BinaryAccuracy


import numpy as np
from prints import printw, printc, printo, printe
import matplotlib.pyplot as plt
from utils.file import get_dict_from_file
import utils.config as cnf
from utils.model import preproc_model_create, aug_model_create, mobilenet_create, classifier_create, hullifier_compile,hullifier_load, hullifier_save, get_lr

model_path = 'models/my_model'
onnx_path = 'models/model.onnx'




def summarize_diagnostics(history, epochs, path='../out_imgs/loss_acc', **kwargs):
    e = range(1, epochs+1)
    fig, ax, = plt.subplots(1,2)
    ax[0].plot(e, history.history['loss'], color='blue', label='train')
    ax[0].plot(e, history.history['val_loss'], color='orange', label='test')
    ax[0].title.set_text('Loss function')
    ax[0].set_xlabel('Epoch')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(e, history.history['binary_accuracy'], color='blue', label='train')
    ax[1].plot(e, history.history['val_binary_accuracy'], color='orange', label='test')
    ax[1].title.set_text('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylim([0.75, 0.94])
    
    params = f'epochs={epochs}'

    for k,v in kwargs.items():
        params += f', {k}={v}'

    fig.suptitle(f"Loss and accuracy graph\n{params}")
    fig.tight_layout()

    params = 'graph_' + params.replace(', ', '_').replace('=', '-')
    
    fig.savefig(path+ params + '.pdf')
    fig.savefig(path+ params + '.png')
    # plt.show()


def hullifier_create(X=None, n_cats=9, lr=2e-4, v2=False, resize=False):
    pre_proc = preproc_model_create(resize)
    augmentation = aug_model_create()
    mobilenet = mobilenet_create(v2=v2)
    if X is None:
        d_shape = (None,7,7,1024)
    else:
        d_shape = mobilenet.predict(X).shape

    classifier = classifier_create(d_shape, n_cats, v2)

    hullifier = Sequential(name = 'Hullifier_v2')
    for l in pre_proc.layers:
        hullifier.add(l)
    for l in augmentation.layers:
        hullifier.add(l)
    for l in mobilenet.layers:
        hullifier.add(l)
    for l in classifier.layers:
        hullifier.add(l)

    hullifier_compile(hullifier, lr)
 
    return hullifier

# clones the old hullifier into a new one
# PS this also has no sub sequentiasls
def hullifier_clone(hullifier=None):
    """ hullifier: if None then the old hullifier is loaded 
    """
    if hullifier is None:
        params = get_dict_from_file(cnf.model_path + 'params.txt')
        hullifier = hullifier_load(cnf.model_path)
        lr = float(params.lr)
    else:
        lr = get_lr(hullifier)
        
    # model = clone_model(hullifier)
    model = hullifier_create(lr=lr)
    hullifier_compile(model, lr=lr)
    model.build((None, 224, 224, 3) )
    model.set_weights(hullifier.get_weights())
    return model


    
def main():
    pass
    mobilenet = mobilenet_create()
    # mobilenet = mobilenet_create(
    #     v2=True
    #     )
    # model = classifier_create(
    #     # (1,7,7,1024),
    #     (1,7,7,1280), v2=True
    # )
    # mb = MobileNet(include_top=False)
    # mb.save('models/mobilenet/')
    # mb = MobileNetV2(include_top=False)
    # mb.save('models/mobilenet_v2/')
    # history = model.fit()
    # -1.107 mm Is good 
    # - 1.102 mm 

if __name__ == "__main__":
    main()
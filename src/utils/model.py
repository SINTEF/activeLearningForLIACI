from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet import MobileNet

from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Conv2D, Reshape, Activation, Dense, Resizing, Rescaling, RandomFlip, RandomRotation
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import BinaryAccuracy

def get_lr(model):
    return model.optimizer.get_config()['learning_rate']

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


def mobilenet_create(v2=False):
    if v2:
        mobilenet = MobileNetV2(include_top=False)
        # mobilenet = load_model('/models/mobilenet/')
    else:
        mobilenet = MobileNet(include_top=False) # Don't include FC-layer/classifiers)
        # mobilenet = load_model('/models/mobilenet_v2/')
    # mobilenet.summary()
    for l in mobilenet.layers:
        l.trainable = False
    return mobilenet

def classifier_create(d_shape=(1,7,7,1024), n_cats=9, v2=False, dr_rate=0.01):
    if v2:
        model = Sequential(name="hullifier_0.02")

        model.add(GlobalAveragePooling2D())
        model.add(Dense(n_cats, activation='sigmoid'))
        
    else:
        # Create my own classifier to train
        model = Sequential(name="hullifier_0.01")

        model.add(GlobalAveragePooling2D(keepdims=True))
        model.add(Dropout(dr_rate))
        model.add(Conv2D(filters=n_cats, kernel_size=(1,1), padding='same', activation='linear'))
        model.add(Reshape((n_cats,)))
        model.add(Activation('sigmoid'))
    
    return model

def hullifier_compile(model, lr=2e-4):
    model.compile(
        optimizer = RMSprop(learning_rate=lr), 
        loss = 'binary_crossentropy',
        metrics=[BinaryAccuracy()]
    )
def hullifier_load(path, resize=False):
    print(f'Trying to load model from: {path}')
    hullifier = load_model(path)
    if resize and type(hullifier.layers[0]) != Resizing:
        hullifier = Sequential([Resizing(224,224), hullifier])
    return hullifier

def hullifier_save(model, path, **kwargs):
    model.save(path)
    params = ''
    for k,v in kwargs.items():
        params += f'{k}={v}\n'

    with open(path + '/params.txt', 'w+') as f:
        f.write(params) 
        
def train(model, X, Y, validation_data, batch_size=50, epochs=10):

    history = model.fit(
        x=X,
        y=Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
    )
    return history, epochs
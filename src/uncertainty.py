import numpy as np
from tensorflow.keras.models import clone_model
from tensorflow import convert_to_tensor, expand_dims
import scipy.stats as stats
import time
import matplotlib.pyplot as plt
from PIL import Image


import utils.config as cnf
from prints import printc
from data import load_from_coco, shuffle_data, split_data, get_cat_lab
from model import hullifier_clone
from utils.model import hullifier_load, hullifier_compile, get_lr, classifier_create, preproc_model_create
from utils.file import get_dict_from_file 
from utils.utils import get_predictions_bool
from utils.uncertainty import find_uncertainty, get_feature_extractor

measure_time = True
if measure_time:
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    printc(f'Started at: {current_time}')
    start = time.time()

def t():
    tm = int(time.time()-start)
    return f'{tm//60}m {tm%60}s'

# [x] Load old model
# [x] Clone old model
# [x] Set dropout layer to higher D/O rate for new model
# [x] Load old weights to new model
# [x] Load test data
# [x] predict(use __call__) with training=true for n (10 maybe?) images from test set
# [x] calculate mu and sigma
# [x] plot gauss
# [x] plot predictions along x axis
# [] 
# [] 
# [] 

def get_classifier(model):
    lr = get_lr(model)

    # Create classifier
    classifier = classifier_create(dr_rate=0.1)
    hullifier_compile(classifier, lr) # Compile
    classifier.build((None, 7, 7, 1024)) # and Build
    classifier.set_weights(model.get_weights()[-len(classifier.get_weights()):]) # Set current models weights
    return classifier


def measure_uncertainty(model, X, n_samples):
    pre_proc = preproc_model_create()
    fe = get_feature_extractor(model)
    classifier = get_classifier(model)
    # labels = get_cat_lab()
    b_preds, preds = get_predictions_bool(model, X)

    uncertainties = np.empty((X.shape[0], n_samples, 9)) # shape (n_images, n_samples, 9)
    
    for i, im in enumerate(X):
        im = expand_dims(convert_to_tensor(im), 0)
        im = pre_proc(im)
        features = fe(im,training=False)
        for k in range(n_samples):
            pred = classifier(features, training=True)[0]
            uncertainties[i, k] = pred
    

    # im = Image.fromarray(X[frame])
    # im.save('../out_imgs/uncertainty/'+ 'image_1.png')
    
    with open('npy/'+'u.npy', 'wb') as f:
        np.save(f, uncertainties)
        np.save(f, b_preds)
        np.save(f, preds)
        np.save(f, X) # Save images sample data comes from


def main():
    # Load parameters from old model
    params = get_dict_from_file(cnf.model_path + 'params.txt')
    # load old model
    hullifier = hullifier_load(cnf.model_path)
    # New cloned model with same weights
    model = hullifier_clone(hullifier)

    ## Now find uncertainty
    
    # Load all the old data
    X, Y = load_from_coco()
    X, Y = shuffle_data(X,Y)
    X,Y,XT,YT = split_data(X,Y,float(params.v_split))

    # Don't set to < 2
    n_unc_im = 4
    # n_samples = cnf.n_samples
    n_samples = 350

    measure_uncertainty(model, XT[51:51+n_unc_im], n_samples)





if __name__ == "__main__":
    main()
    if measure_time:
        print(f'Total time used: {t()}')

    pass
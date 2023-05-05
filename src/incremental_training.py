import numpy as np
from tensorflow.keras.models import clone_model
import matplotlib.pyplot as plt
from time import time

import utils.config as cnf
from prints import printe, printo
from model import hullifier_create
from utils.model import train
from data import load_from_coco, shuffle_data, split_data
# from utils import get_dict_from_file, plot_history, find_uncertainty, history_merge
from utils.file import get_dict_from_file
from utils.utils import history_merge
from utils.plt import plot_history

measure_time = True
if measure_time:
    start = time()
def t():
    tm = int(time()-start)
    return f'{tm//60}m {tm%60}s'
# [x] Create NEW hullifier
# [x] Load old data
# [x] split data into original split
# [x] Split train data into smaller splits
# [x] Train incrementally on parts of data and compare final result to original result
# [] 
# [] Seperate data with uncertain images and non uncertain images
# [] Train on only uncertain images and train on only certain images
# [] Compare results
# []
# []
# []

base_dir = 'npy/incr/'


def incrementally_train(X, Y, XT, YT, lr, epochs, image_budget, fraction, do_all):
    cum_bud = np.cumsum(image_budget)
    incr_epochs = int(epochs * fraction)
    eps = np.full(image_budget.shape[0], incr_epochs, dtype=int)
    eps[0] = epochs # First epoch is initial training phase and is larger
    model = hullifier_create(X[:3], lr=lr)    

    full_hist = []
    prev=0
    for cb, ep in zip(cum_bud, eps):

        if do_all:
            X_train = X[:cb]
            Y_train = Y[:cb]
        else:
            print(f'training on idx[{prev}:{cb}]')
            X_train = X[prev:cb]
            Y_train = Y[prev:cb]
        h, e = train(
            model,
            X_train,
            Y_train,
            batch_size=50,
            epochs=ep,
            validation_data=(XT,YT),
        )
        prev=cb
        full_hist.append(h)
        
        
    full_hist = history_merge(full_hist, eps)

    return full_hist, eps
        
def regular_train(X, Y, XT, YT, lr, epochs):
    # Train a new model the regular way
    for seed in range(10):
        model = hullifier_create(lr=lr)
        h, e = train(
            model,
            X,
            Y,
            batch_size=50,
            epochs=epochs,
            validation_data=(XT,YT),
        )
        
        h = history_merge([h],[e])
        e = np.array([e])
        fn = 'regular'+str(seed)

        save_values(base_dir+'../mean_reg/', fn, e, h)
        printe(f'Finished training {fn} in {t()}')


def save_values(path_to_dir, file_name, e, h):
    path = path_to_dir + file_name +'.npy'
    print(f'Trying to save values to {path}...')
    with open(path, 'wb') as f:
        np.save(f, e)
        np.save(f, h)
        
def incrementally_train_span(X, Y, XT, YT, lr, epochs, image_budget, to, do_all):
    if do_all:
        typ = 'all'
    else:
        typ = 'new'

    for fraction in np.linspace(0.1, to/10, to):
        fn = 'f' + str(fraction.round(2))
        fh, eps = incrementally_train(X, Y, XT, YT, lr, epochs, image_budget, fraction, do_all)
        printe(f'Finished training {fn} in {t()}')
        save_values(base_dir+typ+'/', fn, eps, fh)

def main():
    # Load parameters from old model
    params = get_dict_from_file(cnf.model_path + 'params.txt')
    # load old model
    params.epochs = int(params.epochs)
    params.epochs += 5
    # params.epochs += 5
    # params.epochs = 5
    # params.epochs += 5
    
    # Load all the old data
    X, Y = load_from_coco()
    X, Y = shuffle_data(X,Y)
    # Original split
    X,Y,XT,YT = split_data(X,Y,float(params.v_split))    

    regular_train(X, Y, XT, YT, float(params.lr), params.epochs)

    image_budget = np.array([
        800, 
        100, # 1
        100, # 2
        100, # 3
        100, # 4
        100, # 5
        100, # 6
        100, # 7
        100, # 8
        103  # 9
        ])

    
    do_all = True
    to = 6
    incrementally_train_span(X, Y, XT, YT, float(params.lr), params.epochs, image_budget, do_all=True, to=5)
    
    do_all = False
    to = 10
    incrementally_train_span(X, Y, XT, YT, float(params.lr), params.epochs, image_budget, do_all=do_all, to=to)
    

if __name__ == "__main__":
    
    main()
    if measure_time:
        print(f'Total time used: {t()}')

    pass
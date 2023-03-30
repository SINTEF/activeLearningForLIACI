import numpy as np
from tensorflow.keras.models import clone_model
import matplotlib.pyplot as plt
from time import time

import config as cnf
from prints import printe
from model import hullifier_load, hullifier_compile, hullifier_create, train
from utils import get_dict_from_file, plot_history
from data import load_from_coco, shuffle_data, split_data


# [] Create NEW hullifier
# [] Load old data
# [] split data into original split
# [] Split train data into smaller splits
# [] Train incrementally on parts of data and compare final result to original result
# [] 
# []
# []
# []
# []
# []
# []
def history_merge(hists, eps, loss_measurement, measurement):

    full_hist = np.empty((4, np.sum(eps)))
    cum_e = np.cumsum(eps)

    prev = 0
    for h, e, ce in zip(hists, eps, cum_e):
        full_hist[0, prev:ce] = np.array(h.history[loss_measurement])
        full_hist[1, prev:ce] = np.array(h.history['val_' + loss_measurement])
        full_hist[2, prev:ce] = np.array(h.history[measurement])
        full_hist[3, prev:ce] = np.array(h.history['val_' + measurement])

        prev = ce
    return full_hist

# def incrementally_train(X, Y, epochs):


def main():
    # Load parameters from old model
    params = get_dict_from_file(cnf.model_path + 'params.txt')
    # load old model
    params.epochs = int(params.epochs)
    # params.epochs = 5
    
    # Load all the old data
    X, Y = load_from_coco()
    X, Y = shuffle_data(X,Y)
    X,Y,XT,YT = split_data(X,Y,float(params.v_split))

    images_remain = X.shape[0] 

    model = hullifier_create(X[:3], lr=float(params.lr))

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

    cum_bud = np.cumsum(image_budget)
    incr_epochs = int(params.epochs * 0.2)
    total_epochs =  ((image_budget.shape[0]-1) * incr_epochs) + params.epochs
    eps = np.full(image_budget.shape[0], incr_epochs, dtype=int)
    eps[0] = params.epochs
    cum_e = np.cumsum(eps)
    

    fig, axs = plt.subplots(1,2)
    
    full_hist = []

    prev_i = 0
    
    for cb, ep in zip(cum_bud, eps):

        h, e = train(
            model,
            X[:cb],
            Y[:cb],
            batch_size=50,
            epochs=ep,
            validation_data=(XT,YT),
        )

        full_hist.append(h)
        
        
    full_hist = history_merge(full_hist, eps, 'loss', 'binary_accuracy')
    for ce in cum_e:
        axs[0].axvline(ce, linestyle='dashed', linewidth=1,c='red')
        axs[1].axvline(ce, linestyle='dashed', linewidth=1,c='red')
    plot_history(axs[0], total_epochs, full_hist[:2], subfig_text='Loss function')
    plot_history(axs[1], total_epochs, full_hist[2:], subfig_text='Binary accuracy')
    
    fig.savefig('../out_imgs/'+'test_fig.png')

    # Train a new model the regular way
    model = hullifier_create(X[:3], lr=float(params.lr))
    h, e = train(
        model,
        X,
        Y,
        batch_size=50,
        epochs=params.epochs,
        validation_data=(XT,YT),
    )
    fig, axs = plt.subplots(1,2)

    plot_history(axs[0], e, h, 'loss', 'Loss function')
    plot_history(axs[1], e, h, 'binary_accuracy', 'Binary accuracy')
    fig.savefig('../out_imgs/'+'b.png')

    fig, axs = plt.subplots(1,2)
    # Plot both models, un-normalized
    ## Plotting regular model
    plot_history(axs[0], e, h, measurement='loss', subfig_text='Loss function')
    plot_history(axs[1], e, h, measurement='binary_accuracy', subfig_text='Binary accuracy')

    ## Plotting incremental model
    plot_history(axs[0], total_epochs, full_hist[:2], subfig_text='Loss function')
    plot_history(axs[1], total_epochs, full_hist[2:], subfig_text='Binary accuracy')
    fig.savefig('../out_imgs/'+'c.png')

    fig, axs = plt.subplots(1,2)
    # Plot both models, un-normalized

    ## Plot regular model normalized
    x = np.linspace(0,1, params.epochs)
    axs[0].plot(x, h.history['loss'], label='train')
    axs[0].plot(x, h.history['val_loss'], label='test')
    axs[1].plot(x, h.history['binary_accuracy'], label='train')
    axs[1].plot(x, h.history['val_binary_accuracy'], label='test')

    ## Plot incremental model normalized
    x = np.linspace(0, 1, total_epochs)
    axs[0].plot(x, full_hist[0], label='incr. train')
    axs[0].plot(x, full_hist[1], label='incr. test')
    axs[1].plot(x, full_hist[2], label='incr. train')
    axs[1].plot(x, full_hist[3], label='incr. test')

    ## Set fig info
    axs[0].title.set_text('Loss function')
    axs[1].title.set_text('Binary accuracy')
    axs[0].set_xlabel('Total training period')
    axs[1].set_xlabel('Total training period')
    axs[0].grid()
    axs[1].grid()
    axs[0].legend()
    axs[1].legend()
    fig.savefig('../out_imgs/'+'d.png')



    


    




    




    
        
    
    

if __name__ == "__main__":
    measure_time = True
    if measure_time:
        start = time()
    main()
    if measure_time:
        end = time()
        t = int(end-start)
        print(f'Time used: {t//60}m {t%60}s')

    pass
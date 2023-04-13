import numpy as np
from tensorflow.keras.models import clone_model
import matplotlib.pyplot as plt
from time import time

import config as cnf
from prints import printe, printo
from model import hullifier_load, hullifier_compile, hullifier_create, train
from data import load_from_coco, shuffle_data, split_data
from utils import get_dict_from_file, plot_history, find_uncertainty, history_merge

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



def incrementally_train(X, Y, XT, YT, lr, epochs, image_budget):
    cum_bud = np.cumsum(image_budget)
    incr_epochs = int(epochs * 0.2)
    total_epochs =  ((image_budget.shape[0]-1) * incr_epochs) + epochs
    eps = np.full(image_budget.shape[0], incr_epochs, dtype=int)
    eps[0] = epochs # First epoch is initial training phase and is larger
    cum_e = np.cumsum(eps)
    model = hullifier_create(X[:3], lr=lr)    

    fig, axs = plt.subplots(1,2)
    
    full_hist = []
        
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
    axs[0].set_ylim(top=0.4)
    axs[1].set_ylim(bottom=0.81)

    fig.suptitle(f"Loss and accuracy graph for model trained incrementally")
    fig.tight_layout()
    fig.savefig('../out_imgs/incremental_learning/'+'incr_incrementally_trained.png')
    if measure_time:
        printo(f'1st fig done after {t()} since start')

    # Train a new model the regular way
    model = hullifier_create(X[:3], lr=lr)
    h, e = train(
        model,
        X,
        Y,
        batch_size=50,
        epochs=epochs,
        validation_data=(XT,YT),
    )
    fig, axs = plt.subplots(1,2)

    plot_history(axs[0], e, h, 'loss', 'Loss function')
    plot_history(axs[1], e, h, 'binary_accuracy', 'Binary accuracy')

    ## Set axs info
    axs[0].set_ylim(top=0.4)
    axs[1].set_ylim(bottom=0.81)
    
    ## Set figk info
    fig.suptitle(f"Loss and accuracy graph for\nmodel trained traditionally")
    fig.tight_layout()
    fig.savefig('../out_imgs/incremental_learning/'+'incr_regular_trained.png')
    if measure_time:
        printo(f'2nd fig done after {t()} since start')

    fig, axs = plt.subplots(1,2)
    # Plot both models, un-normalized
    ## Plotting regular model
    plot_history(axs[0], e, h, measurement='loss', subfig_text='Loss function')
    plot_history(axs[1], e, h, measurement='binary_accuracy', subfig_text='Binary accuracy')

    ## Plotting incremental model
    plot_history(axs[0], total_epochs, full_hist[:2], subfig_text='Loss function')
    plot_history(axs[1], total_epochs, full_hist[2:], subfig_text='Binary accuracy')

    ## Set axs info
    axs[0].set_ylim(top=0.4)
    axs[1].set_ylim(bottom=0.81)

    fig.suptitle(f"Loss and accuracy graph for\nboth models")
    fig.tight_layout()
    fig.savefig('../out_imgs/incremental_learning/'+'incr_both_un-normalized.png')
    if measure_time:
        printo(f'3rd fig done after {t()} since start')
    
    # Plot regular model normalized
    fig, axs = plt.subplots(1,2)

    ## Plotting regular model normalized
    x = np.linspace(0,1, epochs)
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

    ## Set axs info
    axs[0].title.set_text('Loss function')
    axs[1].title.set_text('Binary accuracy')
    axs[0].set_xlabel('Total training period')
    axs[1].set_xlabel('Total training period')
    axs[0].grid()
    axs[1].grid()
    axs[0].legend()
    axs[1].legend()
    axs[0].set_ylim(top=0.4)
    axs[1].set_ylim(bottom=0.81)

    fig.suptitle(f"Loss and accuracy graph for\nboth models, normalized")
    fig.tight_layout()
    fig.savefig('../out_imgs/incremental_learning/'+'incr_both_normalized.png')
    if measure_time:
        printo(f'5th fig done after {t()} since start')

def main():
    # Load parameters from old model
    params = get_dict_from_file(cnf.model_path + 'params.txt')
    # load old model
    params.epochs = int(params.epochs)
    # params.epochs += 5
    # params.epochs += 5
    # params.epochs += 5
    # params.epochs = 5
    
    # Load all the old data
    X, Y = load_from_coco()
    X, Y = shuffle_data(X,Y)
    # Original split
    X,Y,XT,YT = split_data(X,Y,float(params.v_split))    

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
    incrementally_train(X, Y, XT, YT, float(params.lr), params.epochs, image_budget)
    

if __name__ == "__main__":
    
    main()
    if measure_time:
        print(f'Total time used: {t()}')

    pass
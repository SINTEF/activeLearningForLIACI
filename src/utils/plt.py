from itertools import product
import numpy as np

def show_bar_value(ax):
    for bars in  ax.containers:
        ax.bar_label(bars)
        
def get_axs_iter(axs):
    return list(product(np.arange(axs.shape[0]), np.arange(axs.shape[1])))


def plot_history(ax, e, h, measurement=None, subfig_text=None, label='train', label_t='test'):
    """ Takes inn ax from a subplots fig """
    e = range(1, e+1)
    if str(type(h)) == "<class 'keras.callbacks.History'>":
        ax.plot(e, h.history[measurement], label='train')
        ax.plot(e, h.history['val_'+ measurement], label='test')
    else:
        ax.plot(e, h[0], label=label)
        ax.plot(e, h[1], label=label_t)

    ax.title.set_text(subfig_text)
    ax.set_xlabel('Epoch')
    ax.grid(visible=True)
    ax.legend()

def ax_plot_training(ax, eps, h, subfig_text=None, label_pf='', normalized=True, variation=0, lsr=1):
    """ variation: if 1 only plot train, if -1 only plot test, if 0 plot both
        lsr: linspace range
    """

    eps = np.linspace(0, lsr, eps) if normalized else np.arange(1, eps+1)

    if variation == 1:
        ax.plot(eps, h[0], label=label_pf+'train')
    elif variation == -1: 
        ax.plot(eps, h[1], label=label_pf+'test')
    else:
        ax.plot(eps, h[0], label=label_pf+'train')
        ax.plot(eps, h[1], linestyle='dashed', label=label_pf+'test')
    if subfig_text:
        ax.title.set_text(subfig_text)
    # ax.set_xlabel('Epoch')
    ax.grid(visible=True)
    ax.legend()
    
# set legend for all axes
def axs_legend(axs):
    for ax in axs.reshape(-1):
        ax.legend()
        
# set grid for all axes
def axs_grid(axs):
    for ax in axs.reshape(-1):
        ax.grid()

def save_fig(path, file_name, fig, tight=True):
    """ Saves fig as png and pdf """
    if tight:
        fig.tight_layout()
    fig.savefig(path+file_name+'.png')
    fig.savefig(path+'pdf/'+file_name+'.pdf')

def set_max_xlim(ax, x, min=0):
    ax.set_xlim(left=min, right=len(x)+1)

def plot_seperate_epochs(epochs, axs):
    cum_e = np.cumsum(epochs)

    for ce in cum_e: # plot lines to seperate epochs
        axs[0].axvline(ce, linestyle='dashed', linewidth=1,c='red')
        axs[1].axvline(ce, linestyle='dashed', linewidth=1,c='red')
    return cum_e[-1] # return total number of epochs
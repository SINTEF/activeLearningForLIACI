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

# set legend for all axes
def axs_legend(axs):
    for ax in axs.reshape(-1):
        ax.legend()
# set grid for all axes
def axs_grid(axs):
    for ax in axs.reshape(-1):
        ax.grid()
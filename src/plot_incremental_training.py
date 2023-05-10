import matplotlib.pyplot as plt
import numpy as np

from utils.plt import plot_history, plot_seperate_epochs, save_fig, ax_plot_training
from prints import printc, printe


ptd = 'npy/incr/'

def get_values(path_to_dir, file_name):
    path = path_to_dir+file_name+'.npy'
    print(f'opening file: {path}')
    with open(path, 'rb') as f:
        print('Loading:')
        eps = np.load(f)
        printc(f'eps: {eps.shape}')
        fh = np.load(f)
        printc(f'fh: {fh.shape}')

    return eps, fh

# def plot_regular_training(eps, fh):

#     fig, axs = plt.subplots(1,2)

    

#     plot_history(axs[0], eps, fh, 'loss')
#     plot_history(axs[1], eps, fh, 'binary_accuracy')

#     axs[0].title.set_text('Loss function')
#     axs[1].title.set_text('Binary accuracy')

#     ## Set axs info
#     axs[0].set_ylim(top=0.4)
#     axs[1].set_ylim(bottom=0.81)
    
#     ## Set fig info
#     fig.suptitle(f"Loss and accuracy graph for\nmodel trained traditionally")
#     fig.tight_layout()
#     fig.savefig('../out_imgs/incremental_learning/'+'incr_regular_trained.png')

def plot_trainings(path_to_dir, npy_fns, p_fn):
    full_hists = []

    variations = [
        '_test', 
        '', 
        '_train'
    ]
    fig_0, axs_0 = plt.subplots(2)
    fig_1, axs_1 = plt.subplots(2)
    fig_2, axs_2 = plt.subplots(2)
    figs = [fig_0, fig_1, fig_2]
    axss = [axs_0, axs_1, axs_2]
    for i, (fig, axs) in enumerate(zip(figs, axss), -1): 
        for fn in npy_fns:
            try:
                eps, fh = get_values(ptd+path_to_dir, fn)
            except:
                printe(f"Can't get {fn}")
                continue
            plot_training(axs, eps.sum(), fh, label_pf=fn+' ', variation=i)

        
        axs[0].title.set_text('Loss function')
        axs[1].title.set_text('Binary accuracy')
        axs[0].set_xlabel('Training period (normalized)')
        axs[1].set_xlabel('Training period (normalized)')
        axs[0].legend(loc = 'lower left')
        axs[1].legend(loc = 'upper left')
        if not i:
            axs[0].get_legend().remove()
            axs[1].get_legend().remove()
        fig.suptitle(f"Loss and accuracy graph for model trained incrementally")
        fig.set_figheight(7.5)
        axs[0].set_ylim(top=0.3)
        axs[1].set_ylim(bottom=0.86)
        axs[0].set_xlim(left=0, right=1)
        axs[1].set_xlim(left=0, right=1)

    for fig,v in zip(figs, variations):
        fig.tight_layout()
        save_fig('../out_imgs/incremental_learning/'+path_to_dir,p_fn+v, fig)

def plot_training(axs, total_epochs, full_hist, label_pf, variation):
    
    ax_plot_training(axs[0], total_epochs, full_hist[:2], label_pf=label_pf, variation=variation, lsr=1)
    ax_plot_training(axs[1], total_epochs, full_hist[2:], label_pf=label_pf, variation=variation, lsr=1)
    




def main():
    
    npy_fns = []
    to = 10
    for fraction in np.linspace(0.1, to/10, to):
        fn = 'f' + str(fraction.round(2))
        npy_fns.append(fn) 
    npy_fns.append('regular')
    print('going through', npy_fns)

    p_fn = 'normalized_trained'
    dir = 'all/'
    plot_trainings(dir, npy_fns, p_fn)
    dir = 'new/'
    plot_trainings(dir, npy_fns, p_fn)
    
    pass

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np

from utils.plt import plot_history, plot_seperate_epochs, save_fig, ax_plot_training
from prints import printc, printe


ptd = 'npy/incr/'
variations = [
    '_test', 
    '', 
    '_train'
]

base_dir = '../out_imgs/incremental_learning/'

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

def plot_trainings(npy_fns, end_sup_tit, path_to_dir, p_fn=''):
    full_hists = []

    
    fig_0, axs_0 = plt.subplots(2)
    fig_1, axs_1 = plt.subplots(2)
    fig_2, axs_2 = plt.subplots(2)
    figs = [fig_0, fig_1, fig_2]
    axss = [axs_0, axs_1, axs_2]
    variations_pretty = [
        'Test', 
        'Train & test', 
        'Train'
    ]
    for i, (fig, axs, v) in enumerate(zip(figs, axss, variations_pretty), -1): 
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
        
        fig.suptitle(v+f" loss and accuracy graph for models trained incrementally on "+end_sup_tit, wrap=True)
        fig.set_figheight(7.5)
        axs[0].set_ylim(top=0.3)
        axs[1].set_ylim(bottom=0.86)
        axs[0].set_xlim(left=0, right=1)
        axs[1].set_xlim(left=0, right=1)
        
    if p_fn:
        for fig,v in zip(figs, variations):
            save_fig(base_dir+path_to_dir,p_fn+v, fig)
    else:
        return figs, axss
    
def save_figs(figs, ptd, fn):
    for fig,v in zip(figs, variations):
        save_fig(base_dir+ptd,fn+v, fig)

def plot_training(axs, total_epochs, full_hist, label_pf, variation):
    
    ax_plot_training(axs[0], total_epochs, full_hist[:2], label_pf=label_pf, variation=variation, lsr=1, legend=False)
    ax_plot_training(axs[1], total_epochs, full_hist[2:], label_pf=label_pf, variation=variation, lsr=1, legend=False)

def set_global_figs_label(figs,axss, x, y):
    for fig, axs in zip(figs, axss):
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(x, y))
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
    end_sup = 'old and new training data'
    figs, axss = plot_trainings(npy_fns, end_sup, dir)
    # Test
    axss[0][0].set_ylim([0.24,0.28])
    axss[0][1].set_ylim([0.88,0.91])
    #Train
    axss[2][0].set_ylim([0.16,0.25])
    axss[2][1].set_ylim([0.89,0.935])
    set_global_figs_label([figs[0],figs[2]], [axss[0],axss[2]], 0.35, 0.6)
    
    save_figs(figs, dir, p_fn)

    dir = 'new/'
    end_sup = 'only new training data'
    figs, axss = plot_trainings(npy_fns, end_sup, dir)
    # Test
    axss[0][0].set_ylim([0.24,0.30])
    axss[0][1].set_ylim([0.87,0.905])
    # All
    axss[1][0].set_ylim([0.17,0.32])
    axss[1][1].set_ylim([0.85,0.935])
    #Train
    axss[2][0].set_ylim([0.17,0.31])
    axss[2][1].set_ylim([0.85,0.935])
    set_global_figs_label([figs[0],figs[2]], [axss[0],axss[2]], 0.32, 0.6)
    
    save_figs(figs, dir, p_fn)
    
    pass

if __name__ == "__main__":
    main()
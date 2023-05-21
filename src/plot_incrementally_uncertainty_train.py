import numpy as np
import matplotlib.pyplot as plt

# from data import get_cat_lab
import utils.config as cnf
from prints import printc
# from utils import plot_history, history_merge, axs_legend, axs_grid
from utils.plt import plot_history, axs_legend, axs_grid, plot_seperate_epochs, save_fig, set_max_xlim, ax_plot_training

base_path = '../out_imgs/incrementally_uncertain/'
def get_values(budget=''):
    
    path = 'npy/'+ 'iut'+budget + '.npy'

    print('Reading values from:', path)
    with open(path, 'rb') as f:
        eps = np.load(f)
        printc(f'eps shape: {eps.shape}')
        fh_0 = np.load(f)
        printc(f'fh_0 shape: {fh_0.shape}')
        fh_1 = np.load(f)
        printc(f'fh_1 shape: {fh_1.shape}')
        fh_2 = np.load(f)
        printc(f'fh_2 shape: {fh_2.shape}')
        fh_3 = np.load(f)
        printc(f'fh_3 shape: {fh_3.shape}')


    return eps, fh_0, fh_1, fh_2, fh_3



def plot_fig(full_hist, eps, suptitle, dir='', fn=''):
    
    fig, axs = plt.subplots(1,2)
    
    total_epochs = plot_seperate_epochs(eps, axs)
    plot_history(axs[0], total_epochs, full_hist[:2], subfig_text='Loss function')
    plot_history(axs[1], total_epochs, full_hist[2:], subfig_text='Binary accuracy')
    axs[0].set_ylim(top=0.4)
    axs[1].set_ylim(bottom=0.81)
    axs[0].title.set_text('Loss function')
    axs[1].title.set_text('Binary Accuracy')

    fig.suptitle(suptitle, wrap=True)
    fig.tight_layout()

    if dir:
        save_fig(base_path+dir, fn, fig)
    else:
        return fig, axs
    
    
    
def plot_fig_all(hists, eps, r_hist=None, r_eps=None, dir='', suptitle=''):
    """ dir: if False returns the fig and axs, else saves to dir """

    fig, axs = plt.subplots(2)
    
    # for hist in hists:
    # plot_history(axs[0], np.sum(eps), hists[0][:2], label='MC Unc train', label_t='MC Unc test')
    plot_seperate_epochs(eps, axs)

    ax_plot_training(axs[0], np.sum(eps), hists[0][:2], label_pf='MC Unc ')
    ax_plot_training(axs[1], np.sum(eps), hists[0][2:], label_pf='MC Unc ')

    ax_plot_training(axs[0], np.sum(eps), hists[1][:2], label_pf='MC Crt ')
    ax_plot_training(axs[1], np.sum(eps), hists[1][2:], label_pf='MC Crt ')
    
    ax_plot_training(axs[0], np.sum(eps), hists[2][:2], label_pf='TI Unc ')
    ax_plot_training(axs[1], np.sum(eps), hists[2][2:], label_pf='TI Unc ')

    ax_plot_training(axs[0], np.sum(eps), hists[3][:2], label_pf='TI Crt ')
    ax_plot_training(axs[1], np.sum(eps), hists[3][2:], label_pf='TI Crt ')

    if r_eps:
        lsr = hists[0].shape[-1]
        ax_plot_training(axs[0], r_eps, r_hist[:2], label_pf='Norm. reg. ', lsr=lsr)
        ax_plot_training(axs[1], r_eps, r_hist[2:], label_pf='Norm. reg. ', lsr=lsr)
    
    axs[0].title.set_text('Loss function')
    axs[1].title.set_text('Binary Accuracy')
    axs[0].set_ylim(top=0.30)
    axs[1].set_ylim(top=0.93, bottom=0.88)

    set_max_xlim(axs[0], hists[0][0])
    set_max_xlim(axs[1], hists[0][0])
    axs[0].set_xlabel('Epoch')
    axs[1].set_xlabel('Epoch')
    axs_grid(axs)
    fig.set_figheight(10)
    if suptitle:
        fig.suptitle(suptitle, wrap=True)
    # fig.tight_layout()

    if dir:
        save_fig(base_path+dir, 'all_trained', fig)
    else:
        return fig, axs

def plot_fig_all_train_test(hists, eps, r_hist=None, r_eps=None, dir='', suptitle=''):
    e = range(1, eps.sum()+1)

    fig0, axs0 = plt.subplots(2)
    
    plot_seperate_epochs(eps, axs0)
    axs0[0].plot(e, hists[0][0], label='MC Unc train')
    axs0[0].plot(e, hists[1][0], label='MC Crt train')
    axs0[0].plot(e, hists[2][0], label='TI Unc train')
    axs0[0].plot(e, hists[3][0], label='TI Crt train')

    axs0[1].plot(e, hists[0][2], label='MC Unc train')
    axs0[1].plot(e, hists[1][2], label='MC Crt train')
    axs0[1].plot(e, hists[2][2], label='TI Unc train')
    axs0[1].plot(e, hists[3][2], label='TI Crt train')
    if r_eps:
            lsr = hists[0].shape[-1]
            ax_plot_training(axs0[0], r_eps, r_hist[:2], label_pf='Norm. reg. ', lsr=lsr, variation=1)
            ax_plot_training(axs0[1], r_eps, r_hist[2:], label_pf='Norm. reg. ', lsr=lsr, variation=1)
    axs_legend(axs0)
    axs_grid(axs0)
    axs0[0].set_ylim(top=0.24)
    axs0[1].set_ylim(bottom=0.89)
    set_max_xlim(axs0[0], hists[0][0])
    set_max_xlim(axs0[1], hists[0][0])
    axs0[0].set_xlabel('Epoch')
    axs0[1].set_xlabel('Epoch')
    axs0[0].title.set_text('Loss function')
    axs0[1].title.set_text('Binary Accuracy')

    if suptitle:
        fig0.suptitle('Training '+suptitle, wrap=True)
        # fig0.suptitle('Training summary for training models with MC=Monte Carlo, TI=Threshold interval, Unc=Uncertain and Crt=Non-Uncertain images', wrap=True)
    fig0.set_figheight(10)
    
    if dir:
        save_fig(base_path+dir, 'all_trained_train', fig0)
    fig, axs = plt.subplots(2)
    plot_seperate_epochs(eps, axs)
    
    axs[0].plot(e, hists[0][1], label='MC Unc test')
    axs[0].plot(e, hists[1][1], label='MC Crt test')
    axs[0].plot(e, hists[2][1], label='TI Unc test')
    axs[0].plot(e, hists[3][1], label='TI Crt test')

    axs[1].plot(e, hists[0][3], label='MC Unc test')
    axs[1].plot(e, hists[1][3], label='MC Crt test')
    axs[1].plot(e, hists[2][3], label='TI Unc test')
    axs[1].plot(e, hists[3][3], label='TI Crt test')

    if r_eps:
        ax_plot_training(axs[0], r_eps, r_hist[:2], label_pf='Norm. reg. ', lsr=lsr, variation=-1)
        ax_plot_training(axs[1], r_eps, r_hist[2:], label_pf='Norm. reg. ', lsr=lsr, variation=-1)

    axs[0].title.set_text('Loss function')
    axs[1].title.set_text('Binary Accuracy')
    axs_legend(axs)
    axs_grid(axs)
    axs[0].set_ylim(top=0.28)
    axs[1].set_ylim(bottom=0.885)
    set_max_xlim(axs[0], hists[0][1])
    set_max_xlim(axs[1], hists[0][1])
    axs[0].set_xlabel('Epoch')
    axs[1].set_xlabel('Epoch')

    
    # fig.suptitle('Test summary for training models with MC=Monte Carlo, TI=Threshold interval, Unc=Uncertain and Crt=Non-Uncertain images', wrap=True)
    fig.set_figheight(10)
    if suptitle:
        fig.suptitle('Test '+suptitle, wrap=True)
    # fig.tight_layout()
    if dir:
        save_fig(base_path+dir, 'all_trained_test', fig)

    else:
        return fig0, axs0, fig, axs
    
def gather_values(dir):
    f0 = []
    f1 = []
    f2 = []
    f3 = []
    for seed in range(10):
        with open('npy/mean_iut/'+dir+'/iut'+str(seed)+ '.npy', "rb") as f:
            eps = np.load(f)
            fh_0 = np.load(f)
            fh_1 = np.load(f)
            fh_2 = np.load(f)
            fh_3 = np.load(f)
        f0.append(fh_0)
        f1.append(fh_1)
        f2.append(fh_2)
        f3.append(fh_3)

    f0 = np.mean(np.array(f0), axis=0)
    f1 = np.mean(np.array(f1), axis=0)
    f2 = np.mean(np.array(f2), axis=0)
    f3 = np.mean(np.array(f3), axis=0)

    return eps, f0, f1, f2, f3
 
def plot_fig_all_mean():

    re, rfh = get_values_reg_mean()

    # any
    dir = 'any/'
    eps, f0, f1, f2, f3 = gather_values(dir[:-1])
    lsr = f1.shape[-1]
    hists = [f0, f1, f2, f3]
    suptitle = 'summary for training models with MC=Monte Carlo, TI=Threshold interval, Unc=Uncertain and Crt=Non-Uncertain images'

    fig, axs = plot_fig_all(hists, eps, rfh, re)
    fig.suptitle('Mean train and test summary of 10 runs, with randomly shuffled train and test data', wrap=True)
    save_fig(base_path+dir, 'mean_all_trained', fig)


    fig0, axs0, fig, axs = plot_fig_all_train_test(hists, eps, rfh, re, suptitle=suptitle)
    # Train
    axs0[0].set_ylim(bottom=0.15,top=0.24)
    axs0[1].set_ylim(bottom=0.90,top=0.935)
    # Test
    axs[0].set_ylim(bottom=0.255)
    axs[1].set_ylim(bottom=0.88, top=0.8975)
    fig0.suptitle('Mean train summary of 10 runs, with randomly shuffled train and test data', wrap=True)
    fig.suptitle('Mean test summary of 10 runs, with randomly shuffled train and test data', wrap=True)
    
    save_fig(base_path+dir, 'mean_all_trained_train', fig0)
    save_fig(base_path+dir, 'mean_all_trained_test', fig)

    # budget
    dir = 'budget/'
    suptitle += ', budget=35'
    eps, f0, f1, f2, f3 = gather_values(dir[:-1])
    hists = [f0, f1, f2, f3]
    fig, axs = plot_fig_all(hists, eps, rfh, re)
    
    # axs[0].set_ylim(bottom=0.26)
    # axs[1].set_ylim(top=0.894)
    fig.suptitle('Mean train and test summary of 10 runs, with randomly shuffled train and test data, budget=35', wrap=True)
    save_fig(base_path+dir, 'mean_all_trained', fig)
    
    fig0, axs0, fig, axs = plot_fig_all_train_test(hists, eps, rfh, re, suptitle=suptitle)
    # Train
    axs0[0].set_ylim(bottom=0.17)
    axs0[1].set_ylim(bottom=0.90,top=0.93)
    # Test
    axs[0].set_ylim(bottom=0.255)
    axs[1].set_ylim(top=0.899)

    fig0.suptitle('Mean train summary of 10 runs, with randomly shuffled train and test data, budget=35', wrap=True)
    fig.suptitle('Mean test summary of 10 runs, with randomly shuffled train and test data, budget=35', wrap=True)
    save_fig(base_path+dir, 'mean_all_trained_train', fig0)
    save_fig(base_path+dir, 'mean_all_trained_test', fig)

def get_values_reg():
    with open('npy/incr/regular.npy', 'rb') as f:
        e = np.load(f)
        fh = np.load(f)
    return e[0], fh

def get_values_reg_mean():
    mfh = []
    for i in range(10):
        with open('npy/mean_reg/regular'+str(i)+'.npy', 'rb') as f:
            e = np.load(f)
            fh = np.load(f)
        mfh.append(fh)
    mfh = np.mean(np.array(mfh), axis=0)
    return e[0], mfh

def main():

    eps, fh_0, fh_1, fh_2, fh_3 = get_values()
    e, fh = get_values_reg()
    pre_path = 'any/'

    suptitle_0 = f"Loss and accuracy graph for model trained incrementally with Monte Carlo dropout uncertain images"
    path_0 = 'incr_uncrt_trained'
    plot_fig(fh_0, eps, suptitle_0, pre_path, path_0)

    suptitle_1 = f"Loss and accuracy graph for model trained incrementally with Monte Carlo dropout non-uncertain images"
    path_1 = 'incr_crt_trained'
    plot_fig(fh_1, eps, suptitle_1, pre_path, path_1)
    
    
    suptitle_2= f"Loss and accuracy graph for model trained incrementally with threshold interval uncertain images"
    path_2 = 'incr_uncrt_wiggle_trained'
    plot_fig(fh_2, eps, suptitle_2, pre_path, path_2)

    suptitle_3 = f"Loss and accuracy graph for model trained incrementally with threshold interval non-uncertain images"
    path_3 = 'incr_crt_wiggle_trained'
    plot_fig(fh_3, eps, suptitle_3, pre_path, path_3)

    # Plot for all
    hists = [fh_0, fh_1, fh_2, fh_3]
    suptitle="summary for training models with MC=Monte Carlo, TI=Threshold interval, Unc=Uncertain and Crt=Non-Uncertain images"
    plot_fig_all(hists, eps, r_hist=fh, r_eps=e, dir=pre_path, suptitle=suptitle)
    plot_fig_all_train_test(hists, eps, r_hist=fh, r_eps=e, suptitle=suptitle, dir=pre_path)



    eps, fh_0, fh_1, fh_2, fh_3 = get_values(budget='b')
    pre_path = 'budget/'
    
    suptitle_0 = f"Loss and accuracy graph for model trained incrementally with Monte Carlo dropout uncertain images, budget=35"
    path_0 = 'incr_uncrt_trained'
    plot_fig(fh_0, eps, suptitle_0, pre_path, path_0)

    suptitle_1 = f"Loss and accuracy graph for model trained incrementally with Monte Carlo dropout non-uncertain images, budget=35"
    path_1 = 'incr_crt_trained'
    plot_fig(fh_1, eps, suptitle_1, pre_path, path_1)
    
    
    suptitle_2= f"Loss and accuracy graph for model trained incrementally with threshold interval uncertain images, budget=35"
    path_2 = 'incr_uncrt_wiggle_trained'
    plot_fig(fh_2, eps, suptitle_2, pre_path, path_2)

    
    suptitle_3 = f"Loss and accuracy graph for model trained incrementally with threshold interval non-uncertain images, budget=35"
    path_3 = 'incr_crt_wiggle_trained'
    plot_fig(fh_3, eps, suptitle_3, pre_path, path_3)

    # Plot for all
    hists = [fh_0, fh_1, fh_2, fh_3]

    
    suptitle +=', budget=35'
    plot_fig_all(hists, eps, r_hist=fh, r_eps=e, dir=pre_path, suptitle=suptitle)
    plot_fig_all_train_test(hists, eps, r_hist=fh, r_eps=e, suptitle=suptitle, dir=pre_path)

    # Plot mean for both
        
    plot_fig_all_mean()
    
    

if __name__ == '__main__':
    main()

    
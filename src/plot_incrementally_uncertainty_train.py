import numpy as np
import matplotlib.pyplot as plt

# from data import get_cat_lab
import config as cnf
from prints import printc
from utils import plot_history, history_merge, axs_legend, axs_grid

def get_values():
    with open('npy/'+'iut.npy', 'rb') as f:
            eps = np.load(f)
            fh_0 = np.load(f)
            fh_1 = np.load(f)
            fh_2 = np.load(f)
            fh_3 = np.load(f)

    printc(f'eps: {eps.shape}')
    printc(f'fh_0: {fh_0.shape}')
    printc(f'fh_1: {fh_1.shape}')
    printc(f'fh_2: {fh_2.shape}')
    printc(f'fh_3: {fh_3.shape}')

    return eps, fh_0, fh_1, fh_2, fh_3

def plot_seperate_epochs(epochs, axs):
    cum_e = np.cumsum(epochs)

    for ce in cum_e: # plot lines to seperate epochs
        axs[0].axvline(ce, linestyle='dashed', linewidth=1,c='red')
        axs[1].axvline(ce, linestyle='dashed', linewidth=1,c='red')
    return cum_e[-1] # return total number of epochs

def plot_fig(full_hist, eps, suptitle, path):
    
    fig, axs = plt.subplots(1,2)
    
    total_epochs = plot_seperate_epochs(eps, axs)
    plot_history(axs[0], total_epochs, full_hist[:2], subfig_text='Loss function')
    plot_history(axs[1], total_epochs, full_hist[2:], subfig_text='Binary accuracy')
    axs[0].set_ylim(top=0.4)
    axs[1].set_ylim(bottom=0.81)
    axs[0].title.set_text('Loss function')
    axs[1].title.set_text('Binary Accuracy')

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig('../out_imgs/incrementally_uncertain/'+path)

def plot_fig_all(hists, eps):
    fig, axs = plt.subplots(1,2)
    
    
    # for hist in hists:
    plot_history(axs[0], np.sum(eps), hists[0][:2], label='MC Unc train', label_t='MC Unc test')
    plot_history(axs[1], np.sum(eps), hists[0][2:], label='MC Unc train', label_t='MC Unc test')

    plot_history(axs[0], np.sum(eps), hists[1][:2], label='MC Crt train', label_t='MC Crt test')
    plot_history(axs[1], np.sum(eps), hists[1][2:], label='MC Crt train', label_t='MC Crt test')
    
    plot_history(axs[0], np.sum(eps), hists[2][:2], label='Int Unc train', label_t='Int Unc test')
    plot_history(axs[1], np.sum(eps), hists[2][2:], label='Int Unc train', label_t='Int Unc test')

    plot_history(axs[0], np.sum(eps), hists[3][:2], label='Int Crt train', label_t='Int Crt test')
    plot_history(axs[1], np.sum(eps), hists[3][2:], label='Int Crt train', label_t='Int Crt test')

    axs[0].set_ylim(top=0.35)
    axs[1].set_ylim(bottom=0.86)
    fig.tight_layout()
    fig.savefig('../out_imgs/incrementally_uncertain/'+'all_trained.png')

def plot_fig_all_train_test(hists, eps):
    e = range(1, eps.sum()+1)

    fig, axs = plt.subplots(2)
    
    axs[0].plot(e, hists[0][0], label='MC Unc train')
    axs[0].plot(e, hists[1][0], label='MC Crt train')
    axs[0].plot(e, hists[2][0], label='Int Unc train')
    axs[0].plot(e, hists[3][0], label='Int Crt train')

    axs[1].plot(e, hists[0][2], label='MC Unc train')
    axs[1].plot(e, hists[1][2], label='MC Crt train')
    axs[1].plot(e, hists[2][2], label='Int Unc train')
    axs[1].plot(e, hists[3][2], label='Int Crt train')

    axs_legend(axs)
    axs_grid(axs)
    axs[0].set_ylim(top=0.28)
    axs[1].set_ylim(bottom=0.89)

    fig.set_figheight(10)
    fig.tight_layout()
    fig.savefig('../out_imgs/incrementally_uncertain/'+'all_trained_train.png')

    fig, axs = plt.subplots(2)
    
    axs[0].plot(e, hists[0][1], label='MC Unc test')
    axs[0].plot(e, hists[1][1], label='MC Crt test')
    axs[0].plot(e, hists[2][1], label='Int Unc test')
    axs[0].plot(e, hists[3][1], label='Int Crt test')

    axs[1].plot(e, hists[0][3], label='MC Unc test')
    axs[1].plot(e, hists[1][3], label='MC Crt test')
    axs[1].plot(e, hists[2][3], label='Int Unc test')
    axs[1].plot(e, hists[3][3], label='Int Crt test')

    axs_legend(axs)
    axs_grid(axs)
    axs[0].set_ylim(top=0.28)
    axs[1].set_ylim(bottom=0.87)

    fig.set_figheight(10)
    fig.tight_layout()
    fig.savefig('../out_imgs/incrementally_uncertain/'+'all_trained_test.png')
     

def main():
    eps, fh_0, fh_1, fh_2, fh_3 = get_values()
    # print(type(eps_0), type(fh_0), type(fh_1), type(fh_/2), type(fh_3))
    # Comment out after next run
    # eps = np.array(eps_0)
    # fh_0 = history_merge(fh_0, eps,'loss', 'binary_accuracy')
    # fh_1 = history_merge(fh_1, eps,'loss', 'binary_accuracy')
    # fh_2 = history_merge(fh_2, eps,'loss', 'binary_accuracy')
    # fh_3 = history_merge(fh_3, eps,'loss', 'binary_accuracy')
    # ------------------

    suptitle_0 = f"Loss and accuracy graph for model\ntrained incrementally and with uncertain images"
    path_0 = 'incr_uncrt_trained.png'
    plot_fig(fh_0, eps, suptitle_0, path_0)

    suptitle_1 = f"Loss and accuracy graph for model\ntrained incrementally and with Monte Carlo Dropout certain images"
    path_1 = 'incr_crt_trained.png'
    plot_fig(fh_1, eps, suptitle_1, path_1)
    
    
    suptitle_2= f"Loss and accuracy graph for model\ntrained incrementally and with certain images"
    path_2 = 'incr_uncrt_wiggle_trained.png'
    plot_fig(fh_2, eps, suptitle_2, path_2)

    suptitle_3 = f"Loss and accuracy graph for model\ntrained incrementally and with certain images"
    path_3 = 'incr_crt_wiggle_trained.png'
    plot_fig(fh_3, eps, suptitle_3, path_3)

    # Plot for all
    hists = [fh_0, fh_1, fh_2, fh_3]
    plot_fig_all(hists, eps)
    plot_fig_all_train_test(hists, eps)
    
    

        
    

if __name__ == '__main__':
    main()

    
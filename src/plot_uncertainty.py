import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from PIL import Image
from data import get_cat_lab
import utils.config as cnf
from prints import printc
from utils.plt import axs_legend, save_fig


labels = get_cat_lab()


def get_values():
    with open('npy/'+'u.npy', 'rb') as f:
        printc('Loaded:')
        values = np.load(f)
        printc(f'Values: {values.shape}')
        b_preds = np.load(f)
        printc(f'b_preds: {b_preds.shape}')
        preds = np.load(f)
        printc(f'preds: {preds.shape}')
        images = np.load(f)
        printc(f'images: {images.shape}')
    return values, b_preds, preds, images

# Plots image true label mus/vars/sigs
def plot_im_tl_mvs(sample_data, b_preds, axs):
    mus = np.empty_like(sample_data)
    vars = np.empty_like(sample_data)
    

    for i in range(sample_data.shape[0]):
        mus[i] = np.mean(sample_data[:i+1],axis=0)
        vars[i] = np.var(sample_data[:i+1],axis=0)
    sigs = np.sqrt(vars)
    
 
    x = np.arange(mus.shape[0])
    annotates_labels = np.flatnonzero(b_preds)

    for al in annotates_labels:
        axs[0].plot(x, vars[:,al], label=f'{labels[al]} $\sigma^2$')
        axs[1].plot(x, sigs[:,al], label=f'{labels[al]} $\sigma$')
        axs[2].plot(x, mus[:,al], label=f'{labels[al]} $\mu$')


def plot_im_tl_mvs_create(sample_data, b_preds, path_to_dir):
    fig, axs = plt.subplots(3)
    
    plot_im_tl_mvs(sample_data, b_preds, axs)
    
    axs_legend(axs)

    axs[0].set_ylabel('$\sigma^2$-value')
    axs[1].set_ylabel('$\sigma$-value')
    axs[2].set_ylabel('$\mu$-value')

    axs[0].set_xlabel('$N$ samples used to calculate $\sigma^2$')
    axs[1].set_xlabel('$N$ samples used to calculate $\sigma$')
    axs[2].set_xlabel('$N$ samples used to calculate $\mu$')

    fig.set_figheight(10)
    fig.suptitle('$\sigma^2$, $\sigma$, & $\mu$  progression throughout all samples')
    fig.tight_layout()
    save_fig(path='../out_imgs/uncertainty/'+path_to_dir, file_name='mu_var_change', fig=fig)
    

def plot_im_tl_pdf(sample_data, b_preds, preds, axs):
    mus = np.mean(sample_data, axis=0)
    vars = np.var(sample_data, axis=0)
    stds = np.sqrt(vars)
    # print(sample_data.shape$)
    annotated_labels = np.flatnonzero(b_preds)
    axs.axvline(cnf.threshold, linestyle='dashed', linewidth=1,c='red', label='Threshold')

    for al in np.flip(annotated_labels):
        mu = mus[al]
        var = vars[al]
        std = stds[al]
        label = labels[al]
        pred = preds[al]
        # if var.round(4) == 0:
        #     continue # var too small to plot

        x = np.linspace(mu-3*std, mu+3*std)
        axs.plot(x, stats.norm.pdf(x, mu, std), label=f'Class {label}, $\sigma^2={np.round(var, 4)}$')
        color = axs.lines[-1].get_color() # get color from last plot
        axs.scatter(mu-2*std, stats.norm.pdf(mu-2*std, mu,std), c=color, marker='X')
        axs.axvline(pred, c=color, linestyle='dashdot', label=f'Original class output value')
        # axs.scatter(sample_data[:,al], stats.norm.pdf(sample_data[:,al], mu, std))

def plot_im_tl_pdf_create(sample_data, b_preds, preds, image, path_to_dir):
    fig, axs = plt.subplots(1)
    plot_im_tl_pdf(sample_data, b_preds, preds, axs)
                
    axs.legend()

    axs.set_xlim([0,1])
    axs.set_ylim(bottom=0)
    axs.set_xlabel('Output neruon value after sigmoid activation')
    fig.suptitle('Probability density function for the annotated labels')
    fig.tight_layout()
    save_fig('../out_imgs/uncertainty/'+ path_to_dir, 'pdf_dist', fig)

    im = Image.fromarray(image)
    im.save('../out_imgs/uncertainty/'+ path_to_dir+ 'image.png')



def main():
    sample_datas, b_preds, preds, images = get_values() # sample_datas (n_im, n_samp, n_lab)
    for frame, (sd, bp, p, im) in enumerate(zip(sample_datas, b_preds, preds, images)):
        path_to_dir = 'im'+str(frame)+'/'
        plot_im_tl_mvs_create(sd, bp, path_to_dir)
        plot_im_tl_pdf_create(sd[:cnf.n_samples], bp, p, im, path_to_dir)
        # plot_im_tl_pdf_create(sample_datas[frame,:cnf.n_samples], b_preds[frame,:cnf.n_samples], preds[frame,:cnf.n_samples], images[frame,:cnf.n_samples])



    if not frame < images.shape[0]:
        print(f'frame needs to be within images, was {frame} and {images.shape[0]}')
        exit()
    # for 
    
    

    
if __name__ == "__main__":
    main()
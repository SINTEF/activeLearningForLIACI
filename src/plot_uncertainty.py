import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from PIL import Image
from data import get_cat_lab
import config as cnf
from prints import printc
from utils import axs_legend
# import itertool


labels = get_cat_lab()


def get_values():
    with open('npy/'+'u.npy', 'rb') as f:
        values = np.load(f)
        b_preds = np.load(f)
        preds = np.load(f)
        images = np.load(f)
    printc('Loaded:')
    printc(f'Values: {values.shape}')
    printc(f'b_preds: {b_preds.shape}')
    printc(f'preds: {preds.shape}')
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
    true_labels = np.flatnonzero(b_preds)

    for tl in true_labels:
        axs[0].plot(x, mus[:,tl], label=f'{labels[tl]} $\mu$')
        axs[1].plot(x, vars[:,tl], label=f'{labels[tl]} $\sigma^2$')
        axs[2].plot(x, sigs[:,tl], label=f'{labels[tl]} $\sigma$')


def plot_im_tl_mvs_create(sample_data, b_preds):
    fig, axs = plt.subplots(3)
    
    plot_im_tl_mvs(sample_data, b_preds, axs)
    
    axs_legend(axs)

    axs[0].set_xlabel('$\mu$ progression')
    axs[1].set_xlabel('$\sigma^2$ progression')
    axs[2].set_xlabel('$\sigma$ progression')

    fig.set_figheight(10)
    fig.suptitle('$\mu$, $\sigma^2$ and $\sigma$ progression throughout all samples')
    fig.tight_layout()
    fig.savefig('../out_imgs/uncertainty/'+ 'mu_var_change.png')

def plot_im_tl_pdf(sample_data, b_preds, preds, axs):
    mus = np.mean(sample_data, axis=0)
    vars = np.var(sample_data, axis=0)
    stds = np.sqrt(vars)
    # print(sample_data.shape$)
    annotated_labels = np.flatnonzero(b_preds)
    axs.axvline(cnf.threshold, linestyle='dashed', linewidth=1,c='red', label='Threshold')

    for al in annotated_labels:
        mu = mus[al]
        var = vars[al]
        std = stds[al]
        label = labels[al]
        pred = preds[al]

        x = np.linspace(mu-3*std, mu+3*std)
        axs.plot(x, stats.norm.pdf(x, mu, std), label=f'Class {label}, $\sigma^2={np.round(var, 4)}$')
        color = axs.lines[-1].get_color() # get color from last plot
        axs.scatter(mu-2*std, stats.norm.pdf(mu-2*std, mu,std), c=color, marker='X')
        axs.axvline(pred, c=color, linestyle='dashdot')
        # axs.scatter(sample_data[:,al], stats.norm.pdf(sample_data[:,al], mu, std))

def plot_im_tl_pdf_create(sample_data, b_preds, preds, image):
    fig, axs = plt.subplots(1)
    plot_im_tl_pdf(sample_data, b_preds, preds, axs)
                
    axs.legend()

    axs.set_xlim([0,1])
    
    fig.suptitle('PDF for the annotated labels')
    fig.tight_layout()
    fig.savefig('../out_imgs/uncertainty/'+ 'pdf_dist_1.png')
    im = Image.fromarray(image)
    im.save('../out_imgs/uncertainty/'+ 'image_1.png')



def main():
    sample_datas, b_preds, preds, images = get_values() # sample_datas (n_im, n_samp, n_lab)
    frame = 3
    if not frame < images.shape[0]:
        print(f'frame needs to be within images, was {frame} and {images.shape[0]}')
        exit()
    plot_im_tl_mvs_create(sample_datas[frame],b_preds[frame])

    plot_im_tl_pdf_create(sample_datas[frame,:cnf.n_samples], b_preds[frame,:cnf.n_samples], preds[frame,:cnf.n_samples], images[frame,:cnf.n_samples])
    

    
if __name__ == "__main__":
    main()
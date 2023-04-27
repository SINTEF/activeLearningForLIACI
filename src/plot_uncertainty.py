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

def plot_pdf(mu, std, pred, label,  ax):
    x = np.linspace(mu-3*std, mu+3*std)
    ax.plot(x, stats.norm.pdf(x, mu, std), label=f'Class {label}, $\sigma={np.round(std, 4)}$')
    color = ax.lines[-1].get_color() # get color from last plot
    ax.scatter(mu-2*std, stats.norm.pdf(mu-2*std, mu,std), c=color, marker='X')
    ax.axvline(pred, c=color, linestyle='dashdot', label=f'Original class output value')
    
    
# Plots image true label mus/vars/sigs
def plot_im_tl_ms_pdf(sample_data, b_preds, preds, axs):
    annotated_labels = np.flatnonzero(b_preds)
    annotated_labels = np.flip(annotated_labels)
    preds = preds[annotated_labels]

    # Calculate progression of mu, var, sig
    mu_prog = np.empty_like(sample_data)
    var_prog = np.empty_like(sample_data)
    

    for i in range(sample_data.shape[0]):
        mu_prog[i] = np.mean(sample_data[:i+1],axis=0)
        var_prog[i] = np.var(sample_data[:i+1],axis=0)
    sig_prog = np.sqrt(var_prog).T[annotated_labels]
    mu_prog = mu_prog.T[annotated_labels]
    
 
    
    pdf_mus = np.mean(sample_data, axis=0)[annotated_labels]
    pdf_vars = np.var(sample_data, axis=0)[annotated_labels]
    pdf_sigs = np.sqrt(pdf_vars)
    
    an_labels = np.array(labels)[annotated_labels]    

    x_prog = np.arange(mu_prog.shape[1])

    for label, mu, std, mp, sp, p in zip(an_labels, pdf_mus, pdf_sigs, mu_prog, sig_prog, preds):        
        if not std.round(3) == 0:
            plot_pdf(mu, std, p, label, axs[0])
        axs[1].plot(x_prog, sp, label=f'{label} $\sigma$')
        axs[2].plot(x_prog, mp, label=f'{label} $\mu$')


def plot_im_tl_ms_pdf_create(sample_data, b_preds, preds, path_to_dir):
    fig, axs = plt.subplots(3)
    
    plot_im_tl_ms_pdf(sample_data, b_preds, preds, axs)
    
    axs_legend(axs)

    # axs[0].set_ylabel('$\sigma^2$-value')
    axs[1].set_ylabel('$\sigma$-value')
    axs[2].set_ylabel('$\mu$-value')

    # axs[0].set_xlabel('$N$ samples used to calculate $\sigma^2$')
    axs[1].set_xlabel('$N$ samples used to calculate $\sigma$')
    axs[2].set_xlabel('$N$ samples used to calculate $\mu$')

    axs[0].set_xlim([0,1])
    axs[0].set_ylim(bottom=0)
    axs[0].set_xlabel('Output neruon value after sigmoid activation')
    
    fig.set_figheight(10)
    fig.suptitle('Probability density function, and $\sigma$ & $\mu$ progression throughout all samples, for the annotated labels', wrap=True)
    fig.tight_layout()
    save_fig(path='../out_imgs/uncertainty/'+path_to_dir, file_name='pdf_mu_std', fig=fig)


def main():
    sample_datas, b_preds, preds, images = get_values() # sample_datas (n_im, n_samp, n_lab)
    for frame, (sd, bp, p, im) in enumerate(zip(sample_datas, b_preds, preds, images)):
        path_to_dir = 'im'+str(frame)+'/'
        plot_im_tl_ms_pdf_create(sd, bp, p, path_to_dir)
        
        im = Image.fromarray(im)
        im.save('../out_imgs/uncertainty/'+ path_to_dir+'image.png')



    if not frame < images.shape[0]:
        print(f'frame needs to be within images, was {frame} and {images.shape[0]}')
        exit()
    # for 
    
    

    
if __name__ == "__main__":
    main()
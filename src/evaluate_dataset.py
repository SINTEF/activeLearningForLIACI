import matplotlib.pyplot as plt
import numpy as np

import utils.config as cnf
from utils.plt import show_bar_value
from utils.utils import recall_precision
from utils.model import hullifier_load
from data import split_data, load_from_coco, get_cat_lab, shuffle_data

def eval_dataset(predictions, truth, split, save_path, labels=None):
    if not labels:
        labels = get_cat_lab()

    x, y, xt, yt = split_data(predictions, truth, split)
    n_labels = truth.sum()

    img_per_cat = [ int(c.sum()) for c in truth.T ]
    img_per_train_cat = [ int(c.sum()) for c in y.T ]
    img_per_test_cat = [ int(c.sum()) for c in yt.T ]

    labels = [l.replace('_', '\n') for l in labels] # Format labels print pretty
    offset = 0.8 / 3
    # Whole dataset
    f, ax = plt.subplots()
    f.suptitle('Number of images for each class in the whole LIACi dataset')
    x_ax = np.arange(len(labels))    
    ax.grid(True)
    
    ax.bar(x_ax-offset, img_per_cat, width=offset, label='All images')
    ax.bar(x_ax, img_per_train_cat, width=offset, label='Train images')
    ax.bar(x_ax+offset, img_per_test_cat, width=offset, label='Test images')
    
    # Show bar value for every bar in the plot
    show_bar_value(ax)
        
    ax.set_xticks(x_ax, labels, rotation=-20)

    ax.set_ylabel('Number of images in each class')
    ax.set_xlabel('Labels')
    ax.legend()
    f.tight_layout()
    f.savefig(save_path+'dataset_stats.png')
    # print(save_path+'pdf/dataset_stats.pdf')
    f.savefig(save_path+'pdf/dataset_stats.pdf')

    # Test dataset
    f, ax = plt.subplots()
    f.suptitle('Number of images for each class in the\ntest part of the LIACi dataset')
    
    ax.grid(True)

    bar = ax.bar(x_ax, img_per_test_cat, label='Test images')
    ax.bar_label(bar)
    # Show bar value for every bar in the plot
    show_bar_value(ax)
        
    ax.set_xticks(x_ax, labels, rotation=-20)

    ax.set_ylabel('Number of images in each class')
    ax.set_xlabel('Labels')
    ax.legend()
    f.tight_layout()
    
    f.savefig(save_path+'dataset_test_stats.png')
    f.savefig(save_path+'pdf/dataset_test_stats.pdf')

    # Show ratio of images
    # Calculate ratio for all, train, test
    n_imgs = sum(img_per_cat)
    img_per_cat = [np.round(im / n_imgs, 2) for i, im in enumerate(img_per_cat)]
    
    n_imgs = sum(img_per_train_cat)
    img_per_train_cat = [np.round(im / n_imgs, 2) for i, im in enumerate(img_per_train_cat)]
    
    n_imgs = sum(img_per_test_cat)
    img_per_test_cat = [np.round(im / n_imgs, 2) for i, im in enumerate(img_per_test_cat)]
        
    f, ax = plt.subplots()
    f.suptitle('Partitioning of images for each class in the LIACi dataset')
    x_ax = np.arange(len(labels))    
    ax.grid(True)
    
    ax.bar(x_ax-offset, img_per_cat, width=offset, label='All images')
    ax.bar(x_ax, img_per_train_cat, width=offset, label='Train images')
    ax.bar(x_ax+offset, img_per_test_cat, width=offset, label='Test images')
    
    # Show bar value for every bar in the plot
    # show_bar_value(ax)
        
    ax.set_xticks(x_ax, labels, rotation=-20)

    ax.set_ylabel('Partitioning of images in each class')
    ax.set_xlabel('Labels')
    ax.legend()
    f.tight_layout()
    f.savefig(save_path+'dataset_ratio_stats.png')
    f.savefig(save_path+'pdf/dataset_ratio_stats.pdf')

    return

if __name__ == "__main__":
    pass
    X, Y = load_from_coco()
    X, Y = shuffle_data(X, Y)
    model = hullifier_load(cnf.model_path)
    predictions = model.predict(X)
    labels = get_cat_lab()
    path = '../out_imgs/ds/'
    eval_dataset(predictions, Y, 0.1, path, labels)
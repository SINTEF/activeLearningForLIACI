import numpy as np
import matplotlib.pyplot as plt
import base64
from itertools import product
from dict2obj import Dict2Obj

def show_bar_value(ax):
    for bars in  ax.containers:
        ax.bar_label(bars)
        
def get_axs_iter(axs):
    return list(product(np.arange(axs.shape[0]), np.arange(axs.shape[1])))

def recall_precision(predictions, tp_table, truth):
    n_predictions = predictions.sum()
    TP = tp_table.sum() # Find true positives / TP / hits
    P = int(truth.sum()) # n Positives

    FP = n_predictions - TP # Find False Positives / FP / misses
    
    recall = np.round(TP / P, 5) if P else 0
    precision = np.round(TP/n_predictions, 5) if n_predictions else 0 
    
    # printo(f'Psum: {n_predictions}, TP: {TP}, FP: {FP}, P: {P}, truth.shape: {truth.shape}, missing hits: {P-TP}, Precision: {precision}, Recall: {recall}')
    return recall, precision

def get_dict_from_file(f_path):
    values = {}
    with open(f_path, 'r') as f:
        for line in f:
            if not '=' in line:
                continue
            (k,v) = line.replace(' ', '').replace('\n','').split('=')
            values[k] = v
    return Dict2Obj(values)

def open_video_b64(path):
    with open(path, "rb") as videoFile:
            video = "data:video/mp4;base64," +  base64.b64encode(videoFile.read()).decode('ascii')
    return video


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
    ax.grid()
    ax.legend()
    
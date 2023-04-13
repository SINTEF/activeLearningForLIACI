import numpy as np
import matplotlib.pyplot as plt
import base64
from itertools import product
from dict2obj import Dict2Obj
from tensorflow import convert_to_tensor, expand_dims
import config as cnf

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
    ax.grid(visible=True)
    ax.legend()

def get_predictions_bool(model, X):
    preds = model.predict(X)
    predictions_bool = np.where(cnf.threshold <= preds, True, False)
    return predictions_bool, preds

# Evaluates single image
def find_uncertainty(image, b_preds, model):

    image = expand_dims(convert_to_tensor(image),0)

    values = np.empty(cnf.n_samples, cnf.n_labels)
    for k in range(cnf.n_samples): # collect samples from n_samples-subnetworks
        pred = model(image, training=True)[0]
        values[k] = pred

    mu = np.mu(values, axis=0)
    var = np.var(values, axis=0)
    sig = np.sqrt(var)
    
    uncertain = np.where(b_preds and cnf.threshold <= mu - (2 * sig), True, False)
    
    return uncertain

def history_merge(hists, eps, loss_measurement, measurement):

    full_hist = np.empty((4, np.sum(eps)))
    cum_e = np.cumsum(eps)

    prev = 0
    for h, e, ce in zip(hists, eps, cum_e):
        full_hist[0, prev:ce] = np.array(h.history[loss_measurement])
        full_hist[1, prev:ce] = np.array(h.history['val_' + loss_measurement])
        full_hist[2, prev:ce] = np.array(h.history[measurement])
        full_hist[3, prev:ce] = np.array(h.history['val_' + measurement])

        prev = ce
    return full_hist
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

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
    return values
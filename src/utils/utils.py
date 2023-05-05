import numpy as np
import matplotlib.pyplot as plt
import base64
import utils.config as cnf

line_styles = ['solid', 'dotted', 'dashed', 'dashdot']


def recall_precision(predictions, tp_table, truth):
    n_predictions = predictions.sum()
    TP = tp_table.sum() # Find true positives / TP / hits
    P = int(truth.sum()) # n Positives

    FP = n_predictions - TP # Find False Positives / FP / misses
    
    recall = np.round(TP / P, 5) if P else 0
    precision = np.round(TP/n_predictions, 5) if n_predictions else 0 
    
    # printo(f'Psum: {n_predictions}, TP: {TP}, FP: {FP}, P: {P}, truth.shape: {truth.shape}, missing hits: {P-TP}, Precision: {precision}, Recall: {recall}')
    return recall, precision



def open_video_b64(path):
    with open(path, "rb") as videoFile:
            video = "data:video/mp4;base64," +  base64.b64encode(videoFile.read()).decode('ascii')
    return video

def get_predictions_bool(model, X):
    preds = model.predict(X)
    predictions_bool = np.where(cnf.threshold <= preds, True, False)
    return predictions_bool, preds

def history_merge(hists, eps, loss_measurement='loss', measurement='binary_accuracy'):
    """ 
        This concatenates the different histories into one cohesive history

        hists: list of history's returned from different training
        eps: list of epoch's returned from different training

        return: np.array ndim:(4,total_training_epochs)
        The 4 dims are, train
  
    """

    full_hist = np.empty((4, np.sum(eps)))
    cum_e = np.cumsum(eps)

    prev = 0
    for h, e, ce in zip(hists, eps, cum_e):
        if type(h) is int: # If training was skipped fill with previous value
            full_hist[0, prev:ce].fill(full_hist[0, prev-1]) 
            full_hist[1, prev:ce].fill(full_hist[1, prev-1])
            full_hist[2, prev:ce].fill(full_hist[2, prev-1])
            full_hist[3, prev:ce].fill(full_hist[3, prev-1])
        else:
            full_hist[0, prev:ce] = np.array(h.history[loss_measurement])
            full_hist[1, prev:ce] = np.array(h.history['val_' + loss_measurement])
            full_hist[2, prev:ce] = np.array(h.history[measurement])
            full_hist[3, prev:ce] = np.array(h.history['val_' + measurement])

        prev = ce
    return full_hist
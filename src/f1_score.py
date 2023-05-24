import matplotlib.pyplot as plt
import numpy as np

import utils.config as cnf
from utils.plt import show_bar_value
from utils.utils import recall_precision
from utils.model import hullifier_load
from utils.plt import save_fig
from data import split_data, load_from_coco, get_cat_lab, shuffle_data


def f1_score(precision, recall):
    return (2 * ((precision * recall) / (precision + recall)))

def compute_best_f1(predictions, threshs, Y, save_path):
    predictions = [np.where(thresh <= predictions, 1, 0) for thresh in threshs]
    
    ps = []
    rs = []
    for pred in predictions:
        TP_table = np.logical_and(pred,Y)
        r,p = recall_precision(pred, TP_table, Y)
        ps.append(p)
        rs.append(r)
        
    ps = np.array(ps)
    rs = np.array(rs)

    f1_scores = f1_score(ps, rs)
    
    best = np.argmax(f1_scores)
    f1_scores = np.round(f1_scores, 4)
    ps = np.round(ps, 4)
    rs = np.round(rs, 4)
    threshs = np.round(threshs, 4)

    fig, ax = plt.subplots(1,2)    

    ax[0].plot(threshs, f1_scores, label='F1 scores')
    ax[0].axhline(f1_scores[best], label='F1 score maxima', c='r', linestyle='dashed')
    ax[0].axvline(threshs[best], c='r', linestyle='dashed')

    ax[0].set_ylabel('F1 score')
    ax[0].set_xlabel('Activation threshold')
    ax[0].set_ylim([0,1])
    ax[0].set_xlim([0,1])
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_title('F1 score graph')

    
    ax[1].set_title(f'Precision Recall Curve')
    # f.suptitle(f'Precision Recall Curve\nBest $f_1-score$ at $thres={threshs[best]}$')
    ax[1].plot(ps,rs, label='Precision Recall Curve')
    ax[1].scatter(ps[best],rs[best], label='Best $F1\ score$',c='r')
    ax[1].set_ylabel('Precision')
    ax[1].set_xlabel('Recall')
    ax[1].grid(True)
    ax[1].legend()
    
    t = threshs[best]
    fs = np.round(f1_scores[best],4)
    p = np.round(ps[best]*100,2)
    r = np.round(rs[best]*100,2)
    fig.suptitle(f'F1 score evaluation, Best F1 score at $threshold={t}$,\n$F1\ score={fs}$, $precision={p}$%, $recall={r}$%')
    save_fig(save_path, 'f1_ev', fig)
    
    return best

# Computes the F1 score
def show_acc(predictions, Y, labels, split, save_path='', **kwargs):
    x, y, xt, yt = split_data(predictions, Y, split)
    thresh_start = 0.01
    thresh_end = 1.0
    thresh_step = 0.01

    params = f'$thresh\ start={thresh_start}$, $thres\ end={thresh_end}$, $thresh\ step={thresh_step}$\n'
    for k,v in kwargs.items():
        params += f'${k}={v}$, '.replace('_', '\ ')
    params = params[:-2]
    
    

    threshs = np.arange(thresh_start, thresh_end, thresh_step) 
    
    best = compute_best_f1(predictions.copy(), threshs, Y, save_path)
    predictions = np.where(threshs[best] <= predictions, 1, 0)

    TP_table = np.logical_and(predictions,Y)
    # compute for overall recall/precision
    r, p = recall_precision(predictions, TP_table, Y)

    # Compute r/p for each category
    cat_r = []
    cat_p = []
    
    f_labs = [lab.replace('_', '\n') for lab in labels]
        
    for pred, cat_tp, t in zip(predictions.T, TP_table.T, Y.T):
        rec, prec = recall_precision(pred, cat_tp, t)
        cat_r.append(rec)
        cat_p.append(prec)
    width = 0.4

    fig, axs = plt.subplots(2,2)
    x_ax = np.arange(len(f_labs))
    
    axs[0][0].bar(x_ax-width/2,  cat_p, width=width, label='Precision')
    axs[0][0].bar(x_ax+width/2, cat_r, width=width, label='Recall')
    axs[0][0].set_xticks(x_ax, f_labs, rotation=-20)
    
    axs[0][0].legend()
    axs[0][0].set_title(f'Precision & Recall\nfor each category of labels in the LIACi dataset')
    axs[0][0].set_ylim([0,1])
    
    fn = 'PR_labels'
    
    # save_fig(save_path, fn, fig)
    # return
    # exit()
    
    # Compute average r/p for each image
    img_r = []
    img_p = []
    for pred, img_tp, t in zip(predictions, TP_table, Y):
        rec, prec = recall_precision(pred, img_tp, t)
        img_r.append(rec)
        img_p.append(prec)
    avg_img_r = round(sum(img_r)/len(img_r),4)
    avg_img_p = round(sum(img_p)/len(img_p),4)
    

    # fig, ax = plt.subplots()
    x_ax = np.arange(predictions.shape[0])
    
    axs[0][1].bar(0, avg_img_p, label='Precision')
    axs[0][1].bar(1, avg_img_r, label='Recall')
    # axs[0][1].bar(x_ax-width/2, avg_img_p, width=width, label='Precision')
    # axs[0][1].bar(x_ax+width/2, avg_img_r, width=width, label='Recall')
    # axs[0][1].set_xticks(x_ax, f_labs, rotation=-20)
    
    show_bar_value(axs[0][1])
    axs[0][1].set_ylim([0,1])
    axs[0][1].set_xticks([0,1], ['Precision', 'Recall'])
    axs[0][0].set_ylim([0,1])
    
    axs[0][1].legend()
    axs[0][1].set_title(f'Average Precision & Recall\n for every image in the LIACi dataset')
    
    fn = 'PR_frames_avg'
    # save_fig(save_path, fn, fig)

    # Compute r/p for category among the test images 
    cat_r = []
    cat_p = []
            
    # print(TP_table[yt.shape[0]:].shape)
    # print(TP_table[-yt.shape[0]:].shape)
    for pred, cat_tp, t in zip(xt.T, TP_table[-yt.shape[0]:].T, yt.T):
        rec, prec = recall_precision(pred, cat_tp, t)
        cat_r.append(rec)
        cat_p.append(prec)
    width = 0.4

    # fig, ax = plt.subplots()
    x_ax = np.arange(len(f_labs))
    
    axs[1][0].bar(x_ax-width/2,  cat_p, width=width, label='Precision')
    axs[1][0].bar(x_ax+width/2, cat_r, width=width, label='Recall')
    axs[1][0].set_xticks(x_ax, f_labs, rotation=-20)
    axs[1][0].set_ylim([0,1])
    axs[1][0].legend()
    axs[1][0].set_title(f'Precision & Recall\nfor each category of the test images in the LIACi dataset')
    
    # fn = 'PR_labels_test'
    # save_fig(save_path, fn, fig)

    # Compute r/p for category among the train images 
    cat_r = []
    cat_p = []
            
    
    for pred, cat_tp, t in zip(x.T, TP_table[:y.shape[0]].T, y.T):
        rec, prec = recall_precision(pred, cat_tp, t)
        cat_r.append(rec)
        cat_p.append(prec)
    width = 0.4

    # fig, ax = plt.subplots()
    x_ax = np.arange(len(f_labs))
    
    axs[1][1].bar(x_ax-width/2,  cat_p, width=width, label='Precision')
    axs[1][1].bar(x_ax+width/2, cat_r, width=width, label='Recall')
    axs[1][1].set_xticks(x_ax, f_labs, rotation=-20)
    
    axs[1][1].set_ylim([0,1])
    axs[1][1].legend()
    axs[1][1].set_title(f'Precision & Recall\nfor each category of the train images in the LIACi dataset')
    
    # fn = 'PR_labels_train'
    fn = 'PR_ev'
    fig.set_figheight(10)
    fig.set_figwidth(10)
    fig.suptitle(f"Precision and recall on the LIACI data set using $threshold={threshs[best]}$")
    save_fig(save_path, fn, fig)




if __name__ == "__main__":
    pass
    X, Y = load_from_coco()
    X, Y = shuffle_data(X, Y)
    model = hullifier_load(cnf.model_path)
    predictions = model.predict(X)

    labels = get_cat_lab()

    path = '../out_imgs/f1/base/'
    print('Computing acc')
    show_acc(predictions, Y, labels, 0.1, path)
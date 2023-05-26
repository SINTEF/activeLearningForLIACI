import numpy as np
import matplotlib.pyplot as plt


from data import load_from_coco, shuffle_data, split_data
from f1_score import f1_scores
from utils.model import hullifier_load
from utils.plt import save_fig

base_dir = '../out_imgs/f1/'

def plot_f1_scores(path, model_names, m_preds, xt, yt):
    threshs = np.arange(0.01, 1.0, 0.01) # Start, stop, step

    m_f1_scrs = []
    m_rs = []
    m_ps = []
    
    for preds in m_preds:
        f1_scrs, ps, rs = f1_scores(preds, threshs, yt)

        f1_scrs = np.round(f1_scrs, 4)
        ps = np.round(ps, 4)
        rs = np.round(rs, 4)
        m_f1_scrs.append(f1_scrs)
        m_rs.append(rs)
        m_ps.append(ps)

    threshs = np.round(threshs, 4)
    

    
    fig, axs = plt.subplots(1, 2)
    # plt.rcParams.update({'font.size': 5})
    plt.rc('legend', fontsize=8) 
    for f1_scrs, ps, rs, mn in zip(m_f1_scrs, m_ps, m_rs, model_names): # Iterate through models

        best = np.argmax(f1_scrs)

        axs[0].plot(threshs, f1_scrs, label=mn+' F1 score graph')
        color = axs[0].lines[-1].get_color() # get color from last plot
        axs[0].axhline(f1_scrs[best], label=mn+f' F1 score {f1_scrs[best]}', linestyle='dashed', c=color)
        axs[0].axvline(threshs[best], linestyle='dashed', c=color)

        axs[1].plot(ps, rs, label='Precision Recall Curve')
        axs[1].scatter(ps[best],rs[best], label='Best $F1\ score$')

    # plt.rcParams.update({'font.size': 20})
    axs[0].set_ylabel('F1 score')
    axs[0].set_xlabel('Classification threshold')
    axs[0].set_ylim([0,1])
    axs[0].set_xlim([0,1])
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_title('F1 score graphs')

    axs[1].set_title(f'Precision Recall Curve')
    axs[1].set_ylabel('Precision')
    axs[1].set_xlabel('Recall')
    axs[1].grid(True)
    axs[1].legend()
    
    # t = threshs[best]
    # fs = np.round(f1_scrs[best],4)
    # p = np.round(ps[best]*100,2)
    # r = np.round(rs[best]*100,2)
    # fig.set_figheight(3)
    fig.suptitle(f'F1 score evaluation'+ (' for models with a budget' if path=='budget/' else '')+',\nmcu=Monte Carlo uncertain, mcc=Monte Carlo non-uncertain, tiu=Threshold interval uncertain, tic=Threshold interval non-uncertain', wrap=True)
    save_fig(base_dir+path,'f1_ev', fig)

    return fig, axs

def plot_version(version, X, Y):
    x,y,xt,yt = split_data(X,Y, 0.1)
    mcu = 'mcu'
    mcc = 'mcc'
    tic = 'tic'
    tiu = 'tiu'
    

    model_names = [
        mcu,
        mcc,
        tiu,
        tic
    ]

    b_models = [hullifier_load('models/iuc_models/'+version+mn) for mn in model_names] # load all models
    
    m_b_preds = [ model.predict(xt) for model in b_models ] # all models predict on test data
    
    
    return plot_f1_scores(version, model_names, m_b_preds, xt, yt)

def main():
    X, Y = load_from_coco()
    X, Y = shuffle_data(X,Y)

    version='budget/'
    fig, axs = plot_version(version, X, Y)
    axs[0].set_xlim([0.2,0.63])
    axs[0].set_ylim([0.78,0.85])

    axs[1].set_xlim([0.77,0.9])
    axs[1].set_ylim([0.65,0.84])
    save_fig(base_dir+version,'zoom_f1_ev', fig)

    version='any/'
    fig, axs = plot_version(version, X, Y)
    axs[0].set_xlim([0.2,0.63])
    axs[0].set_ylim([0.78,0.85])

    axs[1].set_xlim([0.77,0.9])
    axs[1].set_ylim([0.65,0.84])
    save_fig(base_dir+version,'zoom_f1_ev', fig)
    
    


if __name__ == '__main__':
    main()
import numpy as np
from data import get_cat_lab, load_from_coco, get_cats, shuffle_data, split_data
from model import summarize_diagnostics, train, hullifier_create, hullifier_load, hullifier_save
import matplotlib.pyplot as plt
from argparse import ArgumentParser, BooleanOptionalAction
from prints import printe, printw, printo, printc
from utils import show_bar_value, recall_precision

def choose_model(X, n_cats, lr, v2, model_path):
    
    if model_path:
        try:
            print('path: ', model_path)
            model = hullifier_load(path=model_path)
            printo(f"successfully loaded model from {model_path}")
            return model
        except:
            printe("Couldn't load model.")
            inp = input('Do you want to exit?:[y] ').lower()
            if inp == 'y':
                exit()
            
    print("Creating new model...")
    model = hullifier_create(X, n_cats, lr, v2)

    return model

def train_model(model, res, Y, epochs, batch_size, validation_data):
    print(validation_data[0].shape,validation_data[1].shape)
    h, e = train(
        model,
        res,
        Y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
    )
    return h, e



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

    f, ax = plt.subplots()    
    f.suptitle(f'Precision Recall Curve, Best $f_1-score$ at \n$threshold={threshs[best]}$, $f_1-score={np.round(f1_scores[best],4)}$, $precision={np.round(ps[best]*100,2)}$%, $recall={np.round(rs[best]*100,2)}$%')
    # f.suptitle(f'Precision Recall Curve\nBest $f_1-score$ at $thres={threshs[best]}$')
    ax.plot(ps,rs, label='Precision Recall Curve')
    ax.scatter(ps[best],rs[best], label='Best $f_1-score$',c='r')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    
    ax.grid(True)
    ax.legend()
    f.tight_layout()
    f.savefig(save_path+'prec_reca_curve.png')
    f.savefig(save_path+'pdfs/prec_reca_curve.pdf')
    return best

def eval_dataset(predictions, truth, split, save_path, labels):
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
    # print(save_path+'pdfs/dataset_stats.pdf')
    f.savefig(save_path+'pdfs/dataset_stats.pdf')

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
    f.savefig(save_path+'pdfs/dataset_test_stats.pdf')

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
    f.savefig(save_path+'pdfs/dataset_ratio_stats.pdf')

    return


    
def show_acc(predictions, Y, labels, split, save_path='', **kwargs):
    x, y, xt, yt = split_data(predictions, Y, split)
    thresh_start = 0.05
    thresh_end = 0.96
    thresh_step = 0.05

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

    f, ax = plt.subplots()
    x_ax = np.arange(len(f_labs))
    
    ax.bar(x_ax-width/2,  cat_p, width=width, label='Precision')
    ax.bar(x_ax+width/2, cat_r, width=width, label='Recall')
    ax.set_xticks(x_ax, f_labs, rotation=-20)
    
    ax.legend()
    f.suptitle(f'Precision & Recall\nfor each category of labels in the LIACi dataset')
    f.tight_layout()
    
    f.savefig(save_path + 'PR_labels.png')
    f.savefig(save_path + 'pdfs/PR_labels.pdf')
    plt.close()
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
    

    f, ax = plt.subplots()
    x_ax = np.arange(predictions.shape[0])
    
    ax.bar(0, avg_img_p, label='Precision')
    ax.bar(1, avg_img_r, label='Recall')
    # ax.bar(x_ax-width/2, avg_img_p, width=width, label='Precision')
    # ax.bar(x_ax+width/2, avg_img_r, width=width, label='Recall')
    # ax.set_xticks(x_ax, f_labs, rotation=-20)
    
    show_bar_value(ax)
    ax.set_ylim([0,1])
    ax.set_xticks([0,1], ['Precision', 'Recall'])
    
    ax.legend()
    f.suptitle(f'Average Precision & Recall\n for every image in the LIACi dataset')
    f.tight_layout()
    
    f.savefig(save_path + 'PR_frames_avg.png')
    f.savefig(save_path + 'pdfs/PR_frames_avg.pdf')

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

    f, ax = plt.subplots()
    x_ax = np.arange(len(f_labs))
    
    ax.bar(x_ax-width/2,  cat_p, width=width, label='Precision')
    ax.bar(x_ax+width/2, cat_r, width=width, label='Recall')
    ax.set_xticks(x_ax, f_labs, rotation=-20)
    
    ax.legend()
    f.suptitle(f'Precision & Recall\nfor each category of the test images in the LIACi dataset')
    f.tight_layout()
    
    f.savefig(save_path + 'PR_labels_test.png')
    f.savefig(save_path + 'pdfs/PR_labels_test.pdf')

    # Compute r/p for category among the train images 
    cat_r = []
    cat_p = []
            
    print(y.shape)
    print(TP_table[:y.shape[0]].shape)
    for pred, cat_tp, t in zip(x.T, TP_table[:y.shape[0]].T, y.T):
        rec, prec = recall_precision(pred, cat_tp, t)
        cat_r.append(rec)
        cat_p.append(prec)
    width = 0.4

    f, ax = plt.subplots()
    x_ax = np.arange(len(f_labs))
    
    ax.bar(x_ax-width/2,  cat_p, width=width, label='Precision')
    ax.bar(x_ax+width/2, cat_r, width=width, label='Recall')
    ax.set_xticks(x_ax, f_labs, rotation=-20)
    
    ax.legend()
    f.suptitle(f'Precision & Recall\nfor each category of the train images in the LIACi dataset')
    f.tight_layout()
    
    f.savefig(save_path + 'PR_labels_train.png')
    f.savefig(save_path + 'pdfs/PR_labels_train.pdf')

def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('-e','--epochs', required=False, default=25, type=int)
    parser.add_argument('-bs','--batch_size', required=False, default=50, type=int)
    parser.add_argument('-lr','--lr', required=False, default=2e-4, type=float)
    parser.add_argument('-vs','--v_split', required=False, default=0.1, type=float)
    parser.add_argument('-s','--seed', required=False, default=0, type=int)

    parser.add_argument('-p', '--path', help='path to output files', default='', required=False)
    parser.add_argument('-so', '--save_option', default='', required=False)

    parser.add_argument('-mp', '--model_path', help='To run with existing model provide model_path ', default='', required=False)
    parser.add_argument('-n','--n_imgs', required=False, default=0, type=int)
    parser.add_argument('-nc','--n_cats', required=False, default=len(get_cats())-1, type=int)
    parser.add_argument('-v2','--version_2', required=False, default=False, type=bool)
    parser.add_argument('-ts','--target_size', required=False, default=(224,224))
    parser.add_argument('-tl','--transfer_learning', required=False, default=True)
    parser.add_argument('-tl','--evaluate', required=False, default=True)
    args = parser.parse_args()

    return args


def main():
    args = parseArgs()
    printo(str(args)[10:][:-1])
    pipeline_start(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, v_split=args.v_split, seed=args.seed, n_imgs=args.n_imgs, n_cats=args.n_cats, version_2=args.version_2, target_size=args.target_size, transfer_learning=args.transfer_learning)

def pipeline_start(n_imgs=0, n_cats=9, model_path='', transfer_learning=True, version_2=False, epochs=1, batch_size=50, v_split=0.1, seed=0, lr=2e-4, path='../benchmark', save_option='', X=None, Y=None, evaluate=True, **kwargs):
    target_size = kwargs['target_size'] if 'target_size' in kwargs else (224, 224) 

    if not isinstance(X,np.ndarray) or not isinstance(Y, np.ndarray):
        X, Y = load_from_coco(n_imgs=n_imgs, target_size=target_size)
    X, Y = shuffle_data(X,Y,seed=seed)
    t_data = X[0]
    t_data = t_data.reshape(1,t_data.shape[0], t_data.shape[1], t_data.shape[2])
    model = choose_model(X, n_cats, lr, v2=version_2, model_path=model_path)
    
    x, y, xt, yt = split_data(X,Y, v_split)

    if transfer_learning:
        h, e = train_model(model, x, y, epochs, batch_size, (xt, yt))
        summarize_diagnostics(h, e, path,  lr=lr, v_split=v_split, version_2=version_2)

    if evaluate:
        labs = get_cat_lab()
        predictions = model.predict(X)
        eval_dataset(predictions, Y, v_split, path, labs)
        show_acc(predictions, Y, save_path=path, lr=lr, split=v_split, version_2=version_2, labels=labs)
        
    if (save_option.lower() == 'y' or save_option == '') and (save_option.lower() == 'y' or input("Save model?:[y] ").lower() == 'y'):
        hullifier_save(model, path + 'model/', lr=lr, epochs=epochs, v_split=v_split, v2=version_2, seed=seed)
        printo(f'model saved to {path}')


if __name__ == "__main__":
    main()
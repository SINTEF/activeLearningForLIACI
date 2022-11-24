import numpy as np
from data import get_cat_lab, load_from_coco, get_cats, shuffle_data
from model import summarize_diagnostics, train, hullifier_create, hullifier_load, hullifier_save
import matplotlib.pyplot as plt
from argparse import ArgumentParser, BooleanOptionalAction
from prints import printe, printw, printo, printc

def choose_model(X, n_cats, lr, v2, use_old):
    
    if use_old:
        try:
            postfix = '_v2' if v2 else '_v1'
            model = hullifier_load(version=postfix)
            printo("successfully loaded")
            return model
        except:
            printe("Couldn't load model.")
            inp = input('Do you want to exit?:[y] ').lower()
            if inp == 'y':
                exit()
            
    print("Creating new model...")
    model = hullifier_create(X, n_cats, lr, v2)

    return model

def train_model(model, res, Y, epochs, batch_size, validation_split):
    h, e = train(
        model,
        res,
        Y,
        batch_size=batch_size,
        epochs=epochs,
        v_split=validation_split
    )
    return h, e


def show_acc(model, X, Y, thresh=0.5):
    P = model.predict(X)
    P = np.where(thresh <= P, 1, 0)

    hits = np.logical_and(P,Y).sum()
    
    n_err = int((Y).sum() - hits) + ((P).sum() - hits) 
    n_guesses = np.logical_or(P,Y).sum()

    f, ax = plt.subplots()
    f.suptitle("Accuracy scatter plot for whole data set")
    print(f"Accuracy: {n_guesses-n_err}/{n_guesses}={np.round(((n_guesses-n_err)/n_guesses)*100,2)}% (based on amount of total guesses)")# I think this is correct

    d = model.evaluate(X,Y)
    print(d)

    X = [np.where(x)[0] for x in P] # get indecies of guesses for plotting
    Y = [np.where(y)[0] for y in Y] # get indecies of guesses for plotting
        
    labs = get_cat_lab()

    X = X[:100]
    Y = Y[:100]

    for i, x in enumerate(X):
        ax.scatter([i]*len(x), x, color='r', marker='o')
    for i, y in enumerate(Y):
        ax.scatter([i]*len(y), y, color='y', marker='x')
    
    l = np.arange(len(labs))
    ax.set_yticks(l, labels=labs)
    
    f.savefig('../out_imgs/scatter_acc.pdf')

def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('-e','--epochs', required=False, default=25, type=int)
    parser.add_argument('-bs','--batch_size', required=False, default=50, type=int)
    parser.add_argument('-lr','--lr', required=False, default=2e-4, type=float)
    parser.add_argument('-vs','--v_split', required=False, default=0.1, type=float)
    parser.add_argument('-s','--seed', required=False, default=0, type=int)

    parser.add_argument('-p', '--path', help='path to output files', default='', required=False)
    parser.add_argument('-so', '--save_option', default='', required=False)

    parser.add_argument('-o', '--old', help='To run with old model', default=False, required=False, action=BooleanOptionalAction)
    parser.add_argument('-n','--n_imgs', required=False, default=0, type=int)
    parser.add_argument('-nc','--n_cats', required=False, default=len(get_cats())-1, type=int)
    parser.add_argument('-v2','--version_2', required=False, default=False, type=bool)
    parser.add_argument('-ts','--target_size', required=False, default=(224,224))
    parser.add_argument('-tl','--transfer_learning', required=False, default=True, action=BooleanOptionalAction)
    args = parser.parse_args()

    return args


def main():
    args = parseArgs()
    # printo(str(args)[10:][:-1])
    pipeline_start(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, v_split=args.v_split, seed=args.seed, n_imgs=args.n_imgs, n_cats=args.n_cats, version_2=args.version_2, target_size=args.target_size, transfer_learning=args.transfer_learning)

def pipeline_start(n_imgs=0, n_cats=9, old=False, transfer_learning=True, version_2=False, epochs=1, batch_size=50, v_split=0.1, seed=0, lr=2e-4, path='../benchmark', save_option='', **kwargs):
    # exit()
    target_size = (224, 224)    
    if not version_2:
        postfix = '_v2' if version_2 else '_v1'
    


    X, Y = load_from_coco(n_imgs=n_imgs, target_size=target_size)
    X, Y = shuffle_data(X,Y,seed=seed)
    t_data = X[0]
    t_data = t_data.reshape(1,t_data.shape[0], t_data.shape[1], t_data.shape[2])
    model = choose_model(X, n_cats, lr, v2=version_2, use_old=old)
    



    if transfer_learning:
        h, e = train_model(model, X, Y, epochs, batch_size, v_split)
        summarize_diagnostics(h, e, path,  lr=lr, v_split=v_split, version=postfix)

        
    # loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    
    if save_option == 'y' or input("Save model?:[y] ").lower() == 'y':
        hullifier_save(model, path + 'model/', lr=lr, epochs=epochs, v_split=v_split, v2=version_2)
        printo(f'model saved to {path}')


if __name__ == "__main__":
    main()
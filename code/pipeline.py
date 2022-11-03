import numpy as np
from data import get_cat_lab, load_from_coco, get_cats, shuffle_data
from model import hullifier_save, summarize_diagnostics, train, hullifier_create
import matplotlib.pyplot as plt
from argparse import ArgumentParser, BooleanOptionalAction
from prints import printe, printw, printo, printc

def choose_model(X, n_cats, lr, v2, use_old):
    
    if use_old:
        try:
            postfix = '_v2' if v2 else '_v1'
            model = model_load(version=postfix)
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


def show_acc(model, X, Y):
    f, ax = plt.subplots()
    f.suptitle("Accuracy scatter plot for whole data set")
    P = model.predict(X)
    P = np.where(0.5 <= P, 1, 0)
    hits = np.logical_and(P,Y).sum()
    
    n_err = int((Y).sum() - hits) + ((P).sum() - hits) 
    n_guesses = np.logical_or(P,Y).sum()
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
    parser.add_argument('-o', '--old', help='To run with old model', default=False, required=False, action=BooleanOptionalAction)
    parser.add_argument('-n','--n_imgs', required=False, default=0)
    parser.add_argument('-e','--epochs', required=False, default=20)
    parser.add_argument('-bs','--batch_size', required=False, default=50)
    parser.add_argument('-nc','--n_cats', required=False, default=len(get_cats())-1)
    parser.add_argument('-lr','--lr', required=False, default=2e-4)
    parser.add_argument('-vs','--v_split', required=False, default=0.1)
    parser.add_argument('-v2','--v2', required=False, default=True)
    parser.add_argument('-ts','--target_size', required=False, default=(224,224))
    parser.add_argument('-tf','--transfer_learning', default=True, required=False, action=BooleanOptionalAction)
    parser.add_argument('-ft','--fine_tuning', default=False, required=False, action=BooleanOptionalAction)
    args = parser.parse_args()

    return args

def main():
    args = parseArgs()
    printo(args)
    target_size = args.target_size
    if not args.v2:
        target_size = (224,224)
    postfix = '_v2' if args.v2 else '_v1'
    


    X, Y = load_from_coco(n_imgs=args.n_imgs, target_size=target_size)
    X, Y = shuffle_data(X,Y)

    
    model = choose_model(X, args.n_cats, args.lr, v2=args.v2, use_old=args.old)

    if args.transfer_learning:
        h, e = train_model(model, X, Y, args.epochs, args.batch_size, args.v_split)
        summarize_diagnostics(h, e, version=postfix)
    if args.fine_tuning:
        for l in model.layers:
            l.trainable = True
        h, e = train_model(model, X, Y, args.epochs, args.batch_size, args.v_split)
        summarize_diagnostics(h, e, version=postfix)
    # loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    
    if input("Save model?:[y] ").lower() == 'y':
        hullifier_save(model, postfix, lr=args.lr, epochs=args.epochs, v_split=args.v_split)
        printo('model saved')


if __name__ == "__main__":
    main()
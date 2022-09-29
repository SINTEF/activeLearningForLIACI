from termios import N_MOUSE
import numpy as np
from data import get_cat_lab, load_from_coco, get_cats
from model import mobilenet_create, model_create, model_load, model_save, summarize_diagnostics, train
import matplotlib.pyplot as plt
from prints import printe, printw, printo, printc

DEBUG = True

def choose_model(res, n_cats, lr):
    inp = input("Load saved model?:[Y] ").lower()
    if inp == 'y':
        try:
            model = model_load()
            printo("successfully loaded")
        except:
            printe("Couldn't load model.")
            inp = input('Do you want to exit?:[Y] ').lower()
            if inp == 'y':
                exit()
            
    if inp != 'y':
        print("Creating new model...")
        model = model_create(res.shape, n_cats=n_cats, lr=lr)

    return model

def train_model(model, res, Y, epochs, validation_split=0.2):
    # print(Y.shape)
    inp = input("Do you want to train the model?:[y] ").lower()
    if inp == 'y' or not inp:
        h, e = train(
            model,
            res,
            Y,
            epochs=epochs,
            v_split=validation_split
        )
        return h, e
        
def predict_model(model, X):
    result = model.predict(X)
    return result 


def show_acc(model, X, Y):
    f, ax = plt.subplots()
    f.suptitle("Accuracy scatter plot for whole data set")
    P = predict_model(model, X)
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
    
def main():

    n_imgs = 0
    epochs = 35
    n_cats = len(get_cats())-1
    lr = 2e-4
    v_split = 0.2

    X, Y = load_from_coco(n_imgs=n_imgs)

    mobilenet = mobilenet_create()
    
    X = mobilenet.predict(X) # Extract Features from mobilenet
    

    model = choose_model(X, n_cats, lr)
    h, e = train_model(model, X, Y, epochs, v_split)
    show_acc(model, X, Y)
    summarize_diagnostics(h, e)
    
    # loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    
    if False and input("Save model?: ").lower() == 'y':
        model_save(model)
        printo('model saved')

    # mobilenet.summary()
    # model.summary()
    # rm = model.predict(res)
    # print(rm.shape)


if __name__ == "__main__":
    main()
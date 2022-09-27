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

def train_model(model, res, Y, epochs):
    # print(Y.shape)
    inp = input("Do you want to train the model?:[y] ").lower()
    if inp == 'y' or not inp:
        h, e = train(
            model,
            res,
            Y,
            epochs=epochs
        )
        return h, e
        
def predict_model(model, X):
    result = model.predict(X)
    return result 


def show_acc(model, X, Y):
    f, ax = plt.subplots()
    
    res = predict_model(model, X)
    res = np.where(0.5 <= res, 1, 0)

    X = [np.where(x)[0] for x in res] # get indecies of guesses
    Y = [np.where(y)[0] for y in Y] # get indecies of guesses
    labs = get_cat_lab()

    for i, (x, y) in enumerate(zip(X, Y)):
        ax.scatter([i]*len(x), x, color='r', marker='o')
        ax.scatter([i]*len(y), y, color='y', marker='x')
    
    l = np.arange(len(labs))

    
def main():

    n_imgs = 0
    epochs = 100
    n_cats = len(get_cats())-1
    lr = 2e-4

    X, Y = load_from_coco(n_imgs=n_imgs)
    mobilenet = mobilenet_create()
    res = mobilenet.predict(X) # Extract Features from mobilenet

    model = choose_model(res, n_cats, lr)
    h, e = train_model(model, res, Y, epochs)
    show_acc(model, res, Y)
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
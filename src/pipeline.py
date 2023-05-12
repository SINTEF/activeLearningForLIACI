import numpy as np
from data import get_cat_lab, load_from_coco, get_cats, shuffle_data, split_data
from model import summarize_diagnostics, train, hullifier_create, hullifier_load, hullifier_save
import matplotlib.pyplot as plt
from argparse import ArgumentParser, BooleanOptionalAction
from prints import printe, printw, printo, printc

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
    parser.add_argument('-ev','--evaluate', required=False, default=True)
    args = parser.parse_args()

    return args


def main():
    args = parseArgs()
    printo(str(args)[10:][:-1]) # Prints args without "Namespace()"
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
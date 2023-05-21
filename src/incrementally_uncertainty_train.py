import numpy as np
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from model import hullifier_clone
from prints import printe, printo, printw, printc
from data import load_from_coco, shuffle_data, split_data
from model import hullifier_create

from utils.model import hullifier_compile, classifier_create, train, preproc_model_create, get_lr, hullifier_save
from utils.utils import history_merge, get_predictions_bool
from utils.file import get_dict_from_file
from utils.uncertainty import find_uncertainty, get_feature_extractor
import utils.config as cnf


measure_time = True
if measure_time:
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    printc(f'Started at: {current_time}')
    start = time.time()
def t():
    tm = int(time.time()-start)
    return f'{tm//60}m {tm%60}s'



# [x] Create NEW hullifier
# [x] Load old data
# [x] split data into original split
# [x] Split train data into smaller splits
# [x] Train on "inital" data
# [] 
# []
# []
# []
# [] compare final result to original result
# [] Seperate data with uncertain images and non uncertain images
# [] Train on only uncertain images and train on only certain images
# []
# [] Also do this for the other "uncertainty" measurement method
# []
# []
# []
# []
# [] Compare results
# []
def save_values(path, eps, full_hist_0, full_hist_1, full_hist_2, full_hist_3):
    print('Saving values to:', path)
    with open(path, 'wb') as f:
        np.save(f, eps)
        np.save(f, full_hist_0)
        np.save(f, full_hist_1)
        np.save(f, full_hist_2)
        np.save(f, full_hist_3)
    
def get_classifier(model):
    lr = get_lr(model)

    # Create classifier
    classifier = classifier_create(dr_rate=0.1)
    hullifier_compile(classifier, lr) # Compile
    classifier.build((None, 7, 7, 1024)) # and Build

    classifier.set_weights(model.get_weights()[-len(classifier.get_weights()):]) # Set current models weights

    return classifier

# get feature extrctor and classifier
def get_pp_fe_clsf(model): 

    pre_proc = preproc_model_create()
    f_ext = get_feature_extractor(model)
    classifier = get_classifier(model)
    
    return pre_proc, f_ext, classifier

def monte_carlo(predictions_bool, images, model):
    
    
    uncertainties = np.empty(images.shape[0])
    pre_proc, fe, classifier = get_pp_fe_clsf(model)
    for i, (image, pred) in tqdm(enumerate(zip(images, predictions_bool))):        
        uncertainties[i] = find_uncertainty(image, pred, pre_proc, fe, classifier).sum()

    return uncertainties

def wiggle_room(predictions):
    uncertainties = np.where(np.abs(predictions - cnf.threshold) <= cnf.wiggle_room, True, False).sum(1)
    return uncertainties

def training(image_budget, epochs, lr, X, Y, XT, YT, model_path, uncertainty=True, mc=True, budget_limit=False):
    """ uncertainty: Is true if model is trained on uncertain images 
        mc: Is true if monte carlo computation should be used, False indicates wiggleroom tech 
    """
    
    cum_bud = np.cumsum(image_budget)
    incr_epochs = int(epochs * cnf.fraction)

    model = hullifier_create(lr=lr)
    model.predict(X[:2])
    model = hullifier_clone(model)

    full_hist = []
    eps = []
    
    # The training dataset for the model
    training_X = X[:cum_bud[0]]
    training_Y = Y[:cum_bud[0]]
    h, e = train(
        model,
        training_X,
        training_Y,
        batch_size=50,
        epochs=epochs,
        validation_data=(XT,YT),
    )
    
    full_hist.append(h)
    eps.append(e)

    skips = 0
    for prev_cb, cb in zip(cum_bud, cum_bud[1:]):
        # "incremental" Slice of dataset that is used this iteration
        im_slice = slice(prev_cb, cb)

        predictions_bool, predictions = get_predictions_bool(model, X[im_slice])

        if mc:
            uncertainties = monte_carlo(predictions_bool, X[im_slice], model)
        else:
            uncertainties = wiggle_room(predictions)

        
        highest = np.argsort(uncertainties) # get sorted indices
        indices = highest[-budget_limit:] # Only keep indices inside budget, if budget is false all indices are kept
        mask = np.zeros_like(uncertainties, dtype=bool)
        mask[indices] = uncertainties[indices].astype(bool) # Set uncertai/n indices True

        # Mask is True for all images/indices that will be deleted

        if uncertainty: # Mask is default as True for uncertain so flip if necessary
            mask = np.invert(mask)

        new_X = np.delete(X[im_slice], mask, axis=0)
        new_Y = np.delete(Y[im_slice], mask, axis=0)
            
        print(f'Mask removed {np.sum(mask)} images. Kept {np.sum(np.invert(mask))} images. New X is shape: {new_X.shape}')
        eps.append(incr_epochs) 

        if new_X.shape[0] == 0: # If all images masked out
            skips += 1
            full_hist.append(incr_epochs)
            printe("Chaught skip!")
            continue
            
        training_X = np.concatenate((training_X, new_X), 0)
        training_Y = np.concatenate((training_Y, new_Y), 0)
        print(f'Now training on {training_X.shape} data')
        # train on only uncertain images
        h, e = train(
            model,
            training_X,
            training_Y,
            batch_size=50,
            epochs=incr_epochs,
            validation_data=(XT,YT),
        )
        full_hist.append(h)
        # eps.append(e)
        
    if skips:
        printw(f'Skipped {skips} batche(s) because no images were uncertain={uncertainty}')
        # input('Continue?')
    hullifier_save(model, model_path, lr=get_lr(model), total_epochs=np.sum(eps))
    return full_hist, eps





def incrementally_uncertainty_train(X, Y, XT, YT, lr, epochs, image_budget, budget=0):

    path='models/iuc_models/'+ ('any' if budget else 'budget') +'/mcu'
    full_hist_0, eps_0 = training(image_budget, epochs, lr, X, Y, XT, YT, path, uncertainty=True, mc=True, budget_limit=budget)
    eps = np.array(eps_0)
    full_hist_0 = history_merge(np.array(full_hist_0), eps, 'loss', 'binary_accuracy')
    
    if measure_time:
        printo(f'1st done after {t()} since start')

# ==========================================================================================================================================

    path='models/iuc_models/'+ ('any' if budget else 'budget') +'/mcc'
    full_hist_1, _ = training(image_budget, epochs, lr, X, Y, XT, YT, path, uncertainty=False, mc=True, budget_limit=budget)
    full_hist_1 = history_merge(np.array(full_hist_1), eps, 'loss', 'binary_accuracy')


    if measure_time:
        printo(f'2nd done after {t()} since start')

# ==========================================================================================================================================

    path='models/iuc_models/'+ ('any' if budget else 'budget') +'/tiu'
    full_hist_2, _ = training(image_budget, epochs, lr, X, Y, XT, YT, path, uncertainty=True, mc=False, budget_limit=budget)
    full_hist_2 = history_merge(np.array(full_hist_2), eps, 'loss', 'binary_accuracy')



    if measure_time:
        printo(f'3rd done after {t()} since start')

# ==========================================================================================================================================

    path='models/iuc_models/'+ ('any' if budget else 'budget') +'/tic'
    full_hist_3, _ = training(image_budget, epochs, lr, X, Y, XT, YT, path, uncertainty=False, mc=False, budget_limit=budget)
    full_hist_3 = history_merge(np.array(full_hist_3), eps, 'loss', 'binary_accuracy')

    # suptitle = f"Loss and accuracy graph for model trained incrementally and with certain images"
    # path = 'incr_crt_wiggle_trained.png'
    
    if measure_time:
        printo(f'4th done after {t()} since start')
    return eps, full_hist_0, full_hist_1, full_hist_2, full_hist_3
    
    
def regular_train(X, Y, XT, YT, lr, epochs, fn):
    # Train a new model the regular way
    model = hullifier_create(lr=lr)
    h, e = train(
        model,
        X,
        Y,
        batch_size=50,
        epochs=epochs,
        validation_data=(XT,YT),
    )
    
    h = history_merge([h],[e])
    e = np.array([e])

    save_regular('npy/mean_reg/', fn, e, h)
    printe(f'Finished training {fn} in {t()}')

def save_regular(path_to_dir, file_name, e, h):
    path = path_to_dir + file_name +'.npy'
    print(f'Trying to save values to {path}...')
    with open(path, 'wb') as f:
        np.save(f, e)
        np.save(f, h)

def main():
    # Load parameters from old model
    params = get_dict_from_file(cnf.model_path + 'params.txt')
    # load old model
    params.epochs = int(params.epochs)
    params.epochs += 5
    
    # Load all the old data
    X_original, Y_original = load_from_coco()
    for seed in range(10):
        X, Y = shuffle_data(X_original, Y_original, seed)
        # Original split
        X,Y,XT,YT = split_data(X,Y,float(params.v_split))    
        crt_bud = 35
        image_budget = np.array([
            800, 
            103, # 1
            100, # 2
            100, # 3
            100, # 4
            100, # 5
            100, # 6
            100, # 7
            100, # 8
            100  # 9
        ])
        fn = 'regular' + str(seed)
        regular_train(X,Y, XT,YT, float(params.lr),params.epochs, fn)


        path = 'npy/mean_iut/any/'+'iut'+str(seed)+'.npy'
        eps, fh0, fh1, fh2, fh3 = incrementally_uncertainty_train(X, Y, XT, YT, float(params.lr), params.epochs, image_budget)
        save_values(path, eps, fh0, fh1, fh2, fh3)
        printo(f'Seed {seed} "Any" training done {t()} since start')


        path = 'npy/mean_iut/budget/'+'iut'+str(seed)+'.npy'
        eps, fh0, fh1, fh2, fh3 = incrementally_uncertainty_train(X, Y, XT, YT, float(params.lr), params.epochs, image_budget, crt_bud)
        save_values(path, eps, fh0, fh1, fh2, fh3)
        printo(f'Seed {seed} "budget" training done {t()} since start')

if __name__ == "__main__":
    main()
    if measure_time:
        print(f'Total time used: {t()}')
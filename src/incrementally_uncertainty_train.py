import numpy as np
from tensorflow.keras.models import clone_model
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import config as cnf
from prints import printe, printo, printw, printc
from model import hullifier_load, hullifier_compile, hullifier_create, train
from data import load_from_coco, shuffle_data, split_data
from utils import get_dict_from_file, plot_history, find_uncertainty, history_merge, get_predictions_bool


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
def monte_carlo(predictions_bool, images, model):
    uncertainties = np.empty(images.shape[0])
    for i, (image, pred) in tqdm(enumerate(zip(images, predictions_bool))):
        uncertainties[i] = find_uncertainty(image, pred, model).sum()
    return uncertainties

def wiggle_room(predictions):
    uncertainties = np.where(np.abs(predictions - cnf.threshold) <= cnf.wiggle_room, True, False).sum(1)
    return uncertainties

def training(image_budget, epochs, lr, X, Y, XT, YT, uncertainty=True, mc=True, budget_limit=False):
    """ uncertainty: Is true if model is trained on uncertain images 
        mc: Is true if monte carlo computation should be used, False indicates wiggleroom tech 
    """
    
    cum_bud = np.cumsum(image_budget)
    incr_epochs = int(epochs * 0.2)

    model = hullifier_create(X[:3], lr=lr)    
    
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
        mask[indices] = True # Set uncertain indices True

        # Mask is True for all images/indices that will be deleted

        if uncertainty: # Mask is default as True for uncertain so flip if necessary
            mask = np.invert(mask)

        new_X = np.delete(X[im_slice], mask, axis=0)
        new_Y = np.delete(Y[im_slice], mask, axis=0)
            
        print(f'Mask removed {np.sum(mask)} images. Kept {np.sum(np.invert(mask))} images. New X is shape: {new_X.shape}')

        if new_X.shape[0] == 0: # All images masked out
            skips += 1
            printe("Chaught skip!")
            exit()
            # print('skipping')
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
        eps.append(e)
        
    if skips:
        printw(f'Skipped {skips} batche(s) because no images were uncertain={uncertainty}')
        input('Continue?')
    return full_hist, eps





def incrementally_uncertainty_train(X, Y, XT, YT, lr, epochs, image_budget):
    
    full_hist_0, eps_0 = training(image_budget, epochs, lr, X, Y, XT, YT, uncertainty=True, mc=True)
    eps = np.array(eps_0)
    full_hist_0 = history_merge(np.array(full_hist_0), eps, 'loss', 'binary_accuracy')

    
    if measure_time:
        printo(f'1st done after {t()} since start')

# ==========================================================================================================================================

    full_hist_1, _ = training(image_budget, epochs, lr, X, Y, XT, YT, uncertainty=False, mc=True)
    full_hist_1 = history_merge(np.array(full_hist_1), eps, 'loss', 'binary_accuracy')


    if measure_time:
        printo(f'2nd done after {t()} since start')

# ==========================================================================================================================================

    full_hist_2, _ = training(image_budget, epochs, lr, X, Y, XT, YT, uncertainty=True, mc=False)
    full_hist_2 = history_merge(np.array(full_hist_2), eps, 'loss', 'binary_accuracy')



    if measure_time:
        printo(f'3rd done after {t()} since start')

# ==========================================================================================================================================

    full_hist_3, _ = training(image_budget, epochs, lr, X, Y, XT, YT, uncertainty=False, mc=False)
    full_hist_3 = history_merge(np.array(full_hist_3), eps, 'loss', 'binary_accuracy')

    # suptitle = f"Loss and accuracy graph for model trained incrementally and with certain images"
    # path = 'incr_crt_wiggle_trained.png'
    
    if measure_time:
        printo(f'4th done after {t()} since start')

# ==========================================================================================================================================
    # fig, axs = plt.subplots(1)
    # total_epochs = plot_seperate_epochs(eps_3)
    
    
    # plot_history(axs[0], total_epochs, full_hist_0[:2], subfig_text='Loss function')
    # plot_history(axs[1], total_epochs, full_hist_0[2:], subfig_text='Binary accuracy')
    # plot_history(axs[0], total_epochs, full_hist_1[:2], subfig_text='Loss function')
    # plot_history(axs[1], total_epochs, full_hist_1[2:], subfig_text='Binary accuracy')

# ==========================================================================================================================================
    # fig, axs = plt.subplots(1)
    
    # total_epochs = plot_seperate_epochs(eps_3)
    
    
    # plot_history(axs[0], total_epochs, full_hist_0[:2], subfig_text='Loss function')
    # plot_history(axs[1], total_epochs, full_hist_0[2:], subfig_text='Binary accuracy')
    # plot_history(axs[0], total_epochs, full_hist_1[:2], subfig_text='Loss function')
    # plot_history(axs[1], total_epochs, full_hist_1[2:], subfig_text='Binary accuracy')
# ==========================================================================================================================================
    # plot all in same figure

    # fig.tight_layout()
    # fig.savefig('../out_imgs/incrementally_uncertain/'+'all_trained.png')
    # if measure_time:
        # printo(f'5th fig done after {t()} since start')

    with open('npy/'+'iut.npy', 'wb') as f:
        np.save(f, eps)
        np.save(f, full_hist_0)
        np.save(f, full_hist_1)
        np.save(f, full_hist_2)
        np.save(f, full_hist_3)
    
        
        

def main():
    # Load parameters from old model
    params = get_dict_from_file(cnf.model_path + 'params.txt')
    # load old model
    params.epochs = int(params.epochs)
    params.epochs += 5
    # params.epochs += 5
    # params.epochs += 5
    # params.epochs = 5
    # params.epochs = 1
    
    # Load all the old data
    X, Y = load_from_coco()
    X, Y = shuffle_data(X,Y)
    # Original split
    X,Y,XT,YT = split_data(X,Y,float(params.v_split))    

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

    incrementally_uncertainty_train(X, Y, XT, YT, float(params.lr), params.epochs, image_budget)


if __name__ == "__main__":
    main()
    if measure_time:
        print(f'Total time used: {t()}')
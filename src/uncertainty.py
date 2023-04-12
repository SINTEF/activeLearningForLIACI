import numpy as np
from tensorflow.keras.models import clone_model
from tensorflow import convert_to_tensor, expand_dims
import scipy.stats as stats
import time
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import config as cnf
from prints import printc
from model import hullifier_load, hullifier_compile
from utils import get_dict_from_file, get_predictions_bool
from data import load_from_coco, shuffle_data, split_data

measure_time = True
if measure_time:
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    printc(f'Started at: {current_time}')
    start = time.time()

def t():
    tm = int(time.time()-start)
    return f'{tm//60}m {tm%60}s'

# [x] Load old model
# [x] Clone old model
# [x] Set dropout layer to higher D/O rate for new model
# [x] Load old weights to new model
# [x] Load test data
# [x] predict(use __call__) with training=true for n (10 maybe?) images from test set
# [x] calculate mu and sigma
# [x] plot gauss
# [x] plot predictions along x axis
# [] 
# [] 
# [] 


def measure_uncertainty(model, X, Y, n_samples):
    b_pred, preds = get_predictions_bool(model, X)
    print('bp shape:', b_pred.shape)
    values = np.empty((X.shape[0], n_samples, Y.shape[1])) # shape (n_images, n_samples, 9)
    
    for i, im in enumerate(X):
        im = expand_dims(convert_to_tensor(im), 0)
        for k in range(n_samples):
            pred = model(im, training=True)[0]
            values[i, k] = pred
    

    value = values[0] # look at all predictions for a single image

    mu = np.mean(value, axis=0)
    var = np.var(value, axis=0)
    sigma = np.sqrt(var)

    fig, axs = plt.subplots(1)

    # Plot threshold
    axs.axvline(cnf.threshold, linestyle='dashed', linewidth=1,c='red', label='Threshold')
    for i, (m, s, v) in enumerate(zip(mu, sigma, var)): # Loops through all label's, mean, var and sigma
        if not b_pred[1][i]:
            continue
        x = np.linspace(m - 3*s, m + 3*s, 100) # Create X axis range
        axs.plot(x, stats.norm.pdf(x, m, s), label=f'Class {i},\n$\mu={np.round(m,2)}, \sigma^2={np.round(v,4)}$')

        color = plt.gca().lines[-1].get_color() # get color from last plot
        axs.scatter(m-2*s, stats.norm.pdf([m-2*s], m,s), linewidth=1, c=color, marker='x')
        axs.axvline(preds[0][i], c=color)
        
    axs.legend()
    axs.set_xlim([0,1])

    fig.savefig('../out_imgs/uncertainty/'+ 'image_1.png' )
def main():
    # Load parameters from old model
    params = get_dict_from_file(cnf.model_path + 'params.txt')
    # load old model
    hullifier = hullifier_load(cnf.model_path)
    # Set dropout layer to a new value
    hullifier.layers[-1].layers[1].rate = 0.2
    # New cloned model with fresh weights
    model = clone_model(hullifier)
    hullifier_compile(model, float(params.lr))
    # copy weights from old model to new
    model.set_weights(hullifier.get_weights())

    ## Now find uncertainty
    
    # Load all the old data
    X, Y = load_from_coco()
    X, Y = shuffle_data(X,Y)
    X,Y,XT,YT = split_data(X,Y,float(params.v_split))

    # Don't set to less than two
    n_unc_im = 3
    n_samples = 50

    measure_uncertainty(model, XT[:n_unc_im], YT[:n_unc_im], n_samples)





if __name__ == "__main__":
    main()
    if measure_time:
        print(f'Total time used: {t()}')

    pass
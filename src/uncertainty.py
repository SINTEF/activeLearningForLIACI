import numpy as np
from tensorflow.keras.models import clone_model

import config as cnf
from model import hullifier_load, hullifier_compile
from utils import get_dict_from_file
from data import load_from_coco, shuffle_data, split_data

def main():
    # Load parameters from old model
    params = get_dict_from_file(cnf.model_path + 'params.txt')
    # load old model
    hullifier = hullifier_load(cnf.model_path)
    # Set dropout layer to a new value
    hullifier.layers[-1].layers[1].rate = 0.5
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


    




    




    
        
    
    

if __name__ == "__main__":
    main()
    pass
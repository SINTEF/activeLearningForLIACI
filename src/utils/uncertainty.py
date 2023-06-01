from tensorflow import convert_to_tensor, expand_dims
from tensorflow.keras.models import Model
import numpy as np
from model import preproc_model_create, classifier_create, hullifier_compile
import utils.config as cnf



# Evaluates single image for monte carlo
def find_uncertainty(image, b_preds, pre_proc, feature_extractor, classifier):
    """ Return shape (n_labels) with uncertain labels as true  """

    image = expand_dims(convert_to_tensor(image), 0)
    image = pre_proc(image)
    features = feature_extractor(image, training=False) # Extract features

    values = np.empty((cnf.n_samples, cnf.n_labels))
    for k in range(cnf.n_samples): # collect samples from n_samples-subnetworks
        pred = classifier(features, training=True)[0]
        values[k] = pred

    mu = np.mean(values, axis=0)
    var = np.var(values, axis=0)
    sig = np.sqrt(var)

    below_thres = (mu - (2 * sig)) < cnf.threshold
    uncertain = b_preds & below_thres
    
    return uncertain



def get_feature_extractor(model):

    first_name = 'input_1'
    layer_name = 'conv_pw_13_relu'

    extractor = Model(model.layers[3].input, model.get_layer(layer_name).output)
    return extractor

if __name__ == "__main__":
    print(cnf.n_samples)
    # t(classifier_create)
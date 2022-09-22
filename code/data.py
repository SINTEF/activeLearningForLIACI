import matplotlib.pyplot as plt
from requests import post, get
from requests.auth import HTTPBasicAuth
from access_keys import api_key, ic_api, od_api, get_url
# from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from pycocotools.coco import COCO

from json import load

from sys import path
from os import getcwd

from prints import printw, printe, printo, printc

coco_file_path = getcwd() + '/../LIACi_dataset_pretty/coco-labels.json'


def load_im(file_name):
    path = getcwd() + '/../LIACi_dataset_pretty/images/' + file_name

    im = load_img(path, target_size=(224,224))
    im = img_to_array(im)
    im = np.uint8(im)
    im = preprocess_input(im)

    return im

def get_cats(path=coco_file_path):
    c = COCO(path)
    return c.getCatIds()

def get_hot_vec(ann_objs, n_cats):
    """ returns a multi-label hot vector """

    hv = np.zeros(n_cats)

    for a_obj in ann_objs:
        a = a_obj['category_id']
        # hv[a-1] += 1 # Should supervision be done like this?
        hv[a-1] = 1
        
def load_from_coco(n_imgs=1, coco_file=coco_file_path):

    c = COCO(coco_file)
    idxs = list(range(1, n_imgs + 1))
    images = []
    anns = []
    n_cats = len(get_cats())

    for idx in idxs:
        im_obj = c.loadImgs(idx)[0]
        im = load_im(im_obj['file_name'])
        images.append(im)
        
        im_an = c.getAnnIds(idxs)
        ann_objs = c.loadAnns(im_an)
        hv = get_hot_vec(ann_objs, n_cats)
        anns.append(hv)

    anns = np.array(anns)
    images = np.array(images)
    
    return images, anns

if __name__ == "__main__":
    get_data()
    pass

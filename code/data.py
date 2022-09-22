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


def retrieve_data():
    # [?iterationId][&tagIds][&orderBy][&take][&skip]

    
    # endpoint = 'southcentralus.api.cognitiveservies.azure.com'
    endpoint = 'southcentralus.api.cognitive.microsoft.com'
    url = get_url(endpoint, ic_api)
    payload = {
        # 'iterationId': None, 
        'tagIds': 'd01c9230-3ee1-4a88-8b9a-6bc7211b7ff8',
        # 'orderBy': None,
        # 'take': None,
        # 'skip': 0
        }
    headers = {"Training-key":  api_key}

    r = get(
        headers=headers, 
        params=payload,
        # auth=('apikey', api_key),
        url=url 
        )
    
    printw(r.url)
    print(r.status_code)
    print(r.content)
    
    image = r.content
    
    return image

def load_im(file_name):
    path = getcwd() + '/../LIACi_dataset_pretty/images/' + file_name

    im = load_img(path, target_size=(224,224))
    im = img_to_array(im)
    im = np.uint8(im)
    im = preprocess_input(im)

    return im

def get_hot_vec(ann_objs, n_cats):
    """ returns a multi-label hot vector """
    hv = np.zeros(n_cats)
    for a_obj in ann_objs:
        a = a_obj['category_id']
        # hv[a-1] += 1 # Should supervision be done like this?
        hv[a-1] = 1
        
def load_from_coco(coco_file, n_imgs=1):

    c = COCO(coco_file)
    idxs = list(range(1, n_imgs + 1))
    images = []
    anns = []
    n_cats = len(c.getCatIds())

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

def get_data():
    coco_file_path = getcwd() + '/../LIACi_dataset_pretty/coco-labels.json'

    X, Y = load_from_coco(coco_file_path, n_imgs=4)

    return X, Y

if __name__ == "__main__":
    get_data()
    pass

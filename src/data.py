import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
from pycocotools.coco import COCO

from os import getcwd

from prints import printw, printe, printo, printc

coco_file_path = getcwd() + '/../LIACi_dataset_pretty/coco-labels.json'
banned_labels = ['ship_hull']

def load_im(file_name, target_size=(224,224)):
    path = getcwd() + '/../LIACi_dataset_pretty/images/' + file_name

    if target_size:
        im = load_img(path, target_size=target_size)
    else:
        im = load_img(path)
    return np.uint8(im)

def pre_proc_img(im, target_shape=(224,224)):
    im = np.uint8(im)
    printo(type(im))
    im = cv2.resize(im, target_shape)
    if not im.dtype == np.uint8:
        printe(f'ERROR: dtype should be uint8, was {im.dtype}') 
        exit()
    if im.shape[0] == 1:
        printe(f'ERROR: shape is not correct, was {im.shape}') 
        exit()
    
    
    return im
def split_data(X,Y, split):
    
    if not X.shape[0] == Y.shape[0]:
        print('oopsie')
        exit()
    if 0 < split and split < 1:
        split = -int(np.ceil(X.shape[0] * split))

    x = X[:split] # train
    y = Y[:split] # train
    xt = X[split:] # test
    yt = Y[split:] # test
    

    return x, y, xt, yt

def get_cats(path=coco_file_path):
    c = COCO(path)
    return c.getCatIds()

# returns all label names
def get_cat_lab(path=coco_file_path):
    c = COCO(path)
    idx = c.getCatIds()
    c = c.loadCats(idx)
    return [ i['name'] for i in c if not i['name'] in banned_labels ]

def shuffle_data(X,Y,seed=0):
    rs = np.random.RandomState(seed)
    rs.shuffle(X)
    rs = np.random.RandomState(seed)
    rs.shuffle(Y)
    return X, Y
    
def get_hot_vec(ann_objs, n_cats):
    """ returns a multi-label hot vector """

    hv = np.zeros(n_cats)

    for a_obj in ann_objs:
        a = a_obj['category_id']
        if a == 10:
            continue
        hv[a-1] = 1
    return hv
        
def load_from_coco(n_imgs=0, coco_file=coco_file_path, target_size=(224,224)):
        
    c = COCO(coco_file)

    if n_imgs < 1:  # Use all imgs
        n_imgs = len(c.getImgIds()) 
    idxs = list(range(1, n_imgs + 1))
    images = []
    anns = []
    n_cats = len(get_cats())-1

    for idx in idxs:
        im_obj = c.loadImgs(idx)[0]
        im = load_im(im_obj['file_name'], target_size=target_size)
        images.append(im)
        
        im_an = c.getAnnIds(idx)
        ann_objs = c.loadAnns(im_an)
        hv = get_hot_vec(ann_objs, n_cats)
        anns.append(hv)
    
    anns = np.array(anns)
    images = np.array(images)
    printo(f'Loaded {n_imgs} images with annotations')
     
    return images, anns

if __name__ == "__main__":
    load_from_coco()
    pass

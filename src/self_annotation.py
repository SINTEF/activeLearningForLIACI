import cv2
import numpy as np
import utils.config as cnf
from json import dump, load
from os import path, mkdir
from shutil import rmtree
from prints import printw, printe
from PIL import Image

from tensorflow.keras.preprocessing.image import load_img

from data import get_cat_lab



def add_annotated_im(image, labels):
    """ 
    image: a numpy array of the image
    labels: should be a binary multi hot vector of the labels annotated in the image
    """
    with open(cnf.new_images_dir + 'labels.json', 'r+') as f:
        json_file = load(f)
        # images = json_file[images]

        if not len(json_file['images']):
            num = '0'
        else:
            l = sorted(list(json_file['images']))
            num = str(int(l[-1].split('_')[1].split('.jpg')[0]) + 1) # Find new unique image id
            
        name = 'image_' + num.zfill(4) + '.jpg'
        print("Name of new image:",name)
        json_file['images'][name] = {
            'hot_vec_b' : str(labels),
        }
        printe(json_file['images'])
        im = Image.fromarray(image)
        f.seek(0,0)
        
        dump(json_file, f, indent=4)
        f.truncate()
    im.save(cnf.new_images_dir + "images/" + name)

def create_label_file():
    """ 
        Call this function to create a json file to store the label formats, calling this function
        when a file already exist will promt the user to delete the old file by typing 'yes'
    """
    if path.exists(cnf.new_images_dir):
        if input(f"{cnf.new_images_dir} already exists, are you sure you want to overwrite it? [yes]: ") == 'yes':
            printw('Overwriting...')
            rmtree(cnf.new_images_dir)
            mkdir(cnf.new_images_dir)
        else:
            printw('Exiting without overwriting...')
            return
    else:
        mkdir(cnf.new_images_dir)
        print(f'made new dir: {cnf.new_images_dir}')

    with open(cnf.new_images_dir + 'labels.json', 'w') as f:
        labels = get_cat_lab()
        # print(d)
        json_file = {
            'labels' : {k: v for k, v in enumerate(labels)},
            'images' : {}
        }

        dump(json_file, f, indent=4)
        
    if path.exists(cnf.new_images_dir + 'images/'):
        rmtree(cnf.new_images_dir + 'images/')
    mkdir(cnf.new_images_dir + 'images/')

def load_from_user_ann(target_size=(224,224)):

    images = []
    anns = []
    with open(cnf.new_images_dir + 'labels.json', 'r') as f:
        image_d = load(f)
        # print(image_d)

    for i in image_d['images']:
        images.append(np.uint8(load_img(cnf.new_images_dir + 'images/' + i, target_size=target_size))) # open image, cast to uint8 and add to image list
        
        anns.append(np.fromstring(image_d['images'][i]['hot_vec_b'][1:-1],sep=', ',dtype=bool)) # Turn the hot vector into a binary np array and append to list

    return np.array(images), np.array(anns)


if __name__ == '__main__':
    pass
    create_label_file()    
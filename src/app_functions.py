import re
import datetime
import numpy as np
import cv2

from dash_player import DashPlayer
from dash import html
import dash_bootstrap_components as dbc
import dash_daq as daq

import plotly.express as px
import matplotlib.pyplot as plt
from base64 import b64decode
from tqdm import tqdm
import os
from time import time 

from utils.file import get_dict_from_file
from utils.uncertainty import find_uncertainty
from utils.model import train, hullifier_load
import utils.config as cnf

from self_annotation import add_annotated_im, load_from_user_ann
from data import get_cat_lab, load_from_coco, shuffle_data, split_data
from prints import printo, printe, printw, printc

def generate_label_alerts():
                                    # html.I(className="bi bi-info-circle-fill me-2"),
    labs = get_cat_lab()
    alerts = []
    for i, l in enumerate(labs):
        label_str = f"{i}: {l} - p:"
        alert = dbc.Alert(
            children=[
                label_str, 
                html.Div(
                    id=f'label-{i}-p', 
                    children='n/a',
                    style={
                        'margin-left': 'auto', 
                        'margin-right': 0
                    }
                )
            ],
            id=f'label-{i}',
            style={'display': 'inline-block'},
            color="primary",
            className="d-flex align-items-center",
        )

        switch = daq.ToggleSwitch(
            id=f'switch-{i}',
            value=False,
            style={'display':'inline-block'},
            className="d-flex align-items-center",

        )

        alerts.append(alert)
        alerts.append(switch)
    return alerts


class AppFunc:
    def __init__(self, model_path=cnf.model_path):
        self.model = hullifier_load(model_path, resize=False)
        self.labels = get_cat_lab()
        self.tmp_path = cnf.tmp_dir

        # Is set when video is uploaded
        self.predictions_bool = None
        self.predictions = None
        self.vid = None
        self.duration = None #
        self.tnf = None
        self.frames = False # True if video has been parsed and predicted
        self.pif = None # previous image frame

    
    # This is retired and not in use
    def predict_part(self, tnf):
        raise Exception("This is retired and not in use")
        
        frames = np.empty((tnf, 224, 224, 3)).astype(np.uint8) # hardcoded 224,224,3 as the image size is known(for now)

        print('Fetching images from video...')
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0) # make sure video is set to idx=0
        for i in tqdm(range(tnf)):
            succ, im = self.vid.read()
            if not succ:
                raise Exception("Can't parse video image")
            im = cv2.resize(im, (224,224)).astype(np.uint8)
            frames[i,:,:,:] = im

        printo('Done fetching...')        

        print('Preprocessing & predicting...')
        predictions = self.model.predict(frames)
        predictions_bool = np.where(cnf.threshold <= predictions, True, False)
        
        return frames, predictions_bool, predictions

    def predict_part_slow(self, tnf):
        nmf = cnf.n_mem_frames
        mem_frames = np.empty((nmf, 224, 224, 3))
        predictions = np.zeros((tnf, len(self.labels)))
        
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        start = time()
        j = 0
        print(f'Fetching, preprocessing and predicting {tnf} images from video...')
        for i in tqdm(range(tnf)):
            succ, im = self.vid.read()
            if not succ:
                im = np.zeros((224,224,3))
                # raise Exception(f"Can't parse image from video on frame {i}")

            im = cv2.resize(im,(224,224)).astype(np.uint8)            
            im = np.expand_dims(im, axis=0)
            mem_frames[j,:,:,:] = im
            j += 1

            
            if j == nmf: # Predict on frames in memory
                print(f'Predicting images [{i+1-nmf}, {i}]')
                predictions[i+1-nmf:i+1] = self.model.predict(mem_frames)
                j = 0
                
            elif i+1 == tnf: # use predict only on the leftover part (happens on last run)
                print(f'Predicting images [{i+1-j}, {tnf}]')
                predictions[i+1-j:] = self.model.predict(mem_frames[:j])
            
        end = time()
        printo(f'time used: {end-start}')
        
        predictions_bool = np.where(cnf.threshold <= predictions, True, False)

        return True, predictions_bool, predictions
        
    def get_fig_path(self):
        return self.fig_path
    def create_im_from_pred(self, pred, write=False):

        figaro = px.imshow(pred.transpose(), aspect=100)
        y_ticks = np.arange(len(self.labels))
        figaro.update_layout(
            yaxis=dict(
                tickmode = 'array',
                tickvals = y_ticks,
                ticktext = self.labels,
            ),
            xaxis_title="frame number",
        ).update_coloraxes(showscale=False)

        if write:
            figaro.write_image(self.fig_path)
        print(f"Figaro {type(figaro)}")
        return figaro

    def find_uncertainties(self):
        uncertainties = np.empty(self.tnf, self.labels.shape[0])
        
        s = time()
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0) # make sure video is set to idx=0
        for i in tqdm(range(self.tnf, step=int(self.fps))):
            self.vid.self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            succ, im = self.vid.read()
            if not succ or self.predictions_bool[i].sum() == 0:
                uncertainties[i] = np.zeros(self.labels.shape[0])
            else:
                im = cv2.resize(im, (224,224)).astype(np.uint8)
                uncertainties[i] = find_uncertainty(im, self.predictions_bool[i], self.model)

        tm = int(time()-s)
        printc(f'Used {tm//60}m {tm%60}s')

        return uncertainties


    def create_timelines(self, path):
        self.tmp_path = path

        
        self.fig_path = self.tmp_path.rsplit('.', 1)[0] + '_graph.pdf'
        
        self.vid = cv2.VideoCapture(self.tmp_path)

        self.tnf = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)
        self.duration = self.tnf / self.fps

        succ, image = self.vid.read()
        if not succ:
            raise Exception("Can't parse video")

        self.frames, self.predictions_bool, self.predictions  = self.predict_part_slow(self.tnf)
    
        figaro = self.create_im_from_pred(self.predictions_bool, True)

        # Check if prediction is too close to threshold for activation
        # uncertainties = np.where(np.abs(self.predictions - cnf.threshold) <= cnf.wiggle_room, True, False)
        uncertainties = self.find_uncertainties()
        figaro_unsure = self.create_im_from_pred(uncertainties)
        
        return figaro, figaro_unsure
    
    def store_image(self, frame, labels):
        
        print(f'saving frame: {frame}')
        if not self.frames:
            printe('cant capture frame yet')
            return
        succ, im = self.get_img_from_idx(int(frame)) 
        labels = [ int(l) for l in labels ]

        add_annotated_im(im, labels)
        
    def get_img_from_idx(self, idx):
        if not 0 <= idx < self.tnf:
            raise Exception(f'Bad index, was given: {idx}')

        out = self.vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
        succ, im = self.vid.read()
        if not succ:
            return succ, np.zeros((224,224,3))

        return succ, im
        
    def incremental_train(self):
        return
        params = get_dict_from_file(cnf.model_path + 'params.txt')

        ef = 1/5 # Epoch fraction
        seed = 0

        epochs = int(int(params['epochs']) * ef)

        X1, Y1 = load_from_coco()
        X2, Y2 = load_from_user_ann()

        X = np.concatenate((X1,X2))
        Y = np.concatenate((Y1,Y2))
        
        X, Y = shuffle_data(X,Y,seed)
        X, Y , XT, YT = split_data(X, Y, 0.2)

        # print('check prediction')
        a = self.model.evaluate(X,Y)
        # print(a)

        h, e = train(self.model, X, Y, (XT, YT))
        summarize_diagnostics(h,e)
        # print(h.history)
        # print('here')
        # print(np.round(h.history['binary_accuracy'], 3))
        # print(np.round(h.history['val_binary_accuracy'], 3))
        # print(h.history['val_binary_accuracy'])
        # summarize_diagnostics(h,e,path)
        



if __name__ == '__main__':
    pass
    path = '../videoplayback.mp4'
    af = AppFunc()
    af.create_timelines(path)
    af.get_img_from_idx(0)



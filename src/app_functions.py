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

from utils import get_dict_from_file
from self_annotation import add_annotated_im, load_from_user_ann
from data import get_cat_lab, load_from_coco, shuffle_data, split_data
from model import hullifier_load, train
from prints import printo, printe
import config as cnf


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
        self.frames = [] # np arr of all frames
        self.pif = None # previous image frame

    
    def predict_part(self, start, stop=None):
        if stop == None:
            stop = start
            start = 0
        
        frames = []

        self.vid.set(cv2.CAP_PROP_POS_FRAMES, start)
        print('Fetching images from video...')
        for i in tqdm(range(start, stop)):
            succ,im = self.vid.read()
            if not succ:
                raise Exception("Can't parse video image")
                # break
            im = cv2.resize(im, (224,224))
            frames.append(im)

        printo('Done fetching...')
            
        frames = np.array(frames)
            
        print('Preprocessing & predicting...')
        predictions = self.model.predict(frames)
        predictions_bool = np.where(cnf.threshold <=predictions, True, False)
        
        return frames, predictions_bool, predictions

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

        return figaro
        
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

        self.frames, self.predictions_bool, self.predictions  = self.predict_part(self.tnf)
        figaro = self.create_im_from_pred(self.predictions_bool, True)

        # Check if prediction is too close to threshold for activation
        pred_unsure = np.where(np.abs(self.predictions - cnf.threshold) <= cnf.wiggle_room, True, False)
        figaro_unsure = self.create_im_from_pred(pred_unsure)
        
        return figaro, figaro_unsure
    
    def store_image(self, frame, labels):
        
        print(f'saving frame: {frame}')
        if not len(self.frames) or not type(frame) == int :
            printe('cant capture frame yet')
            return
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
        succ,im = self.vid.read()
        labels = [ int(l) for l in labels ]

        add_annotated_im(im, labels)

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
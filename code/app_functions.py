from dash import html
import datetime
from dash_player import DashPlayer
import numpy as np
import cv2
from data import get_cat_lab
import matplotlib.pyplot as plt
from data import pre_proc_img
from model import mobilenet_create, mobilenet_create_def, model_load, model_predict
from typing import overload
from multipledispatch import dispatch
import dash_bootstrap_components as dbc
from tqdm import tqdm
from mpld3 import fig_to_html



def generate_label_alerts():
                                    # html.I(className="bi bi-info-circle-fill me-2"),
    labs = get_cat_lab()
    alerts = []
    for i, l in enumerate(labs):
        alert = dbc.Alert(
            children=f"{i}: {l}",
            id=f'label-{i}',
            style={'display': 'inline-block'},
            color="primary",
            className="d-flex align-items-center",
        )
        alerts.append(alert)
    return alerts


class AppFunc:
    def __init__(self):
        self.model = model_load()
        self.mb = mobilenet_create()
        self.labels = get_cat_lab()
        # self.mbd = mobilenet_create_def()

        # Is set when video is uploaded
        self.frames = None
        self.predictions = None
        self.vid = None
        self.duration = None #
        self.tnf = None
        self.frames = [] # np arr of all frames
        self.predictions = None
        self.available_frames = None
        self.pif = None # previous image frame

    def model_predict(self, frames):
        return model_predict(self.mb, self.model, frames)        
    
    def predict_part(self, start, stop=None, step=1):
        if stop == None:
            stop = start
            start = 0
        
        frames = []
        # self.available_frames = {}

        print('Preprocessing images...')
        for i in tqdm(range(start, stop, step)):
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            succ,im = self.vid.read()
            if not succ:
                raise Exception("Can't parse video image")
            im = pre_proc_img(im)
            frames.append(im)
            # self.available_frames[i] = True # add functionality to not process all images
            
        frames = np.array(frames)
        
        print('Predicting...')
        X = self.model_predict(frames)
        predictions = np.where(0.5<=X, 1, 0)
        return frames, predictions


    def create_timeline(self, url, dur):

        # self.vid = cv2.VideoCapture(url)
        
        self.vid = cv2.VideoCapture('assets/videoplayback.mp4') 
        succ, image = self.vid.read()
        if not succ:
            raise Exception("Can't parse video")

        self.tnf = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames
        self.duration = dur
        self.fps = self.tnf / dur
        
        self.frames, self.predictions = self.predict_part(self.tnf, step=25)
        
        
        fig, ax = plt.subplots(figsize=(10,3))
        ax.imshow(
            self.predictions.transpose(),
            interpolation=None
        )
        

        plt.tight_layout()
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Labels')
        ax.set_title('Videos classes')
        
                
        return fig_to_html(fig)
        


if __name__ == '__main__':
    pass
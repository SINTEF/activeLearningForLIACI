import re
from dash import html
import datetime
from dash_player import DashPlayer
import numpy as np
import cv2
from data import get_cat_lab
import matplotlib.pyplot as plt
from model import hullifier_load
import dash_bootstrap_components as dbc
from tqdm import tqdm
import plotly.express as px
from base64 import b64decode
import os

from prints import printo, printe



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
        self.model = hullifier_load(resize=False)
        self.labels = get_cat_lab()
        self.tmp_path = 'tmp/tmp_video.avi'

        # Is set when video is uploaded
        self.frames = None
        self.predictions = None
        self.vid = None
        self.duration = None #
        self.tnf = None
        self.frames = [] # np arr of all frames
        self.predictions = None
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
        X = self.model.predict(frames)
        predictions = np.where(0.5<=X, 1, 0)
        return frames, predictions

    def tmp_file_rm(self):
        if os.path.isfile(self.tmp_path):
            os.remove(self.tmp_path)
        else:
            printe("Error: %s file not found" % self.tmp_path)

    def tmp_file_save(self, url):
        with open(self.tmp_path, "wb") as fh:
            fh.write(b64decode(url.split(',')[1]))

    def create_timeline(self, url, dur):
        self.tmp_file_save(url)
            
        self.vid = cv2.VideoCapture(self.tmp_path)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)
        self.duration = dur

        succ, image = self.vid.read()
        if not succ:
            raise Exception("Can't parse video")

        self.tnf = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames
        self.frames, self.predictions = self.predict_part(self.tnf)
        self.tmp_file_rm()
        
        figaro = px.imshow(self.predictions.transpose(), aspect=100)
        y_ticks = np.arange(len(self.labels))
        figaro.update_layout(
            yaxis=dict(
                tickmode = 'array',
                tickvals = y_ticks,
                ticktext = self.labels,
            ),
            xaxis_title="frame number",
        ).update_coloraxes(showscale=False)
        
        # ax.set_title('Videos classes')
        return figaro
        


if __name__ == '__main__':
    pass
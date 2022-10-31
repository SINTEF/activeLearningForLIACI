from dash.dependencies import Input, Output, State
import cv2
from prints import printo
from dash.exceptions import PreventUpdate
import numpy as np


        
def get_callbacks(app, af):
    
    @app.callback(
        Output('video-player', 'url'),
        Output('video-filename', 'children'),
        Input('upload-video', 'contents'),
        State('upload-video', 'filename'),
    )
    def set_video(content, filename):
        if not content:
            raise PreventUpdate("no content to update")
        return content, filename

    @app.callback(
        Output('timeline', 'srcDoc'),
        Input('video-player','duration'),
        State('video-player','url'),
        
    )
    def create_timeline(dur, url):
        if not dur:
            raise PreventUpdate("No video duration")
        if not url:
            raise PreventUpdate("No video URL")
        
        print('Creating timeline...')
        
        timeline_fig = af.create_timeline(url, dur)
        printo('Timeline created')
        return timeline_fig
        
    @app.callback(
        Output('test-h', 'children'),
        Output('label-0', 'color'),
        Output('label-1', 'color'),
        Output('label-2', 'color'),
        Output('label-3', 'color'),
        Output('label-4', 'color'),
        Output('label-5', 'color'),
        Output('label-6', 'color'),
        Output('label-7', 'color'),
        Output('label-8', 'color'),
        Input('video-player', 'currentTime'),
    )
    def update_alerts(currentTime):
        if not currentTime or not af.fps or not af.tnf:
            raise PreventUpdate("Can't update alerts")

        frame = int((currentTime * af.fps))
        if not frame < af.tnf:
            raise PreventUpdate("Can't update alerts")
        pred = af.predictions[frame]
        l0, l1, l2, l3, l4, l5, l6, l7, l8 = np.where(pred, 'success', 'danger')
        return frame, l0, l1, l2, l3, l4, l5, l6, l7, l8
        


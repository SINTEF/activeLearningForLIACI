from dash.dependencies import Input, Output, State
import cv2
from prints import printe, printo
from dash.exceptions import PreventUpdate
from dash import dcc
import numpy as np
import dash_uploader as du
import base64
from dash import html
import config as cnf

def get_callbacks(app, af):
    
    @app.callback(
        Output('video-player', 'url'),
        Output('video-filename', 'children'),
        Output('time-line', 'figure'),
        Output('loading', 'children'),
        Input('upload-video', 'isCompleted'),
        State('upload-video', 'fileNames'),
        State('upload-video', 'upload_id'),
    )
    def set_video(iscompleted, filename, upload_id):
        if not iscompleted: 
            raise PreventUpdate("no content to update")
        path = cnf.tmp_dir + upload_id + '/' + filename[0]        

        with open(path, "rb") as videoFile:
            video = "data:video/mp4;base64," +  base64.b64encode(videoFile.read()).decode('ascii')

        timeline = af.create_timeline(path)

        # return None, filename[0], timeline, ""
        return video, filename[0], timeline, ""

    @app.callback(
        Output("download", "data"), 
        [Input("btn", "n_clicks")], 
        State('time-line', 'figure'),
        prevent_initial_call=True
    )
    def download_fig(n_clicks, figure):
        printo(f"n{n_clicks} and f{type(figure)}")
        if not n_clicks or not figure:
            raise PreventUpdate("Can't download figure right now")
        f_path = af.get_fig_path()
        return dcc.send_file(f_path)
        
    @app.callback(
        Output('curr-frame', 'children'),
        Output('label-0', 'color'),
        Output('label-1', 'color'),
        Output('label-2', 'color'),
        Output('label-3', 'color'),
        Output('label-4', 'color'),
        Output('label-5', 'color'),
        Output('label-6', 'color'),
        Output('label-7', 'color'),
        Output('label-8', 'color'),
        Input('inter', 'n_intervals'),
        State('video-player', 'currentTime'),
    )
    def update_alerts(inter, currentTime):
        if not currentTime or not af.fps or not af.tnf:
            raise PreventUpdate("Can't update alerts")

        frame = int((currentTime * af.fps))
        if not frame < af.tnf:
            raise PreventUpdate("Can't update alerts")

        pred = af.predictions[frame]
        labs = tuple(np.where(pred, 'success', 'danger'))
        print('retfranes')
        return (frame,) + labs
        
    @app.callback(
        Output('video-player', 'seekTo'),
        Input('time-line', 'clickData'),
        prevent_initial_call=True
    ) 
    def get_graph_click(clickData):
        frame = clickData['points'][0]['x']
        return frame / af.tnf

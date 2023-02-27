from dash.dependencies import Input, Output, State
from dash import html, ctx
from dash_canvas.utils import array_to_data_url
from dash.exceptions import PreventUpdate
from dash import dcc
import dash_uploader as du

import cv2
import numpy as np
import base64

from prints import printe, printo
import config as cnf

def get_callbacks(app, af):
    
    @app.callback(
        Output('video-player', 'url'),
        Output('video-filename', 'children'),
        Output('time-line', 'figure'),
        Output('time-line-unsure', 'figure'),
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

        prediction_timeline, pred_unsure_timeline = af.create_timelines(path)


        # return None, filename[0], prediction_timeline, ""
        return video, filename[0], prediction_timeline, pred_unsure_timeline, ""

    @app.callback(
        Output("download", "data"), 
        Input("btn", "n_clicks"), 
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
        Output('label-0-p', 'children'),
        Output('label-1-p', 'children'),
        Output('label-2-p', 'children'),
        Output('label-3-p', 'children'),
        Output('label-4-p', 'children'),
        Output('label-5-p', 'children'),
        Output('label-6-p', 'children'),
        Output('label-7-p', 'children'),
        Output('label-8-p', 'children'),
        Input('inter', 'n_intervals'),
        State('video-player', 'currentTime'),
    )
    def update_alerts(inter, currentTime):
        if not currentTime or not af.fps or not af.tnf:
            raise PreventUpdate("Can't update alerts")

        frame = int((currentTime * af.fps))
        if not frame < af.tnf:
            raise PreventUpdate("Can't update alerts")

        pred_bool = af.predictions_bool[frame]
        pred = af.predictions[frame]
        
        labs = tuple(np.where(pred_bool, 'success', 'danger'))
        pred = np.round(pred,3).astype('U')
        # print(pred)
        return (frame,) + labs + tuple(pred)
        
    @app.callback(
        Output('hidden-div','children'), # Add some thing that says iamge was saved
        Input('submit-annotation', 'n_clicks'),
        State('curr-frame', 'children'),
        State('label-alerts', 'children'),
        prevent_initial_call=True
    )
    def store_image(clicks, frame, children):
        print(f'Button is clicked {clicks} times at frame {frame}')
        labels = []
        for child in children:
            if child['type'] == "ToggleSwitch": # check type
                labels.append(child['props']['value'])
        af.store_image(frame, labels)
        
        return 
        

    @app.callback(
        Output('video-player', 'seekTo'),
        Output('label-alerts', 'children'),
        Input('time-line', 'clickData'),
        Input('time-line-unsure', 'clickData'),
        State('label-alerts', 'children'),
        prevent_initial_call=True
    ) 
    def get_graph_click(annotations, uncertainties, children):
        if ctx.triggered_id == 'time-line':
            clickData = annotations
        elif ctx.triggered_id == 'time-line-unsure':
            clickData = uncertainties
        
        frame = clickData['points'][0]['x']
        j = 0
        for i, child in enumerate(children):
            if child['type'] == "ToggleSwitch": # check type
                children[i]['props']['value'] = af.predictions_bool[frame][j]
                j += 1

        print("frame:",frame, frame / af.tnf)
        return (frame / af.tnf), children

    @app.callback(
        Output('hidden-div-upd','children'), # Add some thing that says iamge was saved
        Input('ud-model','n_clicks'),
        prevent_initial_call=True

    )
    def update_model(clicks):
        af.incremental_train()
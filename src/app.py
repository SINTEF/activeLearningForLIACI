from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_player import DashPlayer
import dash_uploader as du

from callbacks import get_callbacks
from os import mkdir, path
from shutil import rmtree

from self_annotation import create_label_file
from app_functions import AppFunc, generate_label_alerts
import utils.config as cnf
from utils.utils import open_video_b64


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]

app = Dash(__name__, external_stylesheets=external_stylesheets, title='The Hullifier')

du.configure_upload(app, cnf.tmp_dir)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(
    children=[
        dbc.Card(
            dbc.CardBody(
                id='video-card',
                children=[
                    html.Div(
                        id='label-alerts',
                        children=generate_label_alerts(),
                        style={'display':'inline-block'},
                    ),
                    # DashPlayer(
                    #     id='video-player',
                    #     url=None,
                    #     controls=True,
                    #     muted = True,
                    #     style={'width':'500px','display': 'inline-block', }#'right': '100px'}
                    # ),
                    html.Img(
                    # dcc.Graph(
                        id='vid-img',
                        # style={'width':'1500px','display': 'inline-block', }
                    ),
                    dbc.Button(
                        'submit',
                        id='submit-annotation',
                        style={'display':'inline-block'}
                    ),
                    html.H2("frame", id='curr-frame'),
                    html.H2(
                        'Model annotations',
                        style={'width':'100%'}
                    ),
                    dcc.Graph(
                        id="time-line", 
                        style={'width': '100%', 
                        'height' : '290px'
                        },
                    ),
                    dcc.Download(id='download'),
                    html.Button('Download', id='btn'),
                    html.H2(
                        'Uncertain labels',
                        style={'width':'100%'}
                    ),
                    dcc.Graph(
                        id="time-line-unsure", 
                        style={'width': '100%', 
                        'height' : '290px'
                        },
                    ),
                    html.H5('Filename',id='video-filename'),
                    dcc.Loading(
                        id="loading",
                        type="default",
                        fullscreen=True,
                        children=html.Div(id="loading-output-1", style={'display':'none'}),
                    ),
                    
                ]
            )
        ),
        du.Upload(
            id='upload-video',
            filetypes=['mp4'],
            max_file_size=1024*10, # a*b -> b represents amounts of GB
        ),
        html.Button('Retrain model', id='ud-model'),
        html.Div(id='hidden-div-upd', children=[], style={'display':'none'}),
        dcc.Interval(
            id='inter', 
            interval=1000,
            disabled=True,
        ),
        html.Div(id='hidden-div', children=[], style={'display':'none'})

    ]
)

af = AppFunc()
get_callbacks(app, af)

if __name__ == '__main__':
    if path.isdir(cnf.tmp_dir):
        rmtree(cnf.tmp_dir) 
    if not path.isdir(cnf.new_images_dir): 
        create_label_file()
    mkdir(cnf.tmp_dir)

    app.run_server(debug=True)
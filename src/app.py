from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_player import DashPlayer
import dash_uploader as du

from callbacks import get_callbacks
from os import mkdir, path
from shutil import rmtree

from app_functions import AppFunc, generate_label_alerts, generate_label_switches
import config as cnf


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
                    DashPlayer(
                        id='video-player',
                        url=None,
                        controls=True,
                        muted = True,
                        style={'width':'500px','display': 'inline-block', }#'right': '100px'}
                    ),
                    dbc.Button(
                        'submit',
                        id='submit-annotation',
                        style={'display':'inline-block'}
                    ),
                    # html.Div(
                    #     id='submit-tags',
                    #     children=generate_label_switches()
                    # ),
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
            # style={"font size":"+2"}
        ),
        dcc.Interval(
            id='inter', 
            interval=1000, 
        ),
        html.Div(id='hidden-div', children=[], style={'display':'none'})

    ]
)

af = AppFunc()
get_callbacks(app, af)

if __name__ == '__main__':
    if path.isdir(cnf.tmp_dir):
        rmtree(cnf.tmp_dir) 
    mkdir(cnf.tmp_dir)

    app.run_server(debug=True)
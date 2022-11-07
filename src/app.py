# from distutils.log import debug
from dash.dependencies import Input, Output, State
from dash import Dash, html, dcc, get_asset_url
import dash_bootstrap_components as dbc
from dash_player import DashPlayer
from callbacks import get_callbacks

import plotly.express as px
import pandas as pd
from PIL import Image
from app_functions import AppFunc, generate_label_alerts



img_dir = 'assets/image_0002.jpg'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]

app = Dash(__name__, external_stylesheets=external_stylesheets)

# app.css.append_css({'external_url': 'reset.css'})
# app.server.static_folder = ''

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
                        # playbackRate=2.0,
                        controls=True,
                        muted = True,
                        style={'width':'50%','display': 'inline-block', 'right': '100px'}
                    ),
                    dcc.Graph(
                        id="time-line", 
                        style={'width': '50%', 
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
        dcc.Upload(
            id='upload-video',
            accept='.mp4',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ],
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            )
        ),
        html.H2("frame", id='test-h'),
        html.Div(id='hidden-div', children=[], style={'display':'none'})
    ]
)

af = AppFunc()
get_callbacks(app, af)

if __name__ == '__main__':
    app.run_server(debug=True)
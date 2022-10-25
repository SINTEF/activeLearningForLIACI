# from distutils.log import debug
from dash.dependencies import Input, Output, State
from dash import Dash, html, dcc, get_asset_url

import plotly.express as px
import pandas as pd
from PIL import Image
from app_functions import parse_upload, VideoStream

# from model import model_load
vid = VideoStream()

img_dir = 'assets/image_0002.jpg'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# app.css.append_css({'external_url': 'reset.css'})
# app.server.static_folder = ''

# model = model_load()
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
    # html.H1(children="hello Dash",style={'textAlign': 'center','color': colors['text']}),
    # html.Div(children=''' Dash: Al web app frema''',style={'textAlign': 'center','color': colors['text']}),
    # dcc.Graph(id='example-graph',figure=fig),

    
    # html.Img(src=img_dir),


    # html.Img(src=get_asset_url('images/image_0008.jpg')),
    html.Div(
        id='output-video-uploa',
        children=[html.Video(
            controls=True,
            id='movie_player',
            # src='https://www.youtube.com/watch?v=mbcST7qJDvY'
            # src='assets/videoplayback.mp4',
            style={
                'width':'50%'
            }
        ),
        html.H5('Filename'),
        html.H6('Da'),
        ]
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
    html.Div(id='output-video-upload')
    ]
)


@app.callback(
    Output('output-video-uploa', 'children'),
    Input('upload-video', 'contents'),
    State('upload-video', 'filename'),
    State('upload-video', 'last_modified')
)
def update_output(content, filename, date):
    children = [parse_upload(content, filename, date)]
    return children

if __name__ == '__main__':
    app.run_server(debug=True)